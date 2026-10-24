import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import (
    EarlyStopping,
    load_checkpoint,
    load_config,
    load_vocab,
    save_checkpoint,
)
from src.dataset import Flickr30kDataset, collate_fn
from src.transform import get_train_transform, get_eval_transform
from src.model import CaptionModel


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for images, captions, lengths in progress:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        # Forward pass through ViT-Transformer
        # outputs: (B, seq_len - 1, vocab_size)
        outputs, _ = model(images, captions, lengths)
        targets = captions[:, 1:]  # (B, seq_len - 1)

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Track non-padding tokens
        batch_tokens = (targets != criterion.ignore_index).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += max(batch_tokens, 1)
        progress.set_postfix(loss=loss.item())

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(loader, desc="Validating", leave=False)
    for images, captions, lengths in progress:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        outputs, _ = model(images, captions, lengths)
        targets = captions[:, 1:]

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1),
        )

        batch_tokens = (targets != criterion.ignore_index).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += max(batch_tokens, 1)
        progress.set_postfix(loss=loss.item())

    return total_loss / max(total_tokens, 1)


def main(config, config_dir, resume: bool = False):
    torch.manual_seed(config["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not resume:
        resume = config.get("train", {}).get("resume", False)

    vocab = load_vocab(config_dir / config["paths"]["vocab"])
    print(f"Vocab size: {len(vocab)}")

    train_set = Flickr30kDataset(
        image_path=config_dir / config["paths"]["images"],
        split_path=config_dir / config["paths"]["train_json"],
        vocab=vocab,
        transform=get_train_transform(config["data"]["image_size"]),
    )
    val_set = Flickr30kDataset(
        image_path=config_dir / config["paths"]["images"],
        split_path=config_dir / config["paths"]["val_json"],
        vocab=vocab,
        transform=get_eval_transform(config["data"]["image_size"]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["train"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["train"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )

    model_config = {
        "d_model": config["model"].get("d_model", 512),
        "nhead": config["model"].get("nhead", 8),
        "num_decoder_layers": config["model"].get("num_decoder_layers", 3),
        "dim_feedforward": config["model"].get("dim_feedforward", 2048),
        "dropout": config["model"].get("dropout", 0.1),
        "tie_weights": config["model"].get("tie_weights", True),
        "vocab_size": len(vocab),
    }

    model = CaptionModel(**model_config).to(device)

    # ViT backbone is completely frozen; optimize only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config["train"]["lr"])

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["train"].get("lr_scheduler_factor", 0.5),
        patience=config["train"].get("lr_scheduler_patience", 2),
    )
    early_stopping = EarlyStopping(
        patience=config["train"].get("early_stopping_patience", 5)
    )

    best_checkpoint_path = config_dir / config["paths"]["checkpoint"]
    last_checkpoint_path = config_dir / config["paths"].get(
        "last_checkpoint", "checkpoints/last_checkpoint.pth"
    )

    start_epoch = 0
    if resume:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Resume requested but last checkpoint not found: {last_checkpoint_path}"
            )
        print(f"Resuming training from checkpoint: {last_checkpoint_path}")
        checkpoint = load_checkpoint(last_checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "early_stopping_state" in checkpoint:
            early_stopping.load_state_dict(checkpoint["early_stopping_state"])
            early_stopping.should_stop = False
        elif best_checkpoint_path.is_file():
            try:
                best_cp = load_checkpoint(best_checkpoint_path, map_location="cpu")
                early_stopping.best_loss = best_cp.get("val_loss", float("inf"))
            except Exception:
                pass

        start_epoch = checkpoint.get("epoch", 0)
        print(
            f"Resumed at epoch {start_epoch} (val_loss={checkpoint.get('val_loss', 'N/A')}). "
            f"Training will continue from epoch {start_epoch + 1}."
        )
    else:
        print("Starting training from scratch (epoch 0)...")

    if start_epoch >= config["train"]["epochs"]:
        print(
            f"Training already completed up to epoch {start_epoch} "
            f"(target epochs: {config['train']['epochs']})."
        )
        return

    print("Starting training with ViT-Transformer...")
    try:
        for epoch in range(start_epoch + 1, config["train"]["epochs"] + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                grad_clip=config["train"].get("grad_clip", 1.0),
            )
            val_loss = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch}/{config['train']['epochs']} "
                f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

            improved = early_stopping.step(val_loss)
            if improved:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    vocab,
                    model_config,
                    best_checkpoint_path,
                    scheduler=scheduler,
                    early_stopping=early_stopping,
                )
                print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

            # Always save last checkpoint after each epoch, regardless of improvement
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                vocab,
                model_config,
                last_checkpoint_path,
                scheduler=scheduler,
                early_stopping=early_stopping,
            )
            print(f"  -> Saved last checkpoint (epoch={epoch})")

            if early_stopping.should_stop:
                print(
                    f"Early stopping at epoch {epoch} (val_loss didn't improve for {early_stopping.patience} epochs)"
                )
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Last checkpoint has been preserved.")

    print("Training finished! Best val_loss:", early_stopping.best_loss)


if __name__ == "__main__":
    config, config_dir = load_config()
    main(config, config_dir)
