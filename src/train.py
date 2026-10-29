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
from src.dataset import CocoDataset, collate_fn
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


def load_pretrained_weights(model, checkpoint, tie_weights=True):
    """Transfer weights from a pretrained checkpoint, supporting partial vocabulary transfer."""
    old_state_dict = checkpoint["model_state_dict"]
    model_state_dict = model.state_dict()
    old_vocab_size = checkpoint.get("model_config", {}).get("vocab_size", 0)

    # 1. Transfer non-vocabulary layers (Encoder and Decoder Transformer layers)
    transferred_count = 0
    for k, v in old_state_dict.items():
        if k in ("decoder.embed.weight", "decoder.fc.weight"):
            continue
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            model_state_dict[k].copy_(v)
            transferred_count += 1

    # 2. Transfer vocabulary embeddings for overlapping token indices
    if "decoder.embed.weight" in old_state_dict and "decoder.embed.weight" in model_state_dict:
        embed_param = model_state_dict["decoder.embed.weight"]
        old_embed = old_state_dict["decoder.embed.weight"]
        copy_len = min(old_embed.size(0), embed_param.size(0))
        embed_param[:copy_len].copy_(old_embed[:copy_len])

        if tie_weights:
            model_state_dict["decoder.fc.weight"].copy_(embed_param)
        elif "decoder.fc.weight" in old_state_dict:
            old_fc = old_state_dict["decoder.fc.weight"]
            fc_param = model_state_dict["decoder.fc.weight"]
            fc_len = min(old_fc.size(0), fc_param.size(0))
            fc_param[:fc_len].copy_(old_fc[:fc_len])

    model.load_state_dict(model_state_dict)
    print(
        f"Transferred {transferred_count} layer tensors and {old_vocab_size} "
        f"token embeddings from pretrained checkpoint."
    )


def main(config, config_dir, resume=None):
    if resume is None:
        resume = config["train"].get("resume", False)

    torch.manual_seed(config["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = load_vocab(config_dir / config["paths"]["vocab"])
    print(f"Vocab size: {len(vocab)}")

    train_set = CocoDataset(
        image_path=config_dir / config["paths"]["train_images"],
        annotation_path=config_dir / config["paths"]["train_annotations"],
        vocab=vocab,
        transform=get_train_transform(config["data"]["image_size"]),
    )
    val_set = CocoDataset(
        image_path=config_dir / config["paths"]["val_images"],
        annotation_path=config_dir / config["paths"]["val_annotations"],
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
        "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"],
        "num_decoder_layers": config["model"]["num_decoder_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"],
        "dropout": config["model"]["dropout"],
        "tie_weights": config["model"]["tie_weights"],
        "train_backbone": config["model"]["train_backbone"],
        "vocab_size": len(vocab),
    }

    model = CaptionModel(**model_config).to(device)

    best_checkpoint_path = config_dir / config["paths"]["checkpoint"]
    last_checkpoint_path = config_dir / config["paths"].get(
        "last_checkpoint", "checkpoints/last_checkpoint_coco.pth"
    )
    pretrained_checkpoint_path = config_dir / config["paths"].get(
        "pretrained_checkpoint", "checkpoints/best_model.pth"
    )

    start_epoch = 0
    checkpoint = None
    is_resumed_from_same_vocab = False
    finetune_transition = False

    if resume:
        if not last_checkpoint_path.is_file():
            if pretrained_checkpoint_path.is_file():
                print(
                    f"Notice: '{last_checkpoint_path}' not found, but pretrained checkpoint "
                    f"'{pretrained_checkpoint_path}' was found. Starting COCO phase from pretrained weights..."
                )
                checkpoint = load_checkpoint(pretrained_checkpoint_path, map_location=device)
                load_pretrained_weights(model, checkpoint, tie_weights=model_config["tie_weights"])
                start_epoch = 0
            else:
                raise FileNotFoundError(
                    f"Resume requested but neither last checkpoint ({last_checkpoint_path}) "
                    f"nor pretrained checkpoint ({pretrained_checkpoint_path}) was found."
                )
        else:
            print(f"Resuming training from checkpoint: {last_checkpoint_path}")
            checkpoint = load_checkpoint(last_checkpoint_path, map_location=device)
            ckpt_vocab_size = checkpoint.get("model_config", {}).get("vocab_size")

            if ckpt_vocab_size == len(vocab):
                last_model_config = checkpoint.get("model_config", {})
                finetune_transition = (
                    config["model"]["train_backbone"]
                    and not last_model_config.get("train_backbone", False)
                )
                if finetune_transition:
                    if not best_checkpoint_path.is_file():
                        raise FileNotFoundError(
                            "Fine-tuning transition requested, but best checkpoint not found: "
                            f"{best_checkpoint_path}"
                        )
                    print(
                        "Fine-tuning transition detected; loading best checkpoint: "
                        f"{best_checkpoint_path}"
                    )
                    checkpoint = load_checkpoint(best_checkpoint_path, map_location=device)

                model.load_state_dict(checkpoint["model_state_dict"])
                start_epoch = checkpoint["epoch"]
                is_resumed_from_same_vocab = True
            else:
                print(
                    f"Checkpoint vocab size ({ckpt_vocab_size}) != current vocab size ({len(vocab)}). "
                    "Performing partial weight transfer..."
                )
                load_pretrained_weights(model, checkpoint, tie_weights=model_config["tie_weights"])
                start_epoch = 0
    elif pretrained_checkpoint_path.is_file():
        print(f"Loading pretrained weights from: {pretrained_checkpoint_path}")
        checkpoint = load_checkpoint(pretrained_checkpoint_path, map_location=device)
        load_pretrained_weights(model, checkpoint, tie_weights=model_config["tie_weights"])
        start_epoch = 0
    else:
        print("Starting training from scratch (no pretrained checkpoint found)...")

    # Group parameters for different learning rates
    backbone_params = []
    if config["model"]["train_backbone"]:
        backbone_module = model.encoder.encoder
        backbone_params = [
            *backbone_module.layers[-2:].parameters(),
            *backbone_module.ln.parameters(),
        ]
    backbone_param_ids = {id(param) for param in backbone_params}
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    base_params = [
        param for param in trainable_params if id(param) not in backbone_param_ids
    ]

    if finetune_transition:
        # Match the frozen checkpoint's original optimizer group before restoring
        # its state, then add newly unfrozen backbone parameters with fresh state.
        optimizer = torch.optim.Adam(base_params, lr=config["train"]["lr"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer.add_param_group(
            {
                "params": backbone_params,
                "lr": config["train"]["lr_finetune"]
            }
        )
    elif (
        is_resumed_from_same_vocab
        and config["model"]["train_backbone"]
        and len(checkpoint["optimizer_state_dict"]["param_groups"]) == 2
    ):
        optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": config["train"]["lr"]},
                {"params": backbone_params, "lr": config["train"]["lr_finetune"]},
            ]
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif config["model"]["train_backbone"] and backbone_params:
        optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": config["train"]["lr"]},
                {"params": backbone_params, "lr": config["train"]["lr_finetune"]},
            ]
        )
        if is_resumed_from_same_vocab and checkpoint is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Notice: Could not restore optimizer state ({e}); using fresh optimizer.")
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=config["train"]["lr"])
        if is_resumed_from_same_vocab and checkpoint is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Notice: Could not restore optimizer state ({e}); using fresh optimizer.")

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

    if is_resumed_from_same_vocab and not finetune_transition and checkpoint is not None:
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "early_stopping_state" in checkpoint:
            early_stopping.load_state_dict(checkpoint["early_stopping_state"])
            early_stopping.should_stop = False

    if is_resumed_from_same_vocab:
        print(
            f"Resumed at epoch {start_epoch} (val_loss={checkpoint['val_loss']}). "
            f"Training will continue from epoch {start_epoch + 1}."
        )
    else:
        print("Starting COCO training phase from epoch 1...")

    if start_epoch >= config["train"]["epochs"]:
        print(
            f"Training already completed up to epoch {start_epoch} "
            f"(target epochs: {config['train']['epochs']})."
        )
        return

    print("Starting training with ViT-Transformer on MS COCO 2017...")
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
