import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from src.utils import load_config, save_checkpoint, EarlyStopping, load_vocab, load_embedding_matrix
from src.dataset import Flickr30kDataset, collate_fn
from src.transform import get_train_transform, get_eval_transform
from src.model import CaptionModel


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=5.0, doubly_stochastic_lambda=1.0):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for images, captions, lengths in progress:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        outputs, alphas = model(images, captions, lengths)
        decode_lengths = (lengths - 1).clamp(min=1)
        max_decode_len = outputs.size(1)

        packed_outputs = pack_padded_sequence(
            outputs, decode_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        ).data
        packed_targets = pack_padded_sequence(
            captions[:, 1:max_decode_len + 1], decode_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        ).data

        ce_loss = criterion(packed_outputs, packed_targets)

        # Doubly stochastic attention regularization:
        mask = torch.arange(max_decode_len, device=device).unsqueeze(0) < decode_lengths.unsqueeze(1)
        masked_alphas = alphas * mask.unsqueeze(2)
        alpha_sum = masked_alphas.sum(dim=1)
        att_reg = doubly_stochastic_lambda * ((1.0 - alpha_sum) ** 2).mean()

        loss = ce_loss + att_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_tokens = packed_targets.size(0)
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        progress.set_postfix(loss=loss.item())

    return total_loss / total_tokens

@torch.no_grad()
def validate(model, loader, criterion, device, doubly_stochastic_lambda=1.0):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(loader, desc="Validating", leave=False)
    for images, captions, lengths in progress:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        outputs, alphas = model(images, captions, lengths)
        decode_lengths = (lengths - 1).clamp(min=1)
        max_decode_len = outputs.size(1)

        packed_outputs = pack_padded_sequence(
            outputs, decode_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        ).data
        packed_targets = pack_padded_sequence(
            captions[:, 1:max_decode_len + 1], decode_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        ).data

        ce_loss = criterion(packed_outputs, packed_targets)

        mask = torch.arange(max_decode_len, device=device).unsqueeze(0) < decode_lengths.unsqueeze(1)
        masked_alphas = alphas * mask.unsqueeze(2)
        alpha_sum = masked_alphas.sum(dim=1)
        att_reg = doubly_stochastic_lambda * ((1.0 - alpha_sum) ** 2).mean()

        loss = ce_loss + att_reg

        batch_tokens = packed_targets.size(0)
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        progress.set_postfix(loss=loss.item())

    return total_loss / total_tokens

def load_training_checkpoint(path):
    """Load a checkpoint while supporting older PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main(config, config_dir, resume=False):
    torch.manual_seed(config["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = load_vocab(config_dir / config["paths"]["vocab"])
    print(f"Vocab size: {len(vocab)}")

    pretrained_embeddings = None
    if config["model"].get("use_pretrained_embedding", False):
        emb_path = config_dir / config["paths"].get("embedding_matrix", "data/processed/embedding_matrix.pt")
        if not emb_path.is_file():
            raise FileNotFoundError(
                f"Pretrained embedding matrix not found at {emb_path}. "
                "Please run notebooks/build_embedding.ipynb first to generate it."
            )
        print(f"Loading pretrained embeddings from: {emb_path}")
        pretrained_embeddings = load_embedding_matrix(emb_path)
        print(f"Pretrained embedding shape: {pretrained_embeddings.shape}")

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
        train_set, batch_size=config["train"]["batch_size"],
        shuffle=True, collate_fn=collate_fn,
        num_workers=config["train"].get("num_workers", 4), pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config["train"]["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=config["train"].get("num_workers", 4), pin_memory=True,
    )

    freeze_embedding = config["model"].get("freeze_embedding", False)
    attention_dim = config["model"].get("attention_dim", 512)
    encoder_dim = config["model"].get("encoder_dim", 2048)
    doubly_stochastic_lambda = config["model"].get("doubly_stochastic_lambda", 1.0)

    model_config = {
        "embed_size": config["model"]["embed_size"],
        "hidden_size": config["model"]["hidden_size"],
        "vocab_size": len(vocab),
        "dropout": config["model"]["dropout"],
        "freeze_embedding": freeze_embedding,
        "attention_dim": attention_dim,
        "encoder_dim": encoder_dim,
        "doubly_stochastic_lambda": doubly_stochastic_lambda,
    }
    model = CaptionModel(
        embed_size=model_config["embed_size"],
        hidden_size=model_config["hidden_size"],
        vocab_size=model_config["vocab_size"],
        dropout=model_config["dropout"],
        pretrained_embeddings=pretrained_embeddings,
        freeze_embedding=freeze_embedding,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
    ).to(device)

    # If fine-tuning the visual backbone is enabled
    if config["model"].get("train_backbone", False):
        model.encoder.unfreeze_backbone()
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": decoder_params, "lr": config["train"]["lr"]},
            {"params": encoder_params, "lr": config["train"].get("lr_finetune", 1e-5)},
        ])
    else:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["train"]["lr"],
        )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["train"]["lr"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["train"].get("lr_scheduler_factor", 0.5),
        patience=config["train"].get("lr_scheduler_patience", 2),
    )
    early_stopping = EarlyStopping(
        patience=config["train"].get("early_stopping_patience", 5)
    )

    checkpoint_path = config_dir / config["paths"]["checkpoint"]
    last_checkpoint_path = config_dir / config["paths"].get(
        "last_checkpoint", "checkpoints/last_checkpoint.pth"
    )
    start_epoch = 1

    if resume:
        resume_candidates = [
            path for path in (last_checkpoint_path, checkpoint_path) if path.is_file()
        ]
        if not resume_candidates:
            raise FileNotFoundError(
                "No checkpoint is available to resume from. Expected one of: "
                f"{last_checkpoint_path} or {checkpoint_path}"
            )
        resume_path = max(resume_candidates, key=lambda path: path.stat().st_mtime)

        checkpoint = load_training_checkpoint(resume_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        saved_early_stopping = checkpoint.get("early_stopping_state")
        if saved_early_stopping:
            early_stopping.best_loss = saved_early_stopping.get(
                "best_loss", checkpoint.get("val_loss", float("inf"))
            )
            early_stopping.counter = saved_early_stopping.get("counter", 0)
            early_stopping.should_stop = saved_early_stopping.get("should_stop", False)
        elif checkpoint.get("val_loss") is not None:
            # Older best-model checkpoints did not include early-stopping state.
            early_stopping.best_loss = checkpoint["val_loss"]

        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_rng_states = checkpoint.get("cuda_rng_state_all")
        if torch.cuda.is_available() and cuda_rng_states:
            for device_index, rng_state in enumerate(cuda_rng_states[:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(rng_state.cpu(), device_index)

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed from {resume_path} at epoch {start_epoch}")
        if early_stopping.should_stop:
            print("This checkpoint had already reached early stopping; no further epochs will run.")
            return

    # --- Vòng lặp epoch ---
    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=config["train"]["grad_clip"],
            doubly_stochastic_lambda=doubly_stochastic_lambda,
        )
        val_loss = validate(
            model, val_loader, criterion, device,
            doubly_stochastic_lambda=doubly_stochastic_lambda,
        )
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{config['train']['epochs']} "
              f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        improved = early_stopping.step(val_loss)
        if improved:
            save_checkpoint(
                model, optimizer, epoch, val_loss, vocab, model_config, checkpoint_path,
                scheduler=scheduler, early_stopping=early_stopping,
            )
            print(f"  -> Saved checkpoint (val_loss={val_loss:.4f})")

        save_checkpoint(
            model, optimizer, epoch, val_loss, vocab, model_config, last_checkpoint_path,
            scheduler=scheduler, early_stopping=early_stopping,
        )
        print(f"  -> Saved latest training state to {last_checkpoint_path}")

        if early_stopping.should_stop:
            print(
                f"Early stopping at epoch {epoch} (val_loss didn't improve {early_stopping.patience} epoch liên tiếp)")
            break

    print("Training done! Best val_loss:", early_stopping.best_loss)


