"""Training and validation loops for the image-captioning model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from src.utils import save_checkpoint, EarlyStopping, load_vocab, load_config
from src.dataset import Flickr30kDataset, collate_fn
from src.transform import get_train_transform, get_eval_transform
from src.model import CaptionModel


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=5.0):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for images, captions, lengths in progress:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        outputs = model(images, captions, lengths)
        # Pack targets in the same order and lengths used by the decoder outputs.
        targets = pack_padded_sequence(
            captions, lengths.cpu(),
            batch_first=True, enforce_sorted=False
        ).data

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_tokens = targets.size(0)
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        progress.set_postfix(loss=loss.item())

    return total_loss / total_tokens

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

        outputs = model(images, captions, lengths)
        # Ignore padded positions when averaging validation loss.
        targets = pack_padded_sequence(
            captions, lengths.cpu(),
            batch_first=True, enforce_sorted=False
        ).data

        loss = criterion(outputs, targets)

        batch_tokens = targets.size(0)
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        progress.set_postfix(loss=loss.item())

    return total_loss / total_tokens

def main(config, config_dir):
    torch.manual_seed(config["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = load_vocab(config_dir / config["paths"]["vocab"])
    print(f"Vocab size: {len(vocab)}")

    train_set = Flickr30kDataset(
        image_dir=config_dir / config["paths"]["images"],
        json_dir=config_dir / config["paths"]["train_json"],
        vocab=vocab,
        transform=get_train_transform(config["data"]["image_size"]),
    )
    val_set = Flickr30kDataset(
        image_dir=config_dir / config["paths"]["images"],
        json_dir=config_dir / config["paths"]["val_json"],
        vocab=vocab,
        transform=get_eval_transform(config["data"]["image_size"]),
    )

    train_loader = DataLoader(
        train_set, batch_size=config["train"]["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config["train"]["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    model_config = {
        "embed_size": config["model"]["embed_size"],
        "hidden_size": config["model"]["hidden_size"],
        "vocab_size": len(vocab),
        "num_layers": config["model"]["num_layers"],
        "dropout": config["model"]["dropout"]
    }
    model = CaptionModel(**model_config).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["train"]["lr"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    early_stopping = EarlyStopping(patience=5)

    checkpoint_path = config_dir / config["paths"]["checkpoint"]

    # --- Vòng lặp epoch ---
    for epoch in range(1, config["train"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=config["train"]["grad_clip"],
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{config['train']['epochs']} "
              f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        improved = early_stopping.step(val_loss)
        if improved:
            save_checkpoint(model, optimizer, epoch, val_loss, vocab, model_config, checkpoint_path)
            print(f"  -> Saved checkpoint (val_loss={val_loss:.4f})")

        if early_stopping.should_stop:
            print(
                f"Early stopping at epoch {epoch} (val_loss didn't improve {early_stopping.patience} epoch liên tiếp)")
            break

    print("Training done! Best val_loss:", early_stopping.best_loss)


if __name__ == "__main__":
    main()
