"""Evaluate generated validation captions with corpus metrics."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.metrics import calculate_caption_metrics
from src.model import CaptionModel
from src.transform import get_eval_transform
from src.utils import load_config, load_splits, load_vocab


class ValidationImageDataset(Dataset):
    """Images paired with their filenames so references can be looked up."""

    def __init__(self, image_dir, image_names, transform):
        self.image_dir = Path(image_dir)
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = self.image_dir / image_name
        with Image.open(image_path) as image_file:
            image = self.transform(image_file.convert("RGB"))
        return image, image_name


def load_model(config, config_dir, vocab, device):
    """Create the caption model and load its saved weights."""
    checkpoint_path = config_dir / config["paths"]["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # Use the architecture recorded with the weights if the config has since changed.
    saved_config = checkpoint.get("model_config") or config["model"]
    model = CaptionModel(
        embed_size=saved_config["embed_size"],
        hidden_size=saved_config["hidden_size"],
        vocab_size=saved_config.get("vocab_size", len(vocab)),
        num_layers=saved_config.get("num_layers", 1),
        dropout=saved_config.get("dropout", 0.5),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_references(split_path):
    """Load image reference captions from a filename-to-captions split file."""
    split = load_splits(split_path)
    if not isinstance(split, dict):
        raise ValueError(f"Split file must contain a filename-to-captions object: {split_path}")

    references = {}
    for image_name, captions in split.items():
        if isinstance(captions, str):
            captions = [captions]
        if not isinstance(captions, list):
            raise ValueError(f"Captions for {image_name!r} must be a string or list")
        clean_captions = [
            caption.strip()
            for caption in captions
            if isinstance(caption, str) and caption.strip()
        ]
        if clean_captions:
            references[str(image_name).strip()] = clean_captions
    return references


@torch.inference_mode()
def evaluate_model(
    config_path="config.yaml",
    method="beam",
    beam_size=3,
    batch_size=16,
    num_workers=0,
):
    """Evaluate validation captions with corpus-level caption metrics."""
    if method not in ("beam", "greedy"):
        raise ValueError("method must be 'beam' or 'greedy'")
    if beam_size < 1 or batch_size < 1:
        raise ValueError("beam_size and batch_size must be positive")

    config, config_dir = load_config(config_path)
    paths = config["paths"]
    vocab_path = config_dir / paths["vocab"]
    split_path = config_dir / paths["val_json"]
    image_dir = config_dir / paths["images"]
    for path in (vocab_path, split_path, image_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required validation input not found: {path}")

    vocab = load_vocab(vocab_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, config_dir, vocab, device)
    references_by_image = load_references(split_path)
    image_names = sorted(references_by_image)
    if not image_names:
        raise ValueError("No validation images with reference captions were found")

    dataset = ValidationImageDataset(
        image_dir,
        image_names,
        get_eval_transform(config["data"]["image_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda"
    )

    references, hypotheses = [], []
    max_len = config["data"].get("max_len", 30)
    for images, batch_names in tqdm(loader, desc="Evaluating captions"):
        predictions = model.generate_caption(
            images.to(device),
            vocab,
            max_len=max_len,
            method=method,
            beam_size=beam_size
        )
        for image_name, prediction in zip(batch_names, predictions):
            references.append([
                vocab.tokenize(caption)
                for caption in references_by_image[image_name]
            ])
            hypotheses.append(vocab.tokenize(" ".join(prediction)))

    scores = calculate_caption_metrics(references, hypotheses)
    scores["num_images"] = len(hypotheses)
    return scores


def main(config, config_dir):
    scores = evaluate_model(
        config_path=config_dir / "config.yaml",
        method=config["evaluation"].get("method", "beam"),
        beam_size=config["evaluation"].get("beam_size", 3),
        batch_size=config["evaluation"].get("batch_size", 16),
        num_workers=config["evaluation"].get("num_workers", 0),
    )
    print(f"Evaluated images: {scores.pop('num_images')}")
    for name, score in scores.items():
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()
