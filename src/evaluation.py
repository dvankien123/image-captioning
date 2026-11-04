"""Evaluate generated validation captions with corpus metrics."""

from pathlib import Path
from statistics import mean, median

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.metrics import calculate_caption_metrics
from src.model import CaptionModel
from src.transform import get_eval_transform
from src.utils import load_checkpoint, load_config, load_splits, load_vocab


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
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    saved_config = checkpoint.get("model_config", {})
    model_cfg = config.get("model", {})
    model_kwargs = {
        "vocab_size": saved_config.get("vocab_size", len(vocab)),
        "d_model": saved_config.get("d_model", model_cfg.get("d_model", 512)),
        "nhead": saved_config.get("nhead", model_cfg.get("nhead", 8)),
        "num_decoder_layers": saved_config.get(
            "num_decoder_layers", model_cfg.get("num_decoder_layers", 3)
        ),
        "dim_feedforward": saved_config.get(
            "dim_feedforward", model_cfg.get("dim_feedforward", 2048)
        ),
        "dropout": saved_config.get("dropout", model_cfg.get("dropout", 0.1)),
        "tie_weights": saved_config.get(
            "tie_weights", model_cfg.get("tie_weights", True)
        ),
    }
    model = CaptionModel(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_references(annotation_path):
    """Load image reference captions from a COCO annotation file or split file."""
    data = load_splits(annotation_path)
    if not isinstance(data, dict):
        raise ValueError(f"Annotation file must contain a JSON object: {annotation_path}")

    # Case 1: MS COCO format (contains 'images' and 'annotations')
    if "annotations" in data and "images" in data:
        id_to_filename = {
            img["id"]: img["file_name"]
            for img in data["images"]
            if "id" in img and "file_name" in img
        }
        references = {}
        for ann in data["annotations"]:
            image_id = ann.get("image_id")
            image_name = id_to_filename.get(
                image_id, f"{image_id:012d}.jpg" if isinstance(image_id, int) else None
            )
            if not image_name:
                continue
            caption = ann.get("caption")
            if isinstance(caption, str) and caption.strip():
                references.setdefault(str(image_name).strip(), []).append(caption.strip())
        return references

@torch.inference_mode()
def evaluate_model(
    config_path="config.yaml",
    method=None,
    beam_size=None,
    length_penalty_alpha=None,
    batch_size=None,
    num_workers=None,
):
    """Evaluate validation captions with corpus-level caption metrics."""
    config, config_dir = load_config(config_path)
    paths = config["paths"]
    eval_cfg = config.get("evaluation", {})

    method = method or eval_cfg.get("method", "beam")
    beam_size = beam_size if beam_size is not None else eval_cfg.get("beam_size", 3)
    length_penalty_alpha = (
        length_penalty_alpha
        if length_penalty_alpha is not None
        else eval_cfg.get("length_penalty_alpha", 0.6)
    )
    batch_size = batch_size if batch_size is not None else eval_cfg.get("batch_size", 16)
    num_workers = num_workers if num_workers is not None else eval_cfg.get("num_workers", 2)

    if method not in ("beam", "greedy"):
        raise ValueError("method must be 'beam' or 'greedy'")
    if beam_size < 1 or batch_size < 1:
        raise ValueError("beam_size and batch_size must be positive")
    if length_penalty_alpha < 0:
        raise ValueError("length_penalty_alpha must be non-negative")

    vocab_path = config_dir / paths["vocab"]
    ann_path_str = paths.get("val_annotations")
    if not ann_path_str:
        raise KeyError("Annotation path for validation not found in config paths")
    annotations_path = config_dir / ann_path_str

    img_dir_str = paths.get("val_images") or paths.get("image_dir")
    if not img_dir_str:
        raise KeyError("Image directory for validation not found in config paths")
    image_dir = config_dir / img_dir_str

    for path in (vocab_path, annotations_path, image_dir):
        if not path.exists():
            raise FileNotFoundError("Required input not found: {path}")

    vocab = load_vocab(vocab_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, config_dir, vocab, device)
    references_by_image = load_references(annotations_path)

    image_names = [
        name for name in sorted(references_by_image.keys())
        if (image_dir / name).is_file()
    ]
    if not image_names:
        raise ValueError(
            f"No images from annotations ({annotations_path}) were found in directory ({image_dir})"
        )

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
        pin_memory=(device.type == "cuda"),
    )

    references, hypotheses = [], []
    max_len = config["data"].get("max_len", 30)
    for images, batch_names in tqdm(loader, desc=f"Evaluating val captions"):
        predictions, _ = model.generate_caption(
            images.to(device),
            vocab,
            max_len=max_len,
            method=method,
            beam_size=beam_size,
            length_penalty_alpha=length_penalty_alpha,
        )
        for image_name, prediction in zip(batch_names, predictions):
            references.append([
                vocab.tokenize(caption)
                for caption in references_by_image[image_name]
            ])
            hypotheses.append(vocab.tokenize(" ".join(prediction)))

    scores = calculate_caption_metrics(references, hypotheses)
    scores["num_images"] = len(hypotheses)
    caption_lengths = [len(hypothesis) for hypothesis in hypotheses]
    if caption_lengths:
        scores["caption_length_mean"] = float(mean(caption_lengths))
        scores["caption_length_median"] = float(median(caption_lengths))
        scores["caption_length_min"] = float(min(caption_lengths))
        scores["caption_length_max"] = float(max(caption_lengths))
    return scores


def main(config=None, config_dir=None):
    if config is None or config_dir is None:
        config, config_dir = load_config()

    eval_cfg = config.get("evaluation", {})
    scores = evaluate_model(
        config_path=config_dir / "config.yaml",
        method=eval_cfg.get("method", "beam"),
        beam_size=eval_cfg.get("beam_size", 3),
        length_penalty_alpha=eval_cfg.get("length_penalty_alpha", 0.6),
        batch_size=eval_cfg.get("batch_size", 16),
        num_workers=eval_cfg.get("num_workers", 2),
    )
    print(f"Evaluated images: {scores.pop('num_images')}")
    for name, score in scores.items():
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    config, config_dir = load_config()
    main(config, config_dir)
