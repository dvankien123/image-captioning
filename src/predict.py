"""Load a trained captioning model and generate captions for images."""

import torch
from PIL import Image

from src.utils import load_vocab, load_config
from src.transform import get_eval_transform
from src.model import CaptionModel


def load_model(config, config_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab(config_dir / config["paths"]["vocab"])

    checkpoint_path = config_dir / config["paths"]["checkpoint"]
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # Checkpoint dimensions must match its saved weights, even if config changed later.
    saved_config = checkpoint.get("model_config") or config["model"]
    model_kwargs = {
        "vocab_size": saved_config.get("vocab_size", len(vocab)),
        "d_model": saved_config.get("d_model", saved_config.get("embed_size", 512)),
        "nhead": saved_config.get("nhead", 8),
        "num_decoder_layers": saved_config.get("num_decoder_layers", 3),
        "dim_feedforward": saved_config.get("dim_feedforward", 2048),
        "dropout": saved_config.get("dropout", 0.1),
        "tie_weights": saved_config.get("tie_weights", True),
    }
    model = CaptionModel(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = get_eval_transform(config["data"]["image_size"])

    return model, vocab, transform, device


def image_caption(
    image_input,
    model,
    vocab,
    transform,
    device,
    maxlen=30,
    method="beam",
    beam_size=3,
    return_alphas=False,
):
    if isinstance(image_input, str):
        raw_image = Image.open(image_input).convert("RGB")
    else:
        raw_image = image_input.convert("RGB")

    image_tensor = transform(raw_image).unsqueeze(0).to(device)  # (1, 3, H, W)
    with torch.no_grad():
        res, alphas = model.generate_caption(
            image_tensor,
            vocab,
            max_len=maxlen,
            method=method,
            beam_size=beam_size,
        )
    if return_alphas:
        return res, alphas
    return res
