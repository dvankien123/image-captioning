"""Load a trained captioning model and generate captions for images."""

import torch
from PIL import Image

from src.utils import load_checkpoint, load_vocab, load_config
from src.transform import get_eval_transform
from src.model import CaptionModel


def load_model(config, config_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab(config_dir / config["paths"]["vocab"])

    checkpoint_path = config_dir / config["paths"]["checkpoint"]
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    saved_config = checkpoint["model_config"]
    model_kwargs = {
        "vocab_size": saved_config["vocab_size"],
        "d_model": saved_config["d_model"],
        "nhead": saved_config["nhead"],
        "num_decoder_layers": saved_config["num_decoder_layers"],
        "dim_feedforward": saved_config["dim_feedforward"],
        "dropout": saved_config["dropout"],
        "tie_weights": saved_config["tie_weights"],
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
