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
    model = CaptionModel(
        embed_size=saved_config["embed_size"],
        hidden_size=saved_config["hidden_size"],
        vocab_size=saved_config.get("vocab_size", len(vocab)),
        num_layers=saved_config.get("num_layers", 1),
        dropout=saved_config.get("dropout", 0.5),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = get_eval_transform(config["data"]["image_size"])

    return model, vocab, transform, device

def image_caption(image_input, model, vocab,
                  transform, device, maxlen=30, method="beam", beam_size=3):
    if isinstance(image_input, str):
        raw_image = Image.open(image_input).convert("RGB")
    else:
        raw_image = image_input.convert("RGB")

    image_tensor = transform(raw_image).unsqueeze(0).to(device)  # (1, 3, H, W)
    with torch.no_grad():
        caption = model.generate_caption(
            image_tensor,
            vocab,
            max_len=maxlen,
            method=method,
            beam_size=beam_size,
        )[0]
    return caption
