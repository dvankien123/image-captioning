import importlib
import sys
from pathlib import Path

from backend.core.config import PROJECT_ROOT, settings


class CaptionModelService:
    """Loads and runs the trained image-captioning model from the model directory."""

    def __init__(self):
        self.model = None
        self.vocab = None
        self.device = None
        self.transform = None

    def load(self):
        self.model = None
        self.vocab = None
        self.device = None
        self.transform = None

        checkpoint_path = Path(settings.MODEL_CHECKPOINT)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}. "
                "Train the model and set MODEL_CHECKPOINT to its checkpoint file."
            )

        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_root = str(PROJECT_ROOT / "model")
        if model_root not in sys.path:
            sys.path.insert(0, model_root)
        importlib.invalidate_caches()

        model_src = str(PROJECT_ROOT / "model" / "src")
        import src
        if model_src not in src.__path__:
            raise ImportError(f"The model source directory is not importable: {model_src}")

        CaptionModel = importlib.import_module("src.model").CaptionModel
        get_eval_transform = importlib.import_module("src.transform").get_eval_transform

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_config = checkpoint.get("model_config")
        vocab = checkpoint.get("vocab")
        if not model_config or vocab is None:
            raise ValueError("Checkpoint must contain 'model_config' and 'vocab'.")

        model_config = dict(model_config)
        model_config["vocab_size"] = len(vocab)
        model = CaptionModel(**model_config).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        transform = get_eval_transform(settings.IMAGE_SIZE)

        self.model = model
        self.vocab = vocab
        self.transform = transform

    def unload(self):
        using_cuda = self.device is not None and self.device.type == "cuda"
        self.model = None
        self.vocab = None
        self.transform = None
        self.device = None
        if using_cuda:
            import torch

            torch.cuda.empty_cache()

    def predict(self, image_path: str) -> str:
        import torch
        from PIL import Image

        if self.model is None or self.vocab is None or self.transform is None:
            raise RuntimeError("Caption model is not loaded.")

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features = self.model.encoder(tensor)
            token_batches = self.model.decoder.sample(
                features, max_len=settings.MAX_CAPTION_LENGTH
            )

        words = []
        for token_id in token_batches[0]:
            if token_id == self.vocab.word2idx.get("<end>", 2):
                break
            if token_id in (
                self.vocab.word2idx.get("<pad>", 0),
                self.vocab.word2idx.get("<start>", 1),
            ):
                continue
            words.append(self.vocab.idx2word.get(token_id, "<unk>"))
        return " ".join(words).strip()


caption_model_service = CaptionModelService()
