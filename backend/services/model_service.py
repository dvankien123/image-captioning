import base64
import importlib
from io import BytesIO
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

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

        checkpoint_path = Path(settings.MODEL_CHECKPOINT).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}. "
                "Place best_model.pth in model/checkpoints or set MODEL_CHECKPOINT."
            )

        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_root_path = PROJECT_ROOT / "model"
        model_src_path = model_root_path / "src"
        if not model_src_path.is_dir():
            raise FileNotFoundError(f"Model source directory not found: {model_src_path}")

        model_root = str(model_root_path)
        if model_root not in sys.path:
            sys.path.insert(0, model_root)
        importlib.invalidate_caches()

        # The repository also has a top-level `src` directory. Pin this package
        # to model/src so checkpoint imports and model imports cannot mix files.
        for module_name in tuple(sys.modules):
            if module_name == "src" or module_name.startswith("src."):
                del sys.modules[module_name]
        model_package = ModuleType("src")
        model_package.__path__ = [str(model_src_path)]
        model_package.__package__ = "src"
        sys.modules["src"] = model_package

        CaptionModel = importlib.import_module("src.model").CaptionModel
        get_eval_transform = importlib.import_module("src.transform").get_eval_transform
        Vocabulary = importlib.import_module("src.vocabulary").Vocabulary

        # Avoid executing arbitrary pickle globals; the checkpoint only needs
        # its project Vocabulary class in addition to PyTorch tensors.
        with torch.serialization.safe_globals([Vocabulary]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        model_config = checkpoint.get("model_config")
        vocab = checkpoint.get("vocab")
        model_state_dict = checkpoint.get("model_state_dict")
        if not model_config or vocab is None or model_state_dict is None:
            raise ValueError(
                "Checkpoint must contain 'model_config', 'model_state_dict', and 'vocab'."
            )

        model_config = dict(model_config)
        model_config["vocab_size"] = len(vocab)
        # The model source requests ImageNet weights in its constructor, but
        # the checkpoint already contains the trained encoder weights. Bypass
        # that download and load the checkpoint directly.
        import torchvision.models as torchvision_models

        resnet50 = torchvision_models.resnet50

        def resnet50_without_pretrained_weights(*args, **kwargs):
            kwargs["weights"] = None
            kwargs.pop("pretrained", None)
            return resnet50(*args, **kwargs)

        with patch.object(
            torchvision_models,
            "resnet50",
            new=resnet50_without_pretrained_weights,
        ):
            model = CaptionModel(**model_config)

        model.load_state_dict(model_state_dict)
        model.to(self.device)
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

    @staticmethod
    def _jet_colorize(heatmap):
        """Convert a normalized [H, W] NumPy heatmap to an RGB JET image."""
        import numpy as np

        values = np.clip(heatmap, 0.0, 1.0)
        red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
        rgb = np.stack((red, green, blue), axis=-1)
        return (rgb * 255).astype(np.uint8)

    def _attention_overlay(self, cropped_image, heatmap):
        import numpy as np
        from PIL import Image

        heatmap = np.nan_to_num(
            np.array(heatmap, dtype=np.float32, copy=True),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        heatmap -= heatmap.min()
        maximum = float(heatmap.max())
        if maximum > 1e-8:
            heatmap /= maximum
        else:
            heatmap.fill(0)

        colors = Image.fromarray(self._jet_colorize(heatmap))
        overlay = Image.blend(cropped_image, colors, alpha=0.45)
        buffer = BytesIO()
        overlay.save(buffer, format="JPEG", quality=68)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def predict_with_attention(self, image_path: str) -> tuple[str, list[dict[str, str]]]:
        """Generate a caption and post-hoc Grad-CAM maps for each decoder step.

        The model has no spatial attention layer. This computes token-conditioned
        Grad-CAM from ResNet layer4 without changing model code or weights. It is
        intentionally opt-in because gradients and overlay encoding cost more
        than normal caption inference.
        """
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from torchvision.transforms import CenterCrop, Resize

        if self.model is None or self.vocab is None or self.transform is None:
            raise RuntimeError("Caption model is not loaded.")

        with torch.inference_mode(False):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                tensor = self.transform(image).unsqueeze(0).to(self.device)
                # Reuse one cropped preview for every per-token overlay.
                cropped_image = CenterCrop(settings.IMAGE_SIZE)(Resize(256)(image))

            encoder = self.model.encoder
            decoder = self.model.decoder
            backbone = encoder.backbone
            if len(backbone) < 2:
                raise RuntimeError("Encoder backbone does not expose a spatial feature layer.")

            # Run the expensive, frozen CNN once. Use the layer4 feature grid
            # before ResNet's global average pool to retain spatial resolution.
            with torch.no_grad():
                spatial = backbone[:-1](tensor)
            spatial = spatial.detach().requires_grad_(True)

            # cuDNN's fused LSTM does not implement backward in eval mode.
            # Keep the model in eval mode (stable BatchNorm/dropout behavior)
            # and use PyTorch's native RNN path for this opt-in gradient pass.
            with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
                pooled = backbone[-1](spatial).flatten(1)
                features = encoder.bn(encoder.fc(pooled))
                inputs = features.unsqueeze(1)
                states = None
                start_id = self.vocab.word2idx.get("<start>", 1)
                end_id = self.vocab.word2idx.get("<end>", 2)
                pad_id = self.vocab.word2idx.get("<pad>", 0)
                context_token = start_id
                words = []
                attention_steps = []

                for _ in range(settings.MAX_CAPTION_LENGTH):
                    hiddens, states = decoder.lstm(inputs, states)
                    logits = decoder.fc(hiddens.squeeze(1))
                    predicted = logits.argmax(dim=1)
                    token_id = int(predicted.item())

                    if token_id == end_id:
                        break

                    # The map explains the decoder's current context token:
                    # <start> for the first prediction, then each emitted word.
                    score = logits[0, token_id]
                    gradients = torch.autograd.grad(
                        score,
                        spatial,
                        retain_graph=True,
                        create_graph=False,
                    )[0]
                    channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
                    cam = torch.relu((channel_weights * spatial).sum(dim=1, keepdim=True))
                    cam = functional.interpolate(
                        cam,
                        size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    heatmap = cam.detach().float().cpu().numpy()
                    label = self.vocab.idx2word.get(context_token, "<unk>")
                    attention_steps.append(
                        {
                            "word": label,
                            "image": self._attention_overlay(cropped_image, heatmap),
                        }
                    )

                    if token_id not in (pad_id, start_id):
                        token_word = self.vocab.idx2word.get(token_id, "<unk>")
                        words.append(token_word)
                    context_token = token_id
                    inputs = decoder.embed(predicted).unsqueeze(1)

                caption = " ".join(words).strip()
                return caption, attention_steps


caption_model_service = CaptionModelService()
