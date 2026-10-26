import base64
import importlib
import math
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from backend.core.config import PROJECT_ROOT, settings


class CaptionModelService:
    """Loads the configured ViT model, falling back to the local ResNet-LSTM."""

    def __init__(self):
        self.model = None
        self.vocab = None
        self.device = None
        self.transform = None
        self.beam_search = None
        self.model_name = None
        self.supports_attention = False

    @staticmethod
    def _activate_model_sources(model_src_path: Path):
        """Bind `src.*` imports to one model directory without cross-contamination."""
        for module_name in tuple(sys.modules):
            if module_name == "src" or module_name.startswith("src."):
                del sys.modules[module_name]
        model_package = ModuleType("src")
        model_package.__path__ = [str(model_src_path)]
        model_package.__package__ = "src"
        sys.modules["src"] = model_package
        importlib.invalidate_caches()
        return importlib.import_module("src.vocabulary").Vocabulary

    def load(self):
        self.unload()

        checkpoint_path = Path(settings.MODEL_CHECKPOINT).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        model_root_path = PROJECT_ROOT / "model_vit"
        model_src_path = model_root_path / "src"
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        legacy_checkpoint_path = PROJECT_ROOT / "model" / "checkpoints" / "best_model.pth"

        if checkpoint_path.is_file():
            model, vocab, transform, beam_search = self._load_vit(
                checkpoint_path, model_src_path, device, torch
            )
            model_name = "ViT-B/16 + Transformer"
            supports_attention = True
        elif legacy_checkpoint_path.is_file():
            model, vocab, transform, beam_search = self._load_legacy(
                legacy_checkpoint_path, device, torch
            )
            model_name = "ResNet-50 + LSTM (local fallback)"
            supports_attention = False
        else:
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint ViT: {checkpoint_path}; "
                f"cũng không có checkpoint dự phòng: {legacy_checkpoint_path}."
            )

        self.model = model
        self.vocab = vocab
        self.device = device
        self.transform = transform
        self.beam_search = beam_search
        self.model_name = model_name
        self.supports_attention = supports_attention

    def _load_checkpoint(self, checkpoint_path: Path, source_path: Path, torch):
        if not source_path.is_dir():
            raise FileNotFoundError(f"Model source directory not found: {source_path}")
        Vocabulary = self._activate_model_sources(source_path)
        with torch.serialization.safe_globals([Vocabulary]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint không hợp lệ: {checkpoint_path}")
        if checkpoint.get("vocab") is None or checkpoint.get("model_state_dict") is None:
            raise ValueError("Checkpoint phải có model_state_dict và vocab.")
        return checkpoint, checkpoint["vocab"]

    def _load_vit(self, checkpoint_path: Path, model_src_path: Path, device, torch):
        checkpoint, vocab = self._load_checkpoint(checkpoint_path, model_src_path, torch)
        CaptionModel = importlib.import_module("src.model").CaptionModel
        get_eval_transform = importlib.import_module("src.transform").get_eval_transform
        beam_search = importlib.import_module("src.beam_search").beam_search

        model_config = checkpoint.get("model_config")
        model_state_dict = checkpoint.get("model_state_dict")
        if not isinstance(model_config, dict) or vocab is None or model_state_dict is None:
            raise ValueError(
                "Checkpoint must contain 'model_config', 'model_state_dict', and 'vocab'."
            )

        required_config = {
            "d_model",
            "nhead",
            "num_decoder_layers",
            "dim_feedforward",
            "dropout",
            "tie_weights",
            "vocab_size",
        }
        missing_config = sorted(required_config.difference(model_config))
        if missing_config:
            raise ValueError(
                "Checkpoint is not from the ViT-Transformer model on main "
                f"(missing model_config fields: {', '.join(missing_config)}). "
                "The existing ResNet-LSTM checkpoint is not compatible."
            )

        model_config = dict(model_config)
        model_config["vocab_size"] = len(vocab)

        # Encoder defaults to ImageNet weights. The checkpoint already contains
        # the trained ViT, so avoid a network download during backend startup.
        import torchvision.models as torchvision_models

        vit_b_16 = torchvision_models.vit_b_16

        def vit_b_16_without_pretrained_weights(*args, **kwargs):
            kwargs["weights"] = None
            kwargs.pop("pretrained", None)
            return vit_b_16(*args, **kwargs)

        with patch.object(
            torchvision_models,
            "vit_b_16",
            new=vit_b_16_without_pretrained_weights,
        ):
            model = CaptionModel(**model_config)

        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        transform = get_eval_transform(settings.IMAGE_SIZE)
        return model, vocab, transform, beam_search

    def _load_legacy(self, checkpoint_path: Path, device, torch):
        source_path = PROJECT_ROOT / "model" / "src"
        checkpoint, vocab = self._load_checkpoint(checkpoint_path, source_path, torch)
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("ResNet-LSTM checkpoint thiếu model_config.")
        required = {"embed_size", "hidden_size", "vocab_size"}
        missing = sorted(required.difference(model_config))
        if missing:
            raise ValueError(
                "ResNet-LSTM checkpoint thiếu cấu hình: " + ", ".join(missing)
            )

        CaptionModel = importlib.import_module("src.model").CaptionModel
        get_eval_transform = importlib.import_module("src.transform").get_eval_transform
        import torchvision.models as torchvision_models

        resnet50 = torchvision_models.resnet50

        def resnet50_without_pretrained_weights(*args, **kwargs):
            kwargs["weights"] = None
            kwargs.pop("pretrained", None)
            return resnet50(*args, **kwargs)

        with patch.object(
            torchvision_models, "resnet50", new=resnet50_without_pretrained_weights
        ):
            model = CaptionModel(**model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        transform = get_eval_transform(settings.IMAGE_SIZE)
        return model, vocab, transform, None

    def unload(self):
        using_cuda = self.device is not None and self.device.type == "cuda"
        self.model = None
        self.vocab = None
        self.transform = None
        self.beam_search = None
        self.device = None
        self.model_name = None
        self.supports_attention = False
        if using_cuda:
            import torch

            torch.cuda.empty_cache()

    def _attention_overlay(self, cropped_image, attention):
        import numpy as np
        from PIL import Image, ImageFilter

        weights = np.nan_to_num(
            np.array(attention, dtype=np.float32, copy=True).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        patch_count = weights.size - 1
        grid_size = math.isqrt(patch_count)
        if grid_size * grid_size == patch_count:
            weights = weights[1:]  # Drop ViT's class token; it has no image patch.
        else:
            grid_size = math.isqrt(weights.size)
            if grid_size * grid_size != weights.size:
                raise RuntimeError(
                    f"Unexpected ViT attention size: {weights.size} tokens."
                )

        weights = np.maximum(weights.reshape(grid_size, grid_size), 0.0)
        w_min = float(weights.min())
        w_max = float(weights.max())
        w_mean = float(weights.mean())
        contrast = (w_max - w_min) / (w_mean + 1e-8)

        if w_max - w_min > 1e-8:
            weights = (weights - w_min) / (w_max - w_min)
        else:
            weights.fill(0.0)

        # Bicubic resize + Gaussian blur to eliminate 14x14 patch boundary artifacts
        heatmap = Image.fromarray(
            np.uint8(np.clip(weights * 255.0, 0.0, 255.0))
        ).resize(cropped_image.size, Image.Resampling.BICUBIC)
        heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=6))

        h_arr = np.asarray(heatmap, dtype=np.float32) / 255.0
        colors = Image.fromarray(
            self._jet_colorize(h_arr)
        )

        # Dynamic alpha blend: keep photo natural, only highlight hot regions
        strength = min(1.0, max(0.25, contrast / 3.0))
        alpha_mask = np.clip((h_arr - 0.20) / 0.65, 0.0, 1.0) * 0.58 * strength
        alpha_3d = np.expand_dims(alpha_mask, axis=-1)

        base_np = np.asarray(cropped_image.convert("RGB"), dtype=np.float32)
        col_np = np.asarray(colors, dtype=np.float32)
        blended = (base_np * (1.0 - alpha_3d) + col_np * alpha_3d).astype(np.uint8)
        overlay = Image.fromarray(blended)

        buffer = BytesIO()
        overlay.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

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

    def _predict(self, image_path: str, with_attention: bool):
        import torch
        from PIL import Image
        from torchvision.transforms import CenterCrop, Resize

        if (
            self.model is None
            or self.vocab is None
            or self.transform is None
        ):
            raise RuntimeError("Caption model is not loaded.")

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            cropped_image = (
                CenterCrop(settings.IMAGE_SIZE)(Resize(256)(image))
                if with_attention
                else None
            )

        with torch.inference_mode():
            features = self.model.encoder(tensor)
            if self.beam_search is not None and settings.DECODING_METHOD == "beam":
                result = self.beam_search(
                    self.model.decoder,
                    features,
                    self.vocab,
                    beam_size=settings.BEAM_SIZE,
                    max_len=settings.MAX_CAPTION_LENGTH,
                    length_penalty_alpha=settings.BEAM_LENGTH_PENALTY_ALPHA,
                    return_alphas=with_attention,
                )
                if with_attention:
                    token_ids, attention = result
                    attention_rows = list(attention)
                else:
                    token_ids = result
                    attention_rows = []
            else:
                if self.beam_search is None:
                    token_batches = self.model.decoder.sample(
                        features,
                        max_len=settings.MAX_CAPTION_LENGTH,
                    )
                    token_ids = token_batches[0]
                    attention_rows = []
                elif with_attention and self.supports_attention:
                    token_batches, attention_batches = self.model.decoder.sample(
                        features,
                        max_len=settings.MAX_CAPTION_LENGTH,
                        return_alphas=True,
                    )
                    token_ids = token_batches[0]
                    attention_rows = list(attention_batches[0])
                else:
                    token_batches = self.model.decoder.sample(
                        features,
                        max_len=settings.MAX_CAPTION_LENGTH,
                        return_alphas=False,
                    )
                    token_ids = token_batches[0]
                    attention_rows = []

        words = []
        token_attention = []
        end_id = self.vocab.word2idx["<end>"]
        excluded_ids = {
            self.vocab.word2idx["<pad>"],
            self.vocab.word2idx["<start>"],
            end_id,
        }
        for index, token_id in enumerate(token_ids):
            token_id = int(token_id)
            if token_id == end_id:
                break
            if token_id in excluded_ids:
                continue
            words.append(self.vocab.idx2word.get(token_id, "<unk>"))
            if with_attention and self.supports_attention and index < len(attention_rows):
                token_attention.append(attention_rows[index])

        caption = " ".join(words).strip()
        if not with_attention or not self.supports_attention or cropped_image is None:
            return caption, []

        steps = [
            {
                "word": word,
                "image": self._attention_overlay(cropped_image, weights),
            }
            for word, weights in zip(words, token_attention)
        ]
        return caption, steps

    def predict(self, image_path: str) -> str:
        caption, _ = self._predict(image_path, with_attention=False)
        return caption

    def predict_with_attention(self, image_path: str) -> tuple[str, list[dict[str, str]]]:
        """Generate a caption with the ViT-Transformer's visual cross-attention."""
        return self._predict(image_path, with_attention=True)


caption_model_service = CaptionModelService()
