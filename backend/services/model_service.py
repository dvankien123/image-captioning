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
    """Loads and runs the configured ViT image-captioning model."""

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
        """Bind `src.*` imports to the model directory without cross-contamination."""
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
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Configured model checkpoint not found: {checkpoint_path}"
            )

        model_src_path = PROJECT_ROOT / "src"
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, vocab, transform, beam_search = self._load_vit(
            checkpoint_path, model_src_path, device, torch
        )

        self.model = model
        self.vocab = vocab
        self.device = device
        self.transform = transform
        self.beam_search = beam_search
        self.model_name = "ViT-B/16 + Transformer"
        self.supports_attention = True

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
        vocab = checkpoint.get("vocab")
        if vocab is None:
            vocab_path = PROJECT_ROOT / "data" / "processed" / "vocab.pickle"
            if vocab_path.is_file():
                import pickle
                with vocab_path.open("rb") as f:
                    vocab = pickle.load(f)
            else:
                raise ValueError("Checkpoint thiếu vocab và không tìm thấy data/processed/vocab.pickle.")
        if checkpoint.get("model_state_dict") is None:
            raise ValueError("Checkpoint phải có model_state_dict.")
        return checkpoint, vocab

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
                "Checkpoint is missing required ViT-Transformer model_config "
                f"fields: {', '.join(missing_config)}."
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

    def _attention_overlay(self, vis_image, attention):
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

        if w_max - w_min > 1e-8:
            weights = (weights - w_min) / (w_max - w_min)
        else:
            weights.fill(0.0)

        # Bicubic resize + Gaussian blur to match standard reference attention overlay
        heatmap = Image.fromarray(
            np.uint8(np.clip(weights * 255.0, 0.0, 255.0))
        ).resize(vis_image.size, Image.Resampling.BICUBIC)
        heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=4))

        h_arr = np.asarray(heatmap, dtype=np.float32) / 255.0
        colors = Image.fromarray(
            self._jet_colorize(h_arr)
        )

        # Uniform alpha blend (0.50) across full image preserving dark-blue cold regions and glowing hot spots
        overlay = Image.blend(vis_image, colors, alpha=0.50)

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

    def predict_detailed(self, image_path: str, include_attention: bool = True) -> dict:
        """Generate model-ranked captions and per-position decoder probabilities."""
        import torch
        from PIL import Image

        if self.model is None or self.vocab is None or self.transform is None:
            raise RuntimeError("Caption model is not loaded.")

        with_attention = include_attention and self.supports_attention
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            if with_attention:
                max_dim = 640
                width, height = image.size
                if max(width, height) > max_dim:
                    scale = max_dim / max(width, height)
                    vis_image = image.resize(
                        (int(width * scale), int(height * scale)), Image.Resampling.LANCZOS
                    )
                else:
                    vis_image = image.copy()
            else:
                vis_image = None

        candidates = []
        decoding_method = "greedy"
        beam_width = 1
        with torch.inference_mode():
            features = self.model.encoder(tensor)
            if self.beam_search is not None and settings.DECODING_METHOD == "beam":
                decoding_method = "beam"
                beam_width = settings.BEAM_SIZE
                decoded = self.beam_search(
                    self.model.decoder,
                    features,
                    self.vocab,
                    beam_size=beam_width,
                    max_len=settings.MAX_CAPTION_LENGTH,
                    length_penalty_alpha=settings.BEAM_LENGTH_PENALTY_ALPHA,
                    return_alphas=with_attention,
                    return_nbest=True,
                    nbest=min(4, beam_width),
                    token_top_k=6,
                )
                for item in decoded:
                    token_ids = item["token_ids"]
                    distributions = item["next_token_distributions"]
                    alpha_rows = list(item["alphas"]) if with_attention else []
                    candidates.append(self._format_candidate(
                        token_ids,
                        item["beam_score"],
                        distributions,
                        alpha_rows,
                        vis_image,
                        include_attention,
                    ))
            else:
                sampled = self.model.decoder.sample(
                    features,
                    max_len=settings.MAX_CAPTION_LENGTH,
                    return_alphas=with_attention,
                )
                if with_attention:
                    token_batches, alpha_batches = sampled
                else:
                    token_batches, alpha_batches = sampled, []
                token_ids = [
                    int(token_id) for token_id in token_batches[0]
                    if int(token_id) != self.vocab.word2idx["<end>"]
                ]
                distributions = self._next_token_distributions(features, token_ids)
                alpha_rows = list(alpha_batches[0]) if alpha_batches else []
                candidates.append(self._format_candidate(
                    token_ids,
                    None,
                    distributions,
                    alpha_rows,
                    vis_image,
                    include_attention,
                ))

        primary = candidates[0] if candidates else {
            "caption": "", "beam_score": None, "tokens": []
        }
        return {
            "caption": primary["caption"],
            "candidates": candidates,
            "attention": {
                "method": "vit_cross_attention" if self.supports_attention else "unavailable",
                "steps": [
                    {"word": token["word"], "image": token.get("image", "")}
                    for token in primary["tokens"]
                ] if with_attention else [],
            },
            "metadata": {
                "model": self.model_name,
                "device": str(self.device),
                "decoding_method": decoding_method,
                "beam_size": beam_width,
            },
        }

    def _next_token_distributions(self, features, token_ids):
        """Compute next-token distributions along a greedy sequence."""
        import torch
        from torch.nn import functional as F

        decoder = self.model.decoder
        prefix = [self.vocab.word2idx["<start>"]]
        distributions = []
        for _token_id in token_ids:
            current = torch.tensor([prefix], dtype=torch.long, device=features.device)
            logits, _ = decoder.forward_step(features, current)
            probabilities = F.softmax(logits, dim=-1).squeeze(0)
            values, indices = probabilities.topk(min(6, probabilities.numel()))
            distributions.append([
                (int(word_id), float(probability))
                for word_id, probability in zip(
                    indices.detach().cpu().tolist(), values.detach().cpu().tolist()
                )
            ])
            prefix.append(int(_token_id))
        return distributions

    def _format_candidate(
        self, token_ids, beam_score, distributions, alpha_rows, vis_image, include_attention
    ):
        special_ids = {
            self.vocab.word2idx.get("<pad>"),
            self.vocab.word2idx.get("<start>"),
            self.vocab.word2idx.get("<end>"),
        }
        tokens = []
        for index, token_id in enumerate(token_ids):
            word = self.vocab.idx2word.get(int(token_id), "<unk>")
            if int(token_id) in special_ids:
                continue
            token = {
                "word": word,
                "next_tokens": [
                    {
                        "word": self.vocab.idx2word.get(int(candidate_id), "<unk>"),
                        "probability": float(probability),
                        "selected": int(candidate_id) == int(token_id),
                    }
                    for candidate_id, probability in (
                        distributions[index] if index < len(distributions) else []
                    )
                    if int(candidate_id) not in special_ids
                ],
            }
            if include_attention and self.supports_attention and vis_image is not None and index < len(alpha_rows):
                token["image"] = self._attention_overlay(vis_image, alpha_rows[index])
            tokens.append(token)
        return {
            "caption": " ".join(token["word"] for token in tokens).strip(),
            "beam_score": float(beam_score) if beam_score is not None else None,
            "tokens": tokens,
        }


caption_model_service = CaptionModelService()
