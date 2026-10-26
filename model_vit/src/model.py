"""Vision Transformer (ViT) encoder and Transformer caption decoder with multi-head attention."""

import torch
import torch.nn as nn

from src.encoder import Encoder
from src.decoder import Decoder
from src.beam_search import beam_search


class CaptionModel(nn.Module):
    """Full Image Captioning Model combining ViT-B/16 Encoder and Transformer Decoder."""

    def __init__(
        self,
        vocab_size=None,
        d_model=512,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        tie_weights=True,
        train_backbone=False
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers

        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            tie_weights=tie_weights,
        )

        if train_backbone:
            self.encoder.unfreeze_backbone()

    def forward(self, images, captions, lengths=None):
        """Forward pass for teacher-forcing training.

        Args:
            images (torch.Tensor): (batch_size, 3, 224, 224)
            captions (torch.Tensor): (batch_size, seq_len)
            lengths (torch.Tensor, optional): (batch_size,)

        Returns:
            outputs (torch.Tensor): Logits (batch_size, seq_len - 1, vocab_size)
            alphas (torch.Tensor): Cross-attention weights (batch_size, seq_len - 1, 197)
        """
        features = self.encoder(images)  # (B, 197, d_model)
        outputs, alphas = self.decoder(features, captions, lengths)
        return outputs, alphas

    @torch.no_grad()
    def generate_caption(
        self,
        image,
        vocab,
        max_len=30,
        method="greedy",
        beam_size=3,
        length_penalty_alpha=0.6,
    ):
        """Generate caption tokens and attention heatmaps for image(s).

        Args:
            image (torch.Tensor): (batch_size, 3, 224, 224)
            vocab: Vocabulary object.
            max_len (int): Maximum caption length.
            method (str): 'greedy' or 'beam'.
            beam_size (int): Beam width for beam search.
            length_penalty_alpha (float): Length penalty exponent.

        Returns:
            results (list[list[str]]): List of generated word token lists per image.
            alphas (list[torch.Tensor]): List of attention weight tensors per image.
        """
        features = self.encoder(image)

        if method == "greedy":
            sampled_ids, alphas = self.decoder.sample(
                features, max_len=max_len, return_alphas=True
            )
            return [vocab.denumericalize(ids) for ids in sampled_ids], alphas

        if method == "beam":
            results = []
            alphas_list = []
            for i in range(features.size(0)):
                best_ids, beam_alphas = beam_search(
                    self.decoder,
                    features[i:i + 1],
                    vocab,
                    beam_size=beam_size,
                    max_len=max_len,
                    length_penalty_alpha=length_penalty_alpha,
                    return_alphas=True,
                )
                results.append(vocab.denumericalize([vocab.word2idx["<start>"]] + best_ids))
                alphas_list.append(beam_alphas)

            return results, alphas_list
