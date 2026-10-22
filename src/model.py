"""Image encoder and caption decoder with visual attention."""

import torch
import torch.nn as nn

from src.encoder import Encoder
from src.decoder import Decoder
from src.beam_search import beam_search


class CaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,
                 attention_dim=512, encoder_dim=2048, dropout=0.5,
                 pretrained_embeddings=None, freeze_embedding=False):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embedding=freeze_embedding,
        )

    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs, alphas = self.decoder(features, captions, lengths)
        return outputs, alphas

    @torch.no_grad()
    def generate_caption(self, image, vocab, max_len=30,
                         method="greedy", beam_size=3,
                         length_penalty_alpha=0.6):
        features = self.encoder(image)

        if method == "greedy":
            sampled_ids, alphas = self.decoder.sample(features, max_len=max_len, return_alphas=True)
            return [vocab.denumericalize(ids) for ids in sampled_ids], alphas

        if method == "beam":
            results = []
            alphas_list = []
            for i in range(features.size(0)):
                best_ids, beam_alphas = beam_search(
                    self.decoder, features[i:i + 1], vocab,
                    beam_size=beam_size, max_len=max_len,
                    length_penalty_alpha=length_penalty_alpha,
                    return_alphas=True,
                )
                results.append(vocab.denumericalize([vocab.word2idx["<start>"]] + best_ids))
                alphas_list.append(beam_alphas)

            return results, alphas_list
