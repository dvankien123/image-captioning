import torch
import torch.nn as nn

from src.encoder import Encoder
from src.decoder import Decoder

class CaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,
                 num_layers=1, dropout=0.5):
        super().__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size,
                               num_layers=num_layers, dropout=dropout)

    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs

    @torch.no_grad()
    def generate_caption(self, image, vocab, max_len=30,
                         method="greedy", beam_size=3):
        features = self.encoder(image)

        if method == "greedy":
            sampled_ids = self.decoder.sample(features, max_len=max_len)
            return [vocab.denumericalize(ids) for ids in sampled_ids]
        # elif method == "beam":
        #     from src.inference.beam_search import beam_search
        #     captions = []
        #     for i in range(features.size(0)):
        #         ids = beam_search(
        #             self.decoder, features[i:i + 1], vocab,
        #             beam_size=beam_size, max_len=max_len,
        #         )
        #         captions.append(vocab.denumericalize(ids))
        #     return captions
