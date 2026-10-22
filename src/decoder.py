"""LSTM caption decoder with visual attention and autoregressive sampling."""

import torch
from torch import nn

from src.attention import VisualAttention


class Decoder(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        attention_dim=512,
        encoder_dim=2048,
        dropout=0.5,
        pretrained_embeddings=None,
        freeze_embedding=False,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim

        self.attention = VisualAttention(
            encoder_dim=encoder_dim,
            hidden_size=hidden_size,
            attention_dim=attention_dim,
        )

        if pretrained_embeddings is not None:
            self.embed = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embedding,
                padding_idx=0,
            )
        else:
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # Initial hidden/cell states projected from mean spatial features
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)

        # LSTMCell input: [word_embedding; context_vector]
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_out):
        """Initialize LSTM hidden and cell states from mean encoder features."""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = torch.tanh(self.init_h(mean_encoder_out))
        c = torch.tanh(self.init_c(mean_encoder_out))
        return h, c

    def forward(self, encoder_out, captions, lengths):
        """Unroll decoding over time steps with teacher forcing.

        Args:
            encoder_out (torch.Tensor): (batch_size, num_pixels, encoder_dim)
            captions (torch.Tensor): (batch_size, max_len)
            lengths (torch.Tensor): (batch_size,)

        Returns:
            predictions (torch.Tensor): (batch_size, max_decode_len, vocab_size)
            alphas (torch.Tensor): (batch_size, max_decode_len, num_pixels)
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        decode_lengths = (lengths - 1).clamp(min=1)
        max_decode_len = int(decode_lengths.max().item())

        embeddings = self.embed(captions)
        h, c = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_len, num_pixels, device=encoder_out.device)

        for t in range(max_decode_len):
            context, alpha = self.attention(encoder_out, h)
            # embeddings[:, t, :] is token y_t (<start> at t=0)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(h))

            predictions[:, t, :] = output
            alphas[:, t, :] = alpha

        return predictions, alphas

    def sample(self, encoder_out, max_len=30, return_alphas=False):
        """Greedily decode captions from spatial image features.

        Args:
            encoder_out (torch.Tensor): (batch_size, num_pixels, encoder_dim)
            max_len (int): Maximum length of generated caption.
            return_alphas (bool): Whether to return attention weights.

        Returns:
            sampled_ids (list[list[int]]): Generated token IDs per image.
            alphas (list[torch.Tensor], optional): Attention weights per image.
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        device = encoder_out.device

        h, c = self.init_hidden_state(encoder_out)
        prev_words = torch.full((batch_size,), 1, dtype=torch.long, device=device)

        sampled_ids = [[] for _ in range(batch_size)]
        all_alphas = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            context, alpha = self.attention(encoder_out, h)
            word_emb = self.embed(prev_words)
            lstm_input = torch.cat([word_emb, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            outputs = self.fc(h).clone()
            outputs[:, 0] = float("-inf")  # <pad>
            outputs[:, 1] = float("-inf")  # <start>

            predicted = outputs.argmax(dim=1)

            for i in range(batch_size):
                if not finished[i]:
                    sampled_ids[i].append(predicted[i].item())
                    if return_alphas:
                        all_alphas[i].append(alpha[i].detach().cpu())
                    if predicted[i].item() == 2:  # <end>
                        finished[i] = True

            if finished.all():
                break

            prev_words = predicted

        if return_alphas:
            stacked_alphas = [
                torch.stack(a) if a else torch.empty(0, num_pixels)
                for a in all_alphas
            ]
            return sampled_ids, stacked_alphas
        return sampled_ids
