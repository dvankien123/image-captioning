"""Transformer caption decoder with multi-head self-attention and visual cross-attention."""

import math
import torch
from torch import nn


class TransformerDecoderLayer(nn.Module):
    """Single layer of Transformer Decoder with cross-attention weights retention."""

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        return_cross_attn=False,
    ):
        # 1. Masked Self-Attention
        tgt2, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. Cross-Attention over visual features (memory)
        tgt2, cross_attn_weights = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=return_cross_attn,
            average_attn_weights=True,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # 3. Position-wise Feed-Forward Network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))

        return tgt, cross_attn_weights


class Decoder(nn.Module):
    """Transformer Caption Decoder using learnable word embeddings and multi-head cross-attention."""

    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=100,
        tie_weights=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Learnable word embedding (no external 3.64GB Word2Vec dependency)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Learnable positional embedding
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.emb_dropout = nn.Dropout(dropout)
        self.emb_layer_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.fc.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(self, encoder_out, captions, lengths=None):
        """Forward pass with teacher forcing for training.

        Args:
            encoder_out (torch.Tensor): Visual memory of shape (batch_size, 197, d_model).
            captions (torch.Tensor): Padded captions of shape (batch_size, seq_len).
            lengths (torch.Tensor, optional): Lengths of captions.

        Returns:
            predictions (torch.Tensor): Output logits of shape (batch_size, seq_len - 1, vocab_size).
            last_alphas (torch.Tensor): Cross-attention weights of shape (batch_size, seq_len - 1, 197).
        """
        device = captions.device
        # Teacher forcing input: tokens from 0 to T-1
        tgt_input = captions[:, :-1]
        seq_len = tgt_input.size(1)

        pos = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        x = self.embed(tgt_input) * math.sqrt(self.d_model) + self.pos_emb(pos)
        x = self.emb_layer_norm(self.emb_dropout(x))

        tgt_mask = self.generate_square_subsequent_mask(seq_len, device)
        tgt_key_padding_mask = (tgt_input == 0)  # <pad> is 0

        last_alphas = None
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            x, alphas = layer(
                tgt=x,
                memory=encoder_out,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                return_cross_attn=is_last,
            )
            if is_last:
                last_alphas = alphas

        x = self.final_norm(x)
        predictions = self.fc(x)  # (B, seq_len, vocab_size)

        return predictions, last_alphas

    def forward_step(self, encoder_out, current_tokens, return_alphas=False):
        """Run a single forward pass given currently generated tokens.

        Args:
            encoder_out (torch.Tensor): (batch_size, 197, d_model)
            current_tokens (torch.Tensor): (batch_size, cur_len)

        Returns:
            logits (torch.Tensor): Logits for the last token position (batch_size, vocab_size).
            last_alpha (torch.Tensor): Cross-attention weights for the last token (batch_size, 197).
        """
        device = current_tokens.device
        cur_len = current_tokens.size(1)

        pos = torch.arange(cur_len, device=device).unsqueeze(0)
        x = self.embed(current_tokens) * math.sqrt(self.d_model) + self.pos_emb(pos)
        x = self.emb_layer_norm(self.emb_dropout(x))

        tgt_mask = self.generate_square_subsequent_mask(cur_len, device)
        tgt_key_padding_mask = (current_tokens == 0)

        last_alpha = None
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            x, alphas = layer(
                tgt=x,
                memory=encoder_out,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                return_cross_attn=is_last and return_alphas,
            )
            if is_last and alphas is not None:
                last_alpha = alphas[:, -1, :]  # (B, 197) for the latest token

        x = self.final_norm(x)
        logits = self.fc(x[:, -1, :])  # (B, vocab_size)
        return logits, last_alpha

    def sample(self, encoder_out, max_len=30, return_alphas=False):
        """Greedily decode captions autoregressively.

        Args:
            encoder_out (torch.Tensor): Visual memory (batch_size, 197, d_model).
            max_len (int): Maximum length to generate.
            return_alphas (bool): Whether to return cross-attention heatmaps.

        Returns:
            sampled_ids (list[list[int]]): Generated token IDs per image.
            alphas (list[torch.Tensor], optional): Cross-attention weights per image.
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        start_id = 1  # <start>
        end_id = 2    # <end>
        pad_id = 0    # <pad>

        current_tokens = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)
        sampled_ids = [[] for _ in range(batch_size)]
        all_alphas = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, step_alpha = self.forward_step(
                encoder_out, current_tokens, return_alphas=return_alphas
            )

            # Prevent sampling <pad> or <start>
            logits[:, pad_id] = float("-inf")
            logits[:, start_id] = float("-inf")

            predicted = logits.argmax(dim=-1)  # (B,)

            for i in range(batch_size):
                if not finished[i]:
                    token_id = predicted[i].item()
                    sampled_ids[i].append(token_id)
                    if return_alphas and step_alpha is not None:
                        all_alphas[i].append(step_alpha[i].detach().cpu())
                    if token_id == end_id:
                        finished[i] = True

            if finished.all():
                break

            current_tokens = torch.cat([current_tokens, predicted.unsqueeze(1)], dim=1)

        if return_alphas:
            stacked_alphas = [
                torch.stack(a) if a else torch.empty(0, encoder_out.size(1))
                for a in all_alphas
            ]
            return sampled_ids, stacked_alphas
        return sampled_ids
