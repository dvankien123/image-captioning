import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualAttention(nn.Module):
    """Additive (Bahdanau-style) visual attention with beta-gating.

    Calculates alignment scores between decoder hidden states and encoder
    spatial feature maps, producing a context vector and attention weights.

    Mathematical formulas:
        1. Energy scores:
           e_{t, i} = v_a^T tanh(W_enc * a_i + W_dec * h_{t-1} + b)
        2. Attention weights:
           alpha_{t, i} = exp(e_{t, i}) / sum_k exp(e_{t, k})
        3. Raw context vector:
           z_t = sum_i (alpha_{t, i} * a_i)
        4. Gating mechanism:
           beta_t = sigmoid(W_beta * h_{t-1} + b_beta)
        5. Modulated context vector:
           z_hat_t = beta_t * z_t
    """

    def __init__(self, encoder_dim=2048, hidden_size=512, attention_dim=512):
        """Initialize attention layers.

        Args:
            encoder_dim (int): Number of channels from the visual encoder (D).
            hidden_size (int): Dimension of decoder LSTM hidden state (H).
            attention_dim (int): Dimension of attention intermediate space (A).
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(hidden_size, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

        self.f_beta = nn.Linear(hidden_size, encoder_dim)

    def forward(self, encoder_out, decoder_hidden):
        """Forward pass for visual attention.

        Args:
            encoder_out (torch.Tensor): Feature map from encoder of shape
                (batch_size, num_pixels, encoder_dim), e.g. (B, 49, 2048).
            decoder_hidden (torch.Tensor): Previous decoder hidden state of shape
                (batch_size, hidden_size), e.g. (B, 512).

        Returns:
            context (torch.Tensor): Context vector of shape (batch_size, encoder_dim).
            alpha (torch.Tensor): Attention weights of shape (batch_size, num_pixels).
        """
        # Step 1: Project encoder features -> (B, L, attention_dim)
        att1 = self.encoder_att(encoder_out)

        # Step 2: Project decoder hidden state -> (B, attention_dim)
        att2 = self.decoder_att(decoder_hidden)

        # Step 3: Broadcast add along spatial dimension L and apply tanh -> (B, L, attention_dim)
        combined = torch.tanh(att1 + att2.unsqueeze(1))

        # Step 4: Project to scalar energy -> (B, L, 1) -> squeeze to (B, L)
        energy = self.full_att(combined).squeeze(2)

        # Step 5: Softmax over spatial locations L -> (B, L)
        alpha = F.softmax(energy, dim=1)

        # Step 6: Weighted sum of encoder features -> (B, encoder_dim)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        # Step 7: Gating mechanism (beta-gate)
        gate = torch.sigmoid(self.f_beta(decoder_hidden))  # (B, encoder_dim)
        context = gate * context

        return context, alpha

