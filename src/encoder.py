"""Vision Transformer (ViT) image encoder extracting patch token representations."""

import torch
from torch import nn
from torchvision import models


class ViTEncoder(nn.Module):
    """Vision Transformer (ViT-B/16) image encoder.

    Extracts sequence of patch tokens from an image:
    (B, 3, 224, 224) -> (B, 197, 768) -> projected to (B, 197, d_model).
    Backbone weights are completely frozen.
    """

    def __init__(self, d_model=512, pretrained=True):
        super().__init__()
        self.d_model = d_model

        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = models.vit_b_16(weights=weights)

        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder
        self.proj = nn.Linear(768, d_model)

        # Completely freeze backbone weights
        for param in self.conv_proj.parameters():
            param.requires_grad = False
        self.class_token.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, images):
        """Forward pass extracting patch tokens.

        Args:
            images (torch.Tensor): (batch_size, 3, 224, 224)

        Returns:
            features (torch.Tensor): (batch_size, 197, d_model)
        """
        # Backbone is completely frozen, run with torch.no_grad() for speed and memory saving
        with torch.no_grad():
            x = self.conv_proj(images)  # (B, 768, 14, 14)
            x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)  # (B, 197, 768)
            features = self.encoder(x)  # (B, 197, 768)

        # Project to decoder dimension d_model (trainable projection layer)
        features = self.proj(features)  # (B, 197, d_model)
        return features

    def unfreeze_backbone(self, num_layers=2):
        """Optional fine-tuning helper (retained for backward compatibility)."""
        for param in self.encoder.layers[-num_layers:].parameters():
            param.requires_grad = True
        for param in self.encoder.ln.parameters():
            param.requires_grad = True


# Backward compatibility alias
Encoder = ViTEncoder
