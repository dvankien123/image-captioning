"""ResNet image encoder extracting spatial feature maps for visual attention."""

import torch
from torch import nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=7):
        super().__init__()
        self.encoded_image_size = encoded_image_size

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove avgpool and fc to retain spatial features (B, 2048, H, W)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, images):
        if self.training and not any(param.requires_grad for param in self.backbone[-1].parameters()):
            # Avoid building a gradient graph while backbone weights are frozen.
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)

        # features: (B, 2048, H, W) -> adaptive pool -> (B, 2048, 7, 7)
        features = self.adaptive_pool(features)
        # Permute to (B, 7, 7, 2048) -> reshape to (B, 49, 2048)
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(features.size(0), -1, 2048)
        return features

    def unfreeze_backbone(self):
        for param in self.backbone[-1].parameters():
            param.requires_grad = True

