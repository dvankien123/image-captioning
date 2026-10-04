import torch
from torch import nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        resnet = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum = 0.01)

    def forward(self, images):
        if self.training and not any(param.requires_grad for param in self.backbone[-2].parameters()):
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)

        features = features.reshape(features.size(0), -1)
        features = self.bn(self.fc(features))
        return features

    def unfreeze_backbone(self):
        for param in self.backbone[-2].parameters():
            param.requires_grad = True