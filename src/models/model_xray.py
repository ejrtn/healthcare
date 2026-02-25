import torch
import torch.nn as nn
from torchvision import models

class XRayDenseNetModel(nn.Module):
    """
    X-ray multi-label classifier using DenseNet-121.
    Designed to work with NIH and CheXpert datasets.
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        
        # Initially freeze backbone for transfer learning
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
