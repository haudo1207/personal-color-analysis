from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class PersonalColorModel(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.head_fitz = nn.Linear(in_features, 6)
        self.head_under = nn.Linear(in_features, 3)
        self.head_season = nn.Linear(in_features, 4)

    def forward(self, x):
        features = self.dropout(self.backbone(x))
        return self.head_fitz(features), self.head_under(features), self.head_season(features)

