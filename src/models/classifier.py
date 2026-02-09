"""Classifier model combining backbone and classification head."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .backbone import CNNBackbone, ResNetBackbone


class Classifier(nn.Module):
    """Full classifier with backbone and linear head.

    Supports:
    - Feature extraction for replay buffer diversity
    - Fisher information computation for EWC
    - Fast weight updates for meta-learning
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 10,
        feature_dim: int = 128
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone (for replay buffer diversity)."""
        return self.backbone(x)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both logits and features."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits, features

    @classmethod
    def create_for_mnist(cls, num_classes: int = 10, feature_dim: int = 128) -> 'Classifier':
        """Factory method for MNIST-style datasets."""
        backbone = CNNBackbone(input_channels=1, feature_dim=feature_dim)
        return cls(backbone, num_classes, feature_dim)

    @classmethod
    def create_for_cifar(cls, num_classes: int = 10, feature_dim: int = 512) -> 'Classifier':
        """Factory method for CIFAR-style datasets."""
        backbone = ResNetBackbone(input_channels=3, feature_dim=feature_dim)
        return cls(backbone, num_classes, feature_dim)

    def get_params_for_ewc(self) -> dict:
        """Get named parameters for EWC regularization."""
        return {name: param.clone() for name, param in self.named_parameters()}

    def copy_weights_from(self, other: 'Classifier'):
        """Copy weights from another model (for meta-learning)."""
        self.load_state_dict(other.state_dict())
