"""Neural network backbones for feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBackbone(nn.Module):
    """Simple CNN backbone for MNIST-like datasets.

    Architecture:
        Conv(1->32) -> BN -> ReLU -> MaxPool
        Conv(32->64) -> BN -> ReLU -> MaxPool
        Conv(64->128) -> BN -> ReLU -> MaxPool
        Linear(1152 -> feature_dim)

    Input: (B, 1, 28, 28) for MNIST
    Output: (B, feature_dim)
    """

    def __init__(self, input_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # After 3 pooling layers: 28 -> 14 -> 7 -> 3 (for MNIST)
        # For CIFAR (32x32): 32 -> 16 -> 8 -> 4
        self._fc_input_dim = None  # Computed dynamically
        self.fc = None
        self._feature_dim = feature_dim

    def _get_conv_output_dim(self, x: torch.Tensor) -> int:
        """Compute conv output dimension dynamically."""
        with torch.no_grad():
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            return x.view(x.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize fc layer on first forward pass
        if self.fc is None:
            self._fc_input_dim = self._get_conv_output_dim(x)
            self.fc = nn.Linear(self._fc_input_dim, self._feature_dim).to(x.device)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward - returns feature embeddings."""
        return self.forward(x)


class ResNetBackbone(nn.Module):
    """ResNet-18 backbone for CIFAR-like datasets.

    Uses smaller kernel size and no initial pooling for 32x32 images.
    """

    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        # Initial conv (no pooling for small images)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
