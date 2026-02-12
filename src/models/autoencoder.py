"""Autoencoder model for continual learning.

Uses encoder to learn latent representations, decoder for reconstruction,
and a classifier head on the latent space for task prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvEncoder(nn.Module):
    """Convolutional encoder for MNIST-like images."""

    def __init__(self, input_channels: int = 1, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # 28x28 -> 14x14 -> 7x7 -> 3x3
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Flatten and project to latent space
        # For 28x28 input: after 3 stride-2 convs -> 4x4
        self._fc_input_dim = None
        self.fc = None

    def _get_conv_output_dim(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return x.view(x.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc is None:
            self._fc_input_dim = self._get_conv_output_dim(x)
            self.fc = nn.Linear(self._fc_input_dim, self.latent_dim).to(x.device)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class ConvDecoder(nn.Module):
    """Convolutional decoder for MNIST-like images."""

    def __init__(self, latent_dim: int = 64, output_channels: int = 1, output_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size

        # Project from latent to spatial
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        # Upsample: 4x4 -> 7x7 -> 14x14 -> 28x28
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 128, 4, 4)

        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))

        # Ensure output matches expected size
        if x.size(-1) != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)

        return x


class AutoencoderClassifier(nn.Module):
    """Autoencoder with classification head for continual learning.

    Architecture:
        Input -> Encoder -> Latent (z) -> Classifier -> Class prediction
                                       -> Decoder -> Reconstruction

    The encoder learns representations via reconstruction loss,
    and classification happens on the latent space.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 28,
        latent_dim: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.feature_dim = latent_dim  # For compatibility with replay buffer
        self.num_classes = num_classes

        # Encoder
        self.encoder = ConvEncoder(input_channels, latent_dim)

        # Decoder
        self.decoder = ConvDecoder(latent_dim, input_channels, input_size)

        # Classification head on latent space
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        z = self.encode(x)
        logits = self.classifier(z)
        return logits

    def forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning logits, latent, and reconstruction."""
        z = self.encode(x)
        logits = self.classifier(z)
        recon = self.decode(z)
        return logits, z, recon

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent features (for replay buffer diversity sampling)."""
        return self.encode(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input."""
        z = self.encode(x)
        return self.decode(z)

    def reconstruction_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss (MSE)."""
        return F.mse_loss(recon, x)

    @classmethod
    def create_for_mnist(cls, num_classes: int = 10, latent_dim: int = 64) -> 'AutoencoderClassifier':
        """Factory for MNIST."""
        return cls(input_channels=1, input_size=28, latent_dim=latent_dim, num_classes=num_classes)

    @classmethod
    def create_for_cifar(cls, num_classes: int = 10, latent_dim: int = 128) -> 'AutoencoderClassifier':
        """Factory for CIFAR."""
        return cls(input_channels=3, input_size=32, latent_dim=latent_dim, num_classes=num_classes)
