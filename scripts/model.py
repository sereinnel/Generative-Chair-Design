"""PointNet-style variational autoencoder for chair point clouds."""

from __future__ import annotations

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    """Encode an unordered point cloud with shared MLP layers and max pooling."""

    def __init__(self, bottleneck_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, bottleneck_dim, kernel_size=1),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Return one global feature vector per point cloud.

        Args:
            points: Tensor with shape ``(batch, 3, num_points)``.
        """
        if points.ndim != 3 or points.shape[1] != 3:
            raise ValueError(
                "PointNetEncoder expects input with shape (batch, 3, num_points)"
            )

        point_features = self.features(points)
        return torch.amax(point_features, dim=2)


class PointNetVAE(nn.Module):
    """Variational autoencoder used in the original chair experiment.

    The fully connected decoder directly predicts every output coordinate.
    It is intentionally kept unchanged so that existing checkpoints remain
    compatible with the cleaned code.
    """

    def __init__(
        self,
        num_points: int = 16_384,
        latent_dim: int = 256,
        bottleneck_dim: int = 512,
    ) -> None:
        super().__init__()

        if num_points <= 0:
            raise ValueError("num_points must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.num_points = num_points
        self.latent_dim = latent_dim

        self.encoder = PointNetEncoder(bottleneck_dim=bottleneck_dim)
        self.mean_head = nn.Linear(bottleneck_dim, latent_dim)
        self.log_variance_head = nn.Linear(bottleneck_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_points * 3),
        )

    def encode(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a point cloud to the parameters of a latent Gaussian."""
        features = self.encoder(points)
        return self.mean_head(features), self.log_variance_head(features)

    @staticmethod
    def reparameterize(
        mean: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Draw a differentiable sample from a diagonal Gaussian."""
        standard_deviation = torch.exp(0.5 * log_variance)
        noise = torch.randn_like(standard_deviation)
        return mean + noise * standard_deviation

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into point clouds with shape ``(B, N, 3)``."""
        coordinates = self.decoder(latent)
        return coordinates.reshape(-1, self.num_points, 3)

    def forward(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct point clouds and return latent distribution parameters."""
        mean, log_variance = self.encode(points)
        latent = self.reparameterize(mean, log_variance)
        reconstruction = self.decode(latent)
        return reconstruction, mean, log_variance
