# model_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, bottleneck_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, bottleneck_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(bottleneck_dim)

    def forward(self, x):
        # x: (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # global maxpool -> (B, bottleneck_dim)
        return x

class PointNetVAE(nn.Module):
    def __init__(self, num_points=16384, latent_dim=256, bottleneck_dim=512):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        # Энкодер
        self.encoder = PointNetEncoder(bottleneck_dim=bottleneck_dim)
        self.fc_mu = nn.Linear(bottleneck_dim, latent_dim)
        self.fc_logvar = nn.Linear(bottleneck_dim, latent_dim)

        # Декодер (MLP -> генерируем num_points * 3)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_points * 3),
        )

    def encode(self, x):
        # x: (B, 3, N)
        feat = self.encoder(x)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: (B, latent_dim)
        out = self.decoder(z)  # (B, num_points*3)
        out = out.view(-1, self.num_points, 3)  # (B, N, 3)
        return out

    def forward(self, x):
        # x: (B, 3, N)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
