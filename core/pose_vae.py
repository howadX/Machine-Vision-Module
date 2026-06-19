"""
core/pose_vae.py — VPoser-style VAE для 33 COCO landmarks.
Учит многообразие (manifold) правдоподобных человеческих поз.

Архитектура:
  Encoder:  [99] → BN → Linear(128) → ReLU → Linear(64) → ReLU → μ/σ: 32
  Decoder:  z(32) → Linear(64) → ReLU → Linear(128) → ReLU → Linear(99) → Tanh

Latent space: 32-мерное, нормально распределённое N(0,I).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PoseVAE(nn.Module):
    """VAE для 33 COCO landmarks (99-мерный вход: x,y,z × 33)."""

    def __init__(
        self,
        input_dim: int = 99,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────
        encoder_layers = [nn.BatchNorm1d(input_dim)]

        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # μ и log(σ²) — отдельные головы на последнем слое
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # ── Decoder ──────────────────────────────────────
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        decoder_layers.append(nn.Linear(in_dim, input_dim))
        # Tanh: выход в [-1, 1] — нормализованные координаты в этом диапазоне
        # (torso=1.0, значит голова может быть -1.5, ноги +2.0 — Tanh с запасом)

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Кодирует вход → μ, logvar."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ·ε."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Декодирует latent → реконструкция."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Полный проход VAE.

        Returns:
            recon: реконструированный вход [B, 99]
            mu: среднее [B, 32]
            logvar: логарифм дисперсии [B, 32]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def encode_to_z(self, x: torch.Tensor) -> torch.Tensor:
        """Инференс: вход → детерминированное z (без сэмплирования)."""
        mu, _logvar = self.encode(x)
        return mu

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ошибка реконструкции для anomaly detection.
        Возвращает MSE на каждый пример батча.
        """
        recon, mu, logvar = self.forward(x)
        return F.mse_loss(recon, x, reduction='none').mean(dim=1)

    def log_prob_z(self, x: torch.Tensor) -> torch.Tensor:
        """
        -log p(z) — «удивление» модели.
        Высокое значение → незнакомая поза.
        """
        mu, logvar = self.encode(x)
        # -log p(z) для нормального распределения N(0,I)
        # = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))
        var = torch.exp(logvar)
        return 0.5 * torch.sum(mu.pow(2) + var - 1 - logvar, dim=1)

    def sample(self, n: int = 1, device: torch.device = None) -> torch.Tensor:
        """Генерирует n случайных поз из N(0,I)."""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    def interpolate(
        self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10
    ) -> torch.Tensor:
        """Интерполирует между двумя позами в latent space."""
        z1 = self.encode_to_z(x1)
        z2 = self.encode_to_z(x2)
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        interpolated = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            interpolated.append(self.decode(z))
        return torch.stack(interpolated)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss: L_recon + β × KL.

    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction: MSE
    recon_loss = F.mse_loss(recon, x, reduction='mean')

    # KL divergence: D_KL(N(μ,σ) || N(0,I))
    # = -0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
