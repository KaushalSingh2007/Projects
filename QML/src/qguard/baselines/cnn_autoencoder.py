"""1D-CNN autoencoder baseline (Avci); reconstruction error = novelty score.

Unsupervised: trained on healthy windows only; per-window MSE is the score.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..data.schema import Recording


class _AE(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(c, 16, 16, stride=4, padding=6), nn.ReLU(),
            nn.Conv1d(16, 8, 8, stride=4, padding=2), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(8, 16, 8, padding=3), nn.ReLU(),
            nn.Conv1d(16, c, 16, padding=7),
        )

    def forward(self, x):
        t = x.shape[-1]
        z = self.enc(x)
        return nn.functional.interpolate(self.dec(z), size=t, mode="linear", align_corners=False)


class CNNAEDetector:
    def __init__(self, epochs: int = 60, lr: float = 1e-3, batch: int = 64, seed: int = 1234):
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.model: _AE | None = None

    def fit(self, healthy: Recording) -> "CNNAEDetector":
        torch.manual_seed(self.seed)
        x = torch.tensor(np.asarray(healthy.signals), dtype=torch.float32)
        self.model = _AE(x.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            for i in range(0, len(x), self.batch):
                b = x[i:i + self.batch]
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.model(b), b)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, rec: Recording) -> np.ndarray:
        self.model.eval()
        x = torch.tensor(np.asarray(rec.signals), dtype=torch.float32)
        out = np.empty(len(x))
        for i in range(0, len(x), self.batch):
            b = x[i:i + self.batch]
            err = ((self.model(b) - b) ** 2).mean(dim=(1, 2))
            out[i:i + self.batch] = err.numpy()
        return out
