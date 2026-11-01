"""Autoregressive (Yule-Walker) coefficient features (Sohn baseline path)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.schema import Recording


def ar_yule_walker(sig2d: np.ndarray, order: int) -> np.ndarray:
    """Batched AR(order) coefficients via Yule-Walker for signals (M, T)."""
    m, t = sig2d.shape
    x = sig2d - sig2d.mean(axis=1, keepdims=True)
    # Autocovariance lags 0..order (vectorized over the batch).
    r = np.empty((m, order + 1))
    for k in range(order + 1):
        r[:, k] = (x[:, k:] * x[:, : t - k]).sum(axis=1) / t
    # Build batched Toeplitz systems R a = r1.
    idx = np.abs(np.subtract.outer(np.arange(order), np.arange(order)))
    R = r[:, idx]  # (M, order, order)
    R[:, np.arange(order), np.arange(order)] += 1e-10  # regularize
    a = np.linalg.solve(R, r[:, 1:order + 1][..., None])[..., 0]
    return a  # (M, order)


@dataclass
class ARFeatures:
    """AR(order) coefficients per window, averaged over sensors -> (n_windows, order)."""

    order: int = 8

    def transform(self, rec: Recording) -> np.ndarray:
        s = np.asarray(rec.signals)  # (n, C, T)
        n, c, t = s.shape
        coeffs = ar_yule_walker(s.reshape(n * c, t), self.order).reshape(n, c, self.order)
        return coeffs.mean(axis=1).astype("float64")

    def config(self) -> dict:
        return {"name": "ARFeatures", "order": self.order}
