"""Welch PSD features (quantum-path representation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

from ..data.schema import Recording


def welch_psd(signals: np.ndarray, fs: float, nperseg: int = 256):
    """Return (freqs, Pxx_mean) where Pxx_mean is averaged over sensors.

    signals: (n_windows, n_sensors, n_timepoints) -> Pxx_mean (n_windows, n_freq).
    """
    nperseg = int(min(nperseg, signals.shape[-1]))
    freqs, pxx = sp_signal.welch(signals, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, pxx.mean(axis=1)  # mean over sensors


def _pool(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Average-pool the frequency axis of (n, F) into n_bins bands."""
    f = x.shape[1]
    if n_bins >= f:
        return x
    edges = np.linspace(0, f, n_bins + 1).astype(int)
    return np.stack([x[:, edges[i]:edges[i + 1]].mean(axis=1) for i in range(n_bins)], axis=1)


@dataclass
class PSDFeatures:
    """log10 Welch PSD, sensor-averaged, optionally pooled to `n_bins`."""

    nperseg: int = 256
    n_bins: int | None = 32
    eps: float = 1e-12

    def transform(self, rec: Recording) -> np.ndarray:
        _, pxx = welch_psd(np.asarray(rec.signals), fs=rec.fs, nperseg=self.nperseg)
        x = np.log10(pxx + self.eps)
        if self.n_bins:
            x = _pool(x, self.n_bins)
        return x.astype("float64")

    def config(self) -> dict:
        return {"name": "PSDFeatures", "nperseg": self.nperseg, "n_bins": self.n_bins}
