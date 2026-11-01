"""Variance-adaptive Median-of-Means: robust aggregation of noisy observations."""

from __future__ import annotations

import numpy as np


def median_of_means(x: np.ndarray, n_groups: int = 8) -> np.ndarray:
    """Median (per coordinate) of `n_groups` group means. Robust to heavy tails."""
    x = np.asarray(x, dtype="float64")
    n_groups = int(max(1, min(n_groups, len(x))))
    means = [g.mean(axis=0) for g in np.array_split(x, n_groups)]
    return np.median(np.stack(means), axis=0)


def variance_adaptive_mean(values: np.ndarray, variances: np.ndarray,
                           eps: float = 1e-12) -> np.ndarray:
    """Inverse-variance weighted mean (more weight to low-variance observations)."""
    values = np.asarray(values, dtype="float64")
    w = 1.0 / (np.asarray(variances, dtype="float64") + eps)
    return (w * values).sum(axis=0) / w.sum(axis=0)
