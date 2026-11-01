"""NoveltyDetector protocol + Mahalanobis (Hotelling-T^2) detector."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.covariance import LedoitWolf


@runtime_checkable
class NoveltyDetector(Protocol):
    """Fit on healthy features only; score returns higher = more anomalous."""

    def fit(self, x_healthy: np.ndarray) -> "NoveltyDetector": ...
    def score(self, x: np.ndarray) -> np.ndarray: ...


class MahalanobisDetector:
    """Shrinkage Mahalanobis (Ledoit-Wolf) novelty score; healthy-fit."""

    def __init__(self) -> None:
        self._cov: LedoitWolf | None = None

    def fit(self, x_healthy: np.ndarray) -> "MahalanobisDetector":
        self._cov = LedoitWolf().fit(np.asarray(x_healthy, dtype="float64"))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self._cov is None:
            raise RuntimeError("MahalanobisDetector used before fit().")
        return self._cov.mahalanobis(np.asarray(x, dtype="float64"))  # squared distance
