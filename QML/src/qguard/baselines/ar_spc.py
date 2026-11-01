"""Sohn-faithful baseline: AR coefficients + Mahalanobis/SPC novelty."""

from __future__ import annotations

from ..data.schema import Recording
from ..features import ARFeatures
from ..scorers.base import MahalanobisDetector


class ARSPCDetector:
    def __init__(self, order: int = 8) -> None:
        self.feat = ARFeatures(order=order)
        self.det = MahalanobisDetector()

    def fit(self, healthy: Recording) -> "ARSPCDetector":
        self.det.fit(self.feat.transform(healthy))
        return self

    def score(self, rec: Recording):
        return self.det.score(self.feat.transform(rec))
