"""PSD + PCA + Hotelling-T^2 (unsupervised) baseline.

Replaces Sohn's supervised Fisher projection (which needs damage labels) with
PCA + Mahalanobis to keep the label-free contract.
"""

from __future__ import annotations

from ..data.schema import Recording
from ..features import FeaturePCA, PSDFeatures
from ..scorers.base import MahalanobisDetector


class PCASPCDetector:
    def __init__(self, nperseg: int = 256, n_bins: int = 32, n_components: int = 8) -> None:
        self.feat = PSDFeatures(nperseg=nperseg, n_bins=n_bins)
        self.pca = FeaturePCA(n_components=n_components)
        self.det = MahalanobisDetector()

    def fit(self, healthy: Recording) -> "PCASPCDetector":
        x = self.pca.fit(self.feat.transform(healthy)).transform(self.feat.transform(healthy))
        self.det.fit(x)
        return self

    def score(self, rec: Recording):
        return self.det.score(self.pca.transform(self.feat.transform(rec)))
