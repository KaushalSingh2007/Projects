"""Feature-space PCA reducer, fit on healthy-train features only."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class FeaturePCA:
    """Thin PCA wrapper. `fit` must receive healthy-train features only."""

    def __init__(self, n_components: int = 8, seed: int = 1234):
        self.n_components = n_components
        self.seed = seed
        self._pca: PCA | None = None

    def fit(self, x_train: np.ndarray) -> "FeaturePCA":
        k = min(self.n_components, x_train.shape[1], x_train.shape[0])
        self._pca = PCA(n_components=k, random_state=self.seed).fit(x_train)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("FeaturePCA used before fit().")
        return self._pca.transform(x)

    def config(self) -> dict:
        return {"name": "FeaturePCA", "n_components": self.n_components}
