"""Q-GUARD quantum scorers operating on statevectors |psi(x)> (ZZ map default).

* TangentMahalanobis (primary): tangent displacement from the healthy base state,
  reduced by healthy-fit PCA, scored by shrinkage Mahalanobis.
* SubspaceFidelity: 1 - fidelity to the top-k healthy state subspace.
* QKernelSVDD: one-class SVM on the precomputed fidelity kernel.
All fit on healthy statevectors only.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from ..features import FeaturePCA
from ..quantum import fidelity_kernel
from .base import MahalanobisDetector


class TangentMahalanobis:
    def __init__(self, n_components: int = 20) -> None:
        self.n_components = n_components

    def _tangent(self, V):
        amp = V @ self.c0.conj()  # <c0|psi>
        T = V - amp[:, None] * self.c0[None, :]  # (I - C0)|psi>
        return np.concatenate([T.real, T.imag], axis=1)

    def fit(self, Vtr: np.ndarray) -> "TangentMahalanobis":
        rho_bar = (Vtr.T @ Vtr.conj()) / len(Vtr)
        _, U = np.linalg.eigh(rho_bar)
        self.c0 = U[:, -1]  # principal (base) state
        F = self._tangent(Vtr)
        self.pca = FeaturePCA(min(self.n_components, len(Vtr) - 1)).fit(F)
        self.det = MahalanobisDetector().fit(self.pca.transform(F))
        return self

    def score(self, V: np.ndarray) -> np.ndarray:
        return self.det.score(self.pca.transform(self._tangent(V)))


class SubspaceFidelity:
    def __init__(self, k: int = 20) -> None:
        self.k = k

    def fit(self, Vtr: np.ndarray) -> "SubspaceFidelity":
        G = Vtr.conj().T @ Vtr
        _, U = np.linalg.eigh(G)
        self.S = U[:, -min(self.k, Vtr.shape[0]):]
        return self

    def score(self, V: np.ndarray) -> np.ndarray:
        amp = V @ self.S.conj()  # <s_j|psi>
        return 1.0 - (np.abs(amp) ** 2).sum(axis=1)


class QKernelSVDD:
    def __init__(self, nu: float = 0.1) -> None:
        self.nu = nu

    def fit(self, Vtr: np.ndarray) -> "QKernelSVDD":
        self.Vtr = Vtr
        self.m = OneClassSVM(kernel="precomputed", nu=self.nu).fit(fidelity_kernel(Vtr))
        return self

    def score(self, V: np.ndarray) -> np.ndarray:
        return -self.m.decision_function(fidelity_kernel(V, self.Vtr))
