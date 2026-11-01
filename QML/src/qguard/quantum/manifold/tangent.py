"""Tangent space, projector, and symmetric retraction on the pure-state manifold.

Base state C = |c><c|. Tangent space (Lumbreras-Tomamichel, eq. 12):
    T_C M = { V = (1/sqrt2)(|phi><c| + |c><phi|) : <phi|c> = 0 }.
Orthogonal projector (Lemma 2):  P_C(X) = C X (I-C) + (I-C) X C.
Retraction (Lemma 3) via the geodesic |gamma(tau)> = cos(tau/sqrt2)|c> + sin(tau/sqrt2)|phi>.
"""

from __future__ import annotations

import numpy as np

_SQRT2 = np.sqrt(2.0)


def density(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=complex)
    c = c / np.linalg.norm(c)
    return np.outer(c, c.conj())


def tangent_projector(C: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Frobenius-orthogonal projection of X onto T_C M."""
    eye = np.eye(C.shape[0], dtype=complex)
    return C @ X @ (eye - C) + (eye - C) @ X @ C


def tangent_from_phi(c: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Build a (normalized) tangent vector from base c and orthogonal phi."""
    c = np.asarray(c, dtype=complex) / np.linalg.norm(c)
    phi = np.asarray(phi, dtype=complex)
    phi = phi - (c.conj() @ phi) * c  # enforce orthogonality
    phi = phi / np.linalg.norm(phi)
    return (np.outer(phi, c.conj()) + np.outer(c, phi.conj())) / _SQRT2


def retract_vector(c: np.ndarray, V: np.ndarray, tau: float) -> np.ndarray:
    """Retract along tangent V by step tau; returns a unit state vector."""
    c = np.asarray(c, dtype=complex) / np.linalg.norm(c)
    phi = V @ c  # = (1/sqrt2)|phi>
    n = np.linalg.norm(phi)
    if n < 1e-15:
        return c.copy()
    phi = phi / n
    return np.cos(tau / _SQRT2) * c + np.sin(tau / _SQRT2) * phi


def symmetric_pair(c: np.ndarray, V: np.ndarray, tau: float):
    """Return density matrices (A+, A-) from +/- tau retractions."""
    return density(retract_vector(c, V, tau)), density(retract_vector(c, V, -tau))
