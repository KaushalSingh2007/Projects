"""Exact state-fidelity kernel: K(x,y) = |<psi(x)|psi(y)>|^2."""

from __future__ import annotations

import numpy as np


def fidelity_kernel(va: np.ndarray, vb: np.ndarray | None = None) -> np.ndarray:
    """Gram matrix of fidelities between statevector sets `va` (and `vb`).

    For pure states this is real, symmetric (when vb is None), PSD, in [0, 1],
    with unit diagonal.
    """
    vb = va if vb is None else vb
    overlaps = np.conjugate(va) @ vb.T
    return np.abs(overlaps) ** 2
