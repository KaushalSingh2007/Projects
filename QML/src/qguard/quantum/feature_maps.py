"""Quantum feature maps: x -> |psi(x)> on CP^(2^n - 1) (statevector sim).

* "angle": single-layer RY angle encoding (product state, no entanglement).
* "zz":    IQP-style entangling map (Havlicek), ZZ interactions, repeatable.

Requires ``pennylane`` (install extra: ``pip install -e ".[quantum]"``).
"""

from __future__ import annotations

import numpy as np


def make_statevector_fn(kind: str, n_qubits: int, reps: int = 2):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        if kind == "angle":
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        elif kind == "zz":
            qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=reps)
        else:
            raise ValueError(f"unknown feature map: {kind}")
        return qml.state()

    return circuit


def statevectors(X: np.ndarray, kind: str, n_qubits: int, reps: int = 2) -> np.ndarray:
    """Return (n_samples, 2**n_qubits) complex statevectors for each row of X."""
    fn = make_statevector_fn(kind, n_qubits, reps)
    return np.stack([np.asarray(fn(np.asarray(x, dtype="float64"))) for x in X])
