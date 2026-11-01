"""M7: feature-map + fidelity-kernel validation."""

import numpy as np
import pytest

pytest.importorskip("pennylane")

from qguard.quantum import fidelity_kernel, statevectors  # noqa: E402


@pytest.mark.parametrize("kind", ["angle", "zz"])
def test_kernel_properties(kind):
    rng = np.random.default_rng(0)
    n_qubits = 4
    X = rng.uniform(0, np.pi, size=(15, n_qubits))
    V = statevectors(X, kind, n_qubits, reps=2)
    K = fidelity_kernel(V)
    assert K.shape == (15, 15)
    assert np.allclose(K, K.T, atol=1e-10)          # symmetric
    assert np.allclose(np.diag(K), 1.0, atol=1e-8)   # unit diagonal
    assert K.min() > -1e-10 and K.max() < 1.0 + 1e-8  # in [0,1]
    assert np.linalg.eigvalsh(K).min() > -1e-8        # PSD


def test_identical_inputs_have_fidelity_one():
    x = np.array([0.3, 1.1, 2.0, 0.7])
    V = statevectors(np.stack([x, x]), "zz", 4)
    assert abs(fidelity_kernel(V)[0, 1] - 1.0) < 1e-8
