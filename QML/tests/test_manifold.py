"""M8+M9 geometric validation: projector, retraction, curvature-cancellation, MoM."""

import numpy as np

from qguard.quantum.manifold import (
    density,
    median_of_means,
    retract_vector,
    symmetric_pair,
    tangent_from_phi,
    tangent_projector,
)


def _base_and_tangent(d=5, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    c = c / np.linalg.norm(c)
    phi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    V = tangent_from_phi(c, phi)
    return c, density(c), V


def test_projector_idempotent_selfadjoint_and_kills_base():
    c, C, V = _base_and_tangent()
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    X = (X + X.conj().T) / 2
    P = tangent_projector(C, X)
    assert np.allclose(tangent_projector(C, P), P, atol=1e-10)        # idempotent
    Y = (X + 0.3 * V)
    lhs = np.vdot(Y, tangent_projector(C, X))
    rhs = np.vdot(tangent_projector(C, Y), X)
    assert abs(lhs - rhs) < 1e-9                                      # self-adjoint
    assert np.linalg.norm(tangent_projector(C, C)) < 1e-10           # P(C)=0
    assert np.allclose(tangent_projector(C, V), V, atol=1e-9)        # identity on tangent


def test_retraction_on_manifold_and_fidelity():
    c, C, V = _base_and_tangent()
    g = retract_vector(c, V, 0.7)
    assert abs(np.linalg.norm(g) - 1.0) < 1e-10                       # unit -> on manifold
    fid = abs(np.vdot(c, g)) ** 2
    assert abs(fid - np.cos(0.7 / np.sqrt(2)) ** 2) < 1e-8


def test_symmetric_difference_cancels_curvature():
    c, C, V = _base_and_tangent()
    Ap, Am = symmetric_pair(c, V, 0.4)
    even = tangent_projector(C, Ap + Am - 2 * C)   # curvature/even part
    odd = tangent_projector(C, Ap - Am)            # tangent/odd part
    assert np.linalg.norm(even) < 1e-8             # curvature cancels
    assert np.linalg.norm(odd) > 1e-3              # tangent info survives


def test_numerical_stability_small_and_large_tau():
    c, C, V = _base_and_tangent(d=8)
    for tau in (1e-6, 3.0):
        g = retract_vector(c, V, tau)
        assert abs(np.linalg.norm(g) - 1.0) < 1e-9


def test_mom_robust_to_outliers():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((200, 1))
    x[:20] += 50.0  # 10% gross outliers
    mom = median_of_means(x, n_groups=10)[0]
    assert abs(mom) < abs(x.mean()) and abs(mom) < 0.5
