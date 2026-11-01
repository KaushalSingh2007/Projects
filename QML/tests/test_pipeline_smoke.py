"""Smoke tests for the M0 scaffold.

These intentionally depend only on numpy + the package itself, so they run in a
minimal environment (and in CI) without pennylane/torch installed. Heavier,
component-specific tests arrive with their milestones (M2+).
"""

import numpy as np

import qguard


def test_package_imports_and_has_version():
    assert isinstance(qguard.__version__, str)
    assert qguard.__version__.count(".") >= 2


def test_numpy_available_and_sane():
    # Guards against a broken scientific stack before real pipeline code lands.
    x = np.linspace(0.0, 1.0, 11)
    assert x.shape == (11,)
    assert np.isclose(x.mean(), 0.5)


def test_label_quarantine_placeholder():
    """Placeholder asserting the unsupervised contract we will enforce for real
    in tests/test_label_quarantine.py at M2: fitting must never touch labels.
    """
    healthy_only_fit = True  # to be replaced by a real fit-path assertion
    assert healthy_only_fit
