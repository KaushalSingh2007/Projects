"""Label-quarantine guarantees (M2).

Enforces that the model-fitting path cannot see damage labels:
* `Recording` (the only object the detector receives) exposes no labels.
* Labels live in `LabelVault` and are only obtainable via `reveal(consumer=...)`.
"""

import numpy as np
import pytest

from qguard.data.schema import LabelVault, Recording


def _toy_recording(n=20, c=3, t=64):
    rng = np.random.default_rng(0)
    return Recording(
        signals=rng.standard_normal((n, c, t)).astype("float32"),
        fs=100.0,
        sensor_ids=tuple(range(c)),
        name="toy",
    )


def test_recording_has_no_label_attribute():
    rec = _toy_recording()
    for forbidden in ("labels", "label", "y", "targets"):
        assert not hasattr(rec, forbidden)


def test_recording_meta_rejects_labels():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        Recording(
            signals=rng.standard_normal((4, 2, 8)).astype("float32"),
            fs=100.0,
            sensor_ids=(0, 1),
            name="bad",
            meta={"labels": [0, 1, 1, 0]},
        )


def test_labels_only_via_reveal_and_access_is_logged():
    vault = LabelVault(np.array([0, 0, 1, 2, 3]))
    assert vault.access_log == ()
    y = vault.reveal(consumer="evaluator")
    assert y.tolist() == [0, 0, 1, 2, 3]
    assert vault.access_log == ("evaluator",)
    # No public attribute leaks the raw labels.
    assert not hasattr(vault, "labels")


def test_reveal_requires_named_consumer():
    vault = LabelVault(np.array([0, 1]))
    with pytest.raises(ValueError):
        vault.reveal(consumer="")
