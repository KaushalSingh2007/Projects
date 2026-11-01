"""M3 preprocessing validation.

Covers the stated requirements:
* scaler fit only on healthy TRAINING data (eval never influences fit),
* spatial PCA fit on train only (never sees eval/labels),
* no overlap between train/eval windows is preserved through preprocessing+cache,
* all transforms reproducible (deterministic) given a fixed split,
* fit path receives a label-free Recording.
"""

import numpy as np
import pytest

from qguard.data.schema import DatasetSplit, LabelVault, Recording
from qguard.data.loaders.z24 import make_split, group_ids_for
from qguard.preprocess import (
    Detrend,
    PerChannelStandardizer,
    SpatialPCA,
    build_pipeline,
)
from qguard.preprocess.cache import cache_split, load_cached


def _toy_split(seed=0, c=3, t=64):
    """Synthetic Z24-shaped split: 8 groups x 10 windows; labels 0 (healthy) and 1."""
    labels = np.array([0] * 50 + [1] * 30)  # 8 groups of 10
    groups = group_ids_for(80, segments_per_group=10)
    split = make_split(labels, groups, healthy_labels=(0,), seed=seed, segments_per_group=10)
    rng = np.random.default_rng(seed)
    signals = (rng.standard_normal((80, c, t)).astype("float32") * 2.0 + 5.0)
    train = Recording(signals[split.train_indices], 100.0, tuple(range(c)), "tr")
    evalr = Recording(signals[split.eval_indices], 100.0, tuple(range(c)), "ev")
    vault = LabelVault(labels[split.eval_indices])
    return DatasetSplit(train, evalr, vault, split), signals


def test_pipeline_fit_requires_recording():
    pipe = build_pipeline()
    with pytest.raises(TypeError):
        pipe.fit(np.zeros((4, 3, 16), dtype="float32"))


def test_scaler_fits_on_train_only():
    pipe = build_pipeline({"detrend": {"kind": "constant"}, "standardize": True})
    split, _ = _toy_split()
    pipe.fit(split.train)
    scaler = next(t for t in pipe.transforms if isinstance(t, PerChannelStandardizer))
    mean_before = scaler.mean_.copy()
    std_before = scaler.std_.copy()
    # Transforming eval must NOT change fitted statistics.
    pipe.transform(split.eval)
    assert np.array_equal(scaler.mean_, mean_before)
    assert np.array_equal(scaler.std_, std_before)


def test_standardized_train_has_unit_scale():
    pipe = build_pipeline({"detrend": {"kind": "constant"}, "standardize": True})
    split, _ = _toy_split()
    pipe.fit(split.train)
    out = pipe.transform(split.train).signals
    per_ch_std = out.std(axis=(0, 2))
    assert np.allclose(per_ch_std, 1.0, atol=1e-2)


def test_detrend_removes_window_mean():
    x = np.random.default_rng(1).standard_normal((5, 3, 50)).astype("float32") + 10.0
    out = Detrend(kind="constant").transform(x)
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-4)


def test_pipeline_is_deterministic():
    split, _ = _toy_split()
    a = build_pipeline().fit_transform_split(split)
    b = build_pipeline().fit_transform_split(split)
    assert np.array_equal(a.train.signals, b.train.signals)
    assert np.array_equal(a.eval.signals, b.eval.signals)


def test_spatial_pca_fits_on_train_only_and_reduces_channels():
    cfg = {"detrend": {"kind": "constant"}, "standardize": False,
           "spatial_pca": {"n_components": 2}}
    split, _ = _toy_split(c=3)
    out = build_pipeline(cfg).fit_transform_split(split)
    assert out.train.signals.shape[1] == 2
    assert out.eval.signals.shape[1] == 2

    # PCA basis must depend only on train signals (not eval/labels): fitting the
    # transform directly on the detrended train data reproduces it exactly.
    pca_pipe = build_pipeline(cfg)
    pca_pipe.fit(split.train)
    pca = next(t for t in pca_pipe.transforms if isinstance(t, SpatialPCA))
    detrended_train = Detrend(kind="constant").transform(np.asarray(split.train.signals))
    ref = SpatialPCA(n_components=2).fit(detrended_train)
    assert np.allclose(pca.components_, ref.components_)


def test_preprocessing_preserves_no_leakage():
    split, _ = _toy_split()
    out = build_pipeline().fit_transform_split(split)
    out.split.assert_no_leakage()  # raises on any window/group overlap
    assert out.split.train_indices.tolist() == split.split.train_indices.tolist()


def test_cache_roundtrip_and_quarantine(tmp_path):
    split, _ = _toy_split()
    pipe = build_pipeline()
    pp = pipe.fit_transform_split(split)
    cache_split(pp, pipe, out_dir=tmp_path)
    loaded = load_cached(tmp_path)

    assert np.array_equal(loaded.train.signals, pp.train.signals)
    assert np.array_equal(loaded.eval.signals, pp.eval.signals)
    # Eval labels survive sealed and reveal correctly.
    assert loaded.eval_labels.reveal("test").tolist() == \
        split.eval_labels.reveal("ref").tolist()
    # No-leakage preserved through the cache round-trip.
    loaded.split.assert_no_leakage()
    assert loaded.train.signals.ndim == 3
