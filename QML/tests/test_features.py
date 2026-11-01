"""M4 feature-extraction validation: shapes, determinism, correctness, healthy-fit."""

import numpy as np

from qguard.data.schema import Recording
from qguard.features import ARFeatures, FeaturePCA, PSDFeatures, ar_yule_walker, welch_psd


def _rec(n=12, c=4, t=1024, fs=100.0, seed=0):
    rng = np.random.default_rng(seed)
    return Recording(rng.standard_normal((n, c, t)).astype("float32"), fs, tuple(range(c)), "r")


def test_psd_shape_and_determinism():
    rec = _rec()
    f = PSDFeatures(nperseg=256, n_bins=32)
    a, b = f.transform(rec), f.transform(rec)
    assert a.shape == (12, 32)
    assert np.array_equal(a, b)


def test_psd_peak_at_known_frequency():
    fs, t = 100.0, 1024
    tt = np.arange(t) / fs
    sig = np.sin(2 * np.pi * 10.0 * tt)[None, None, :].repeat(3, axis=1)  # 10 Hz, 1 win, 3 sens
    freqs, pxx = welch_psd(sig.astype("float64"), fs=fs, nperseg=256)
    assert abs(freqs[np.argmax(pxx[0])] - 10.0) < 1.0


def test_ar_shape_and_recovers_ar1():
    rng = np.random.default_rng(1)
    t, phi = 4000, 0.6
    x = np.zeros(t)
    e = rng.standard_normal(t)
    for k in range(1, t):
        x[k] = phi * x[k - 1] + e[k]
    a = ar_yule_walker(x[None, :], order=1)
    assert a.shape == (1, 1)
    assert abs(a[0, 0] - phi) < 0.1
    feats = ARFeatures(order=4).transform(_rec())
    assert feats.shape == (12, 4)


def test_feature_pca_fits_on_train_only():
    rng = np.random.default_rng(2)
    x_train = rng.standard_normal((40, 16))
    x_eval = rng.standard_normal((50, 16))
    pca = FeaturePCA(n_components=5).fit(x_train)
    comp_before = pca._pca.components_.copy()
    pca.transform(x_eval)
    assert np.array_equal(pca._pca.components_, comp_before)
    assert pca.transform(x_eval).shape == (50, 5)
