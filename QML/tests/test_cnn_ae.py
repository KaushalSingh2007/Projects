"""M6 CNN-AE smoke: trains on healthy, flags shifted anomaly, deterministic."""

import numpy as np

from qguard.baselines import CNNAEDetector
from qguard.data.schema import Recording


def _rec(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return Recording((rng.standard_normal((n, 4, 256)).astype("float32") + shift),
                     100.0, tuple(range(4)), "r")


def test_cnn_ae_scores_and_determinism():
    train = _rec(40)
    det = CNNAEDetector(epochs=5).fit(train)
    s_h = det.score(_rec(20, 0.0, seed=1))
    s_a = det.score(_rec(20, 3.0, seed=2))
    assert s_a.mean() > s_h.mean()
    assert det.score(train).shape == (40,)
    s_again = CNNAEDetector(epochs=5).fit(train).score(train)
    assert np.allclose(s_again, det.score(train), atol=1e-5)
