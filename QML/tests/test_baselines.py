"""M5 baseline + decision + metric sanity."""

import numpy as np

from qguard.decision import classify, spc_threshold
from qguard.eval import evaluate
from qguard.scorers import MahalanobisDetector


def test_mahalanobis_separates_shifted_anomaly():
    rng = np.random.default_rng(0)
    healthy = rng.standard_normal((300, 6))
    anomaly = rng.standard_normal((100, 6)) + 5.0
    det = MahalanobisDetector().fit(healthy)
    s_h, s_a = det.score(healthy), det.score(anomaly)
    assert s_a.mean() > s_h.mean()
    scores = np.concatenate([s_h, s_a])
    y = np.concatenate([np.zeros(300), np.ones(100)])
    assert evaluate(scores, y)["roc_auc"] > 0.95


def test_spc_threshold_and_classify():
    rng = np.random.default_rng(1)
    train = rng.standard_normal(1000)
    ucl = spc_threshold(train, 0.99)
    assert classify(np.array([ucl + 1]), ucl)[0] == 1
    assert classify(np.array([ucl - 10]), ucl)[0] == 0
