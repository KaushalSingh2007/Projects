"""Statistical-process-control thresholding on healthy novelty scores."""

from __future__ import annotations

import numpy as np


def spc_threshold(train_scores: np.ndarray, quantile: float = 0.99) -> float:
    """Upper control limit from the healthy-train score distribution."""
    return float(np.quantile(np.asarray(train_scores), quantile))


def classify(scores: np.ndarray, ucl: float) -> np.ndarray:
    """1 = out-of-control (flagged damage), 0 = in-control."""
    return (np.asarray(scores) > ucl).astype(int)
