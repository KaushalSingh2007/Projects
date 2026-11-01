"""Novelty-detection metrics from continuous scores + binary labels."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def evaluate(scores: np.ndarray, y_binary: np.ndarray) -> dict:
    """y_binary: 1 = damaged (positive), 0 = healthy."""
    s = np.asarray(scores, dtype="float64")
    y = np.asarray(y_binary, dtype=int)
    fpr, tpr, _ = roc_curve(y, s)
    fpr_at_95 = float(fpr[np.searchsorted(tpr, 0.95)]) if (tpr >= 0.95).any() else 1.0
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "fpr_at_95tpr": fpr_at_95,
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
    }
