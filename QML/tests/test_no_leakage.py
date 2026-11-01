"""No-leakage guarantees for the train/eval split (M2).

Two distinct leakage modes are checked:
1. Window-level: no window index appears in both train and eval.
2. Group-level (the subtle one): the 10 segments of a single Z24 recording must
   not straddle the train/eval boundary, or correlated segments would leak.
Also checks the unsupervised contract at the split level: train is healthy-only.
"""

import numpy as np

from qguard.data.loaders.z24 import group_ids_for, make_split


def _synthetic_z24(n_scenarios=5, n_setups=4, segs=10):
    """Build a tiny Z24-shaped label vector: scenario-major, then setup, then segment."""
    n = n_scenarios * n_setups * segs
    labels = np.repeat(np.arange(n_scenarios), n_setups * segs)
    groups = group_ids_for(n, segments_per_group=segs)
    return labels, groups, segs


def test_split_is_window_and_group_disjoint():
    labels, groups, segs = _synthetic_z24()
    split = make_split(labels, groups, healthy_labels=(0,), seed=7, segments_per_group=segs)
    # Raises on any window- or group-level overlap.
    split.assert_no_leakage()
    assert split.n_train > 0 and split.n_eval > 0


def test_train_is_healthy_only():
    labels, groups, segs = _synthetic_z24()
    split = make_split(labels, groups, healthy_labels=(0,), seed=7, segments_per_group=segs)
    train_labels = labels[split.train_indices]
    assert set(train_labels.tolist()) == {0}


def test_eval_contains_healthy_and_damaged():
    labels, groups, segs = _synthetic_z24()
    split = make_split(labels, groups, healthy_labels=(0,), seed=7, segments_per_group=segs)
    eval_labels = set(labels[split.eval_indices].tolist())
    assert 0 in eval_labels  # held-out healthy for the ROC negative class
    assert len(eval_labels - {0}) > 0  # damaged classes present


def test_no_segment_straddles_boundary():
    labels, groups, segs = _synthetic_z24()
    split = make_split(labels, groups, healthy_labels=(0,), seed=7, segments_per_group=segs)
    train_groups = set(groups[split.train_indices].tolist())
    eval_groups = set(groups[split.eval_indices].tolist())
    assert train_groups.isdisjoint(eval_groups)


def test_split_is_deterministic_under_seed():
    labels, groups, segs = _synthetic_z24()
    a = make_split(labels, groups, seed=42, segments_per_group=segs)
    b = make_split(labels, groups, seed=42, segments_per_group=segs)
    assert a.train_indices.tolist() == b.train_indices.tolist()
    assert a.eval_indices.tolist() == b.eval_indices.tolist()
