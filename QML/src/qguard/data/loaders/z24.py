"""Z24 Bridge loader: canonical split with group-aware, leakage-free partition.

Label semantics (resolved during M2)
-------------------------------------
The HuggingFace mirror stores ``labels.npy`` as integers 0..16 (one per scenario
of the Z24 Progressive Damage Test, PDT), with 90 windows each (9 setups x 10
segments). The mirror does NOT publish the integer->scenario mapping, so we use
the canonical PDT ordering from the Z24 literature (Reynders et al. 2008;
Krämer, De Roeck & Reynders). The first scenarios are reference/undamaged
states; damage is introduced progressively.

IMPORTANT (honesty): the alignment between the mirror's integer index and the
canonical scenario below is the *most likely* one (scenario-major reshape, in
order) but is **not documented by the mirror**. We therefore:
  * default the healthy/reference class to label 0, and
  * mark the mapping as TO-BE-EMPIRICALLY-VALIDATED in EDA (M4): the healthy
    class should form the tightest, most self-consistent feature cluster.
The healthy label set is configurable, so this assumption is never hard-coded
into results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import DatasetSplit, LabelVault, Recording, SplitMetadata

# Canonical Z24 PDT scenario order (index -> description). Documented assumption.
Z24_SCENARIO_NAMES: dict[int, str] = {
    0: "reference / undamaged (initial)",
    1: "reference (settlement system installed)",
    2: "pier settlement, 20 mm",
    3: "pier settlement, 40 mm",
    4: "pier settlement, 80 mm",
    5: "pier settlement, 95 mm",
    6: "tilt of foundation",
    7: "new reference (pier lifted back)",
    8: "spalling of concrete, ~24 m^2",
    9: "spalling of concrete, ~12 m^2",
    10: "landslide at abutment",
    11: "concrete hinge failure",
    12: "anchor head failure I",
    13: "anchor head failure II",
    14: "tendon rupture, 2 wires",
    15: "tendon rupture, 4 wires",
    16: "tendon rupture, 6 wires",
}

# Z24 measurements were sampled at 100 Hz in the benchmark. The mirror does not
# restate fs; we record it as a DOCUMENTED ASSUMPTION used for the PSD frequency
# axis (M4). See OQ-fs in PROGRESS_LOG.md.
DEFAULT_FS_HZ = 100.0
DEFAULT_SEGMENTS_PER_GROUP = 10
DEFAULT_HEALTHY_LABELS = (0,)
DEFAULT_HEALTHY_EVAL_FRAC = 0.3


def load_raw(raw_dir: str | Path = "data/raw/z24", mmap: bool = True):
    """Load (signals, labels) arrays from disk. Uses mmap to avoid a ~1 GB copy."""
    raw_dir = Path(raw_dir)
    inputs_p = raw_dir / "inputs.npy"
    labels_p = raw_dir / "labels.npy"
    if not inputs_p.exists() or not labels_p.exists():
        raise FileNotFoundError(
            f"Z24 arrays not found in {raw_dir}. Run: python -m qguard.data.download_z24"
        )
    signals = np.load(inputs_p, mmap_mode="r" if mmap else None)
    labels = np.load(labels_p)
    return signals, labels


def group_ids_for(n_windows: int, segments_per_group: int = DEFAULT_SEGMENTS_PER_GROUP):
    """Map each window to its source recording group ((scenario, setup) -> id).

    The reshape is scenario-major then setup then segment, so 10 consecutive
    windows share one continuous recording. Grouping by ``i // 10`` keeps a
    recording's segments together, preventing segment-level train/eval leakage.
    """
    return np.arange(n_windows) // segments_per_group


def make_split(
    labels: np.ndarray,
    group_ids: np.ndarray,
    healthy_labels: tuple[int, ...] = DEFAULT_HEALTHY_LABELS,
    healthy_eval_frac: float = DEFAULT_HEALTHY_EVAL_FRAC,
    seed: int = 1234,
    segments_per_group: int = DEFAULT_SEGMENTS_PER_GROUP,
) -> SplitMetadata:
    """Group-aware split: healthy groups -> {train, eval}; all damaged -> eval.

    Pure function over labels + group ids (no signal I/O) so it is unit-testable
    on tiny synthetic arrays.
    """
    labels = np.asarray(labels)
    group_ids = np.asarray(group_ids)
    healthy_set = set(int(h) for h in healthy_labels)

    # Each group has a single label (constant within a recording).
    unique_groups = np.unique(group_ids)
    group_label = {int(g): int(labels[group_ids == g][0]) for g in unique_groups}

    healthy_groups = np.array([g for g in unique_groups if group_label[int(g)] in healthy_set])
    damaged_groups = np.array([g for g in unique_groups if group_label[int(g)] not in healthy_set])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(healthy_groups)
    n_eval_h = max(1, int(round(healthy_eval_frac * len(perm)))) if len(perm) > 1 else 0
    eval_healthy_groups = perm[:n_eval_h]
    train_groups = perm[n_eval_h:]
    eval_groups = np.concatenate([eval_healthy_groups, damaged_groups]) if len(damaged_groups) \
        else eval_healthy_groups

    train_mask = np.isin(group_ids, train_groups)
    eval_mask = np.isin(group_ids, eval_groups)

    return SplitMetadata(
        strategy="group-aware-healthy-only-train",
        healthy_labels=tuple(sorted(healthy_set)),
        segments_per_group=segments_per_group,
        seed=seed,
        train_indices=np.where(train_mask)[0],
        eval_indices=np.where(eval_mask)[0],
        train_groups=tuple(sorted(int(g) for g in train_groups)),
        eval_groups=tuple(sorted(int(g) for g in eval_groups)),
        notes=(
            "Healthy recordings split into train/eval at the GROUP level so a "
            "recording's 10 segments never straddle the boundary; all damaged "
            "recordings are eval-only. Healthy labels: "
            f"{tuple(sorted(healthy_set))}."
        ),
    )


def load_z24(
    raw_dir: str | Path = "data/raw/z24",
    fs: float = DEFAULT_FS_HZ,
    healthy_labels: tuple[int, ...] = DEFAULT_HEALTHY_LABELS,
    healthy_eval_frac: float = DEFAULT_HEALTHY_EVAL_FRAC,
    seed: int = 1234,
    mmap: bool = True,
) -> DatasetSplit:
    """Load Z24 and return a leakage-free `DatasetSplit` (fit-only train, sealed eval labels)."""
    signals, labels = load_raw(raw_dir, mmap=mmap)
    n = signals.shape[0]
    groups = group_ids_for(n)
    split = make_split(labels, groups, healthy_labels, healthy_eval_frac, seed)
    sensor_ids = tuple(range(signals.shape[1]))

    base_meta = {
        "dataset": "Z24",
        "fs_assumed": fs is DEFAULT_FS_HZ,
        "scenario_names": Z24_SCENARIO_NAMES,
        "healthy_labels": tuple(sorted(set(int(h) for h in healthy_labels))),
    }
    train = Recording(
        signals=np.asarray(signals[split.train_indices]),
        fs=fs,
        sensor_ids=sensor_ids,
        name="z24-train-healthy",
        meta={**base_meta, "role": "train", "label_free": True},
    )
    evalr = Recording(
        signals=np.asarray(signals[split.eval_indices]),
        fs=fs,
        sensor_ids=sensor_ids,
        name="z24-eval",
        meta={**base_meta, "role": "eval"},
    )
    eval_labels = LabelVault(np.asarray(labels)[split.eval_indices], purpose="evaluation")
    return DatasetSplit(train=train, eval=evalr, eval_labels=eval_labels, split=split)
