"""Canonical data containers and the label-quarantine mechanism.

Design goal: make the unsupervised contract *structural*, not aspirational.

* `Recording` is label-free. The model-fitting path only ever receives a
  `Recording`, so it is impossible to train on labels by construction.
* `LabelVault` seals damage labels. They can only be obtained via
  `reveal(consumer=...)`, which records who looked — the evaluator does this;
  the detector never gets a vault.
* `SplitMetadata` records exactly how windows were partitioned into the
  fit set and the evaluation set, including group-aware boundaries that prevent
  segment-level leakage (the 10 segments of one Z24 recording must not straddle
  the train/eval boundary).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Recording:
    """Label-free canonical container of vibration windows.

    Shapes: ``signals`` is ``(n_windows, n_sensors, n_timepoints)``.
    """

    signals: np.ndarray
    fs: float
    sensor_ids: tuple[int, ...]
    name: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.signals.ndim != 3:
            raise ValueError(
                f"signals must be 3D (n_windows, n_sensors, n_timepoints); "
                f"got shape {self.signals.shape}"
            )
        # Intentionally NO label attribute: this is what guarantees the fit path
        # cannot see labels.
        if any(name in self.meta for name in ("labels", "label", "y")):
            raise ValueError("Recording.meta must not carry labels (quarantine breach).")

    @property
    def n_windows(self) -> int:
        return int(self.signals.shape[0])

    @property
    def n_sensors(self) -> int:
        return int(self.signals.shape[1])

    @property
    def n_timepoints(self) -> int:
        return int(self.signals.shape[2])


class LabelVault:
    """Sealed container for evaluation labels.

    Labels are only obtainable through :meth:`reveal`, which logs the consumer.
    There is deliberately no public attribute exposing the raw labels, so any
    code that needs them must announce itself — the model-fitting code never
    does, and tests assert this.
    """

    def __init__(self, labels: np.ndarray, purpose: str = "evaluation") -> None:
        self.__labels = np.asarray(labels)
        self.purpose = purpose
        self._access_log: list[str] = []

    def reveal(self, consumer: str) -> np.ndarray:
        """Return a copy of the labels and record who requested them."""
        if not consumer:
            raise ValueError("reveal() requires a non-empty consumer name.")
        self._access_log.append(consumer)
        return self.__labels.copy()

    @property
    def n(self) -> int:
        return int(self.__labels.shape[0])

    @property
    def access_log(self) -> tuple[str, ...]:
        return tuple(self._access_log)


@dataclass(frozen=True)
class SplitMetadata:
    """Describes how windows were partitioned. Holds indices, not signals."""

    strategy: str
    healthy_labels: tuple[int, ...]
    segments_per_group: int
    seed: int
    train_indices: np.ndarray
    eval_indices: np.ndarray
    train_groups: tuple[int, ...]
    eval_groups: tuple[int, ...]
    notes: str = ""

    @property
    def n_train(self) -> int:
        return int(self.train_indices.shape[0])

    @property
    def n_eval(self) -> int:
        return int(self.eval_indices.shape[0])

    def assert_no_leakage(self) -> None:
        """Raise if train/eval share any window or any group (segment leakage)."""
        win_overlap = set(self.train_indices.tolist()) & set(self.eval_indices.tolist())
        if win_overlap:
            raise AssertionError(f"window-level leakage: {len(win_overlap)} shared indices")
        grp_overlap = set(self.train_groups) & set(self.eval_groups)
        if grp_overlap:
            raise AssertionError(f"group-level leakage: shared groups {sorted(grp_overlap)}")


@dataclass(frozen=True)
class DatasetSplit:
    """Bundle returned by loaders: fit-only data, eval data, sealed eval labels."""

    train: Recording  # healthy only, label-free
    eval: Recording  # healthy + damaged, label-free signals
    eval_labels: LabelVault  # sealed; evaluator-only
    split: SplitMetadata
