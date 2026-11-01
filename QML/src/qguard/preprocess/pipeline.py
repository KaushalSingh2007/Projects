"""Composable, deterministic preprocessing transforms and the Pipeline.

Transforms operate on arrays shaped ``(n_windows, n_sensors, n_timepoints)``.

Statefulness contract
----------------------
* Stateless transforms (`Detrend`, `BandpassFilter`, `Taper`) have a no-op `fit`.
* Stateful transforms (`PerChannelStandardizer`, `SpatialPCA`) learn parameters in
  `fit` and MUST be fit on healthy training signals only. The `Pipeline` enforces
  this by only ever calling `fit` from `Pipeline.fit`, which receives the healthy
  train `Recording`.
* All transforms are deterministic given their inputs (no RNG), so a fixed split
  seed makes the whole pipeline reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal

from ..data.schema import DatasetSplit, Recording


class Transform:
    """Base transform. Override `transform`; override `fit` if stateful."""

    def fit(self, x: np.ndarray) -> "Transform":  # noqa: D401 - simple contract
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def config(self) -> dict:
        return {"name": type(self).__name__}

    def state_summary(self) -> dict:
        """Small, JSON-able summary of fitted state for the manifest (no big arrays)."""
        return {}


@dataclass
class Detrend(Transform):
    """Remove a per-window, per-channel trend along the time axis."""

    kind: str = "constant"  # "constant" (mean) or "linear"

    def transform(self, x: np.ndarray) -> np.ndarray:
        return sp_signal.detrend(x, axis=-1, type=self.kind).astype(x.dtype, copy=False)

    def config(self) -> dict:
        return {"name": "Detrend", "kind": self.kind}


@dataclass
class BandpassFilter(Transform):
    """Zero-phase Butterworth bandpass along time. Disabled unless low/high set.

    NOTE: depends on `fs`; with the assumed Z24 fs=100 Hz the usable band is
    < 50 Hz. Off by default to avoid imposing an unvalidated band (see OQ-fs).
    """

    low: float | None = None
    high: float | None = None
    fs: float = 100.0
    order: int = 4

    def _sos(self):
        nyq = 0.5 * self.fs
        if self.low and self.high:
            return sp_signal.butter(self.order, [self.low / nyq, self.high / nyq],
                                    btype="band", output="sos")
        if self.high:
            return sp_signal.butter(self.order, self.high / nyq, btype="low", output="sos")
        if self.low:
            return sp_signal.butter(self.order, self.low / nyq, btype="high", output="sos")
        return None

    def transform(self, x: np.ndarray) -> np.ndarray:
        sos = self._sos()
        if sos is None:
            return x
        return sp_signal.sosfiltfilt(sos, x, axis=-1).astype(x.dtype, copy=False)

    def config(self) -> dict:
        return {"name": "BandpassFilter", "low": self.low, "high": self.high,
                "fs": self.fs, "order": self.order}


@dataclass
class Taper(Transform):
    """Apply a spectral window (taper) along time before frequency-domain features.

    The Z24 mirror already segments signals into fixed 6000-sample analysis
    windows; we adopt those as the analysis window and optionally taper them.
    """

    kind: str = "none"  # "none" | "hann" | "hamming"

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "none":
            return x
        win = {"hann": np.hanning, "hamming": np.hamming}[self.kind](x.shape[-1])
        return (x * win.astype(x.dtype)).astype(x.dtype, copy=False)

    def config(self) -> dict:
        return {"name": "Taper", "kind": self.kind}


@dataclass
class PerChannelStandardizer(Transform):
    """Standardize each sensor channel using statistics from TRAINING data only."""

    mean_: np.ndarray | None = field(default=None, repr=False)
    std_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, x: np.ndarray) -> "PerChannelStandardizer":
        # Aggregate over windows and time -> one (mean, std) per sensor channel.
        self.mean_ = x.mean(axis=(0, 2)).astype("float64")
        std = x.std(axis=(0, 2)).astype("float64")
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("PerChannelStandardizer used before fit().")
        m = self.mean_[None, :, None]
        s = self.std_[None, :, None]
        return ((x - m) / s).astype(x.dtype, copy=False)

    def config(self) -> dict:
        return {"name": "PerChannelStandardizer", "scope": "per-sensor"}

    def state_summary(self) -> dict:
        if self.mean_ is None:
            return {"fitted": False}
        return {"fitted": True, "n_channels": int(self.mean_.shape[0]),
                "mean_l2": float(np.linalg.norm(self.mean_)),
                "std_l2": float(np.linalg.norm(self.std_))}


@dataclass
class SpatialPCA(Transform):
    """Optional Sohn-style spatial compression across sensors (fit on train).

    Treats each (window, timepoint) as a sample of the sensor vector in R^C and
    projects onto the top-k eigenvectors of the training sensor covariance. PCA
    is unsupervised; it never sees labels, and is fit on healthy train only.
    Disabled when ``n_components`` is None (identity passthrough).
    """

    n_components: int | None = None
    components_: np.ndarray | None = field(default=None, repr=False)  # (C, k)
    mean_: np.ndarray | None = field(default=None, repr=False)  # (C,)

    def fit(self, x: np.ndarray) -> "SpatialPCA":
        if self.n_components is None:
            return self
        n, c, t = x.shape
        samples = np.transpose(x, (0, 2, 1)).reshape(n * t, c).astype("float64")
        self.mean_ = samples.mean(axis=0)
        cov = np.cov((samples - self.mean_), rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
        self.components_ = eigvecs[:, ::-1][:, : self.n_components]  # (C, k)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.n_components is None:
            return x
        if self.components_ is None:
            raise RuntimeError("SpatialPCA used before fit().")
        n, c, t = x.shape
        samples = np.transpose(x, (0, 2, 1)).reshape(n * t, c) - self.mean_
        proj = samples @ self.components_  # (n*t, k)
        return np.transpose(proj.reshape(n, t, self.n_components), (0, 2, 1)).astype(x.dtype)

    def config(self) -> dict:
        return {"name": "SpatialPCA", "n_components": self.n_components}

    def state_summary(self) -> dict:
        if self.n_components is None:
            return {"enabled": False}
        return {"enabled": True, "n_components": int(self.n_components),
                "fitted": self.components_ is not None}


class Pipeline:
    """Ordered transforms; fit ONLY on the healthy train Recording."""

    def __init__(self, transforms: list[Transform], seed: int | None = None):
        self.transforms = transforms
        self.seed = seed
        self._fitted = False

    def fit(self, train: Recording) -> "Pipeline":
        if not isinstance(train, Recording):
            raise TypeError("Pipeline.fit must receive a label-free Recording.")
        x = np.asarray(train.signals)
        for t in self.transforms:
            t.fit(x)
            x = t.transform(x)
        self._fitted = True
        return self

    def transform(self, rec: Recording) -> Recording:
        if not self._fitted:
            raise RuntimeError("Pipeline.transform called before fit().")
        x = np.asarray(rec.signals)
        for t in self.transforms:
            x = t.transform(x)
        new_sensors = tuple(range(x.shape[1]))  # may change if SpatialPCA active
        return Recording(
            signals=x, fs=rec.fs, sensor_ids=new_sensors, name=f"{rec.name}+pp",
            meta={**rec.meta, "preprocessed": True},
        )

    def fit_transform_split(self, split: DatasetSplit) -> DatasetSplit:
        """Fit on healthy train; transform both; keep eval labels sealed and the
        leakage-free split metadata unchanged."""
        self.fit(split.train)
        return DatasetSplit(
            train=self.transform(split.train),
            eval=self.transform(split.eval),
            eval_labels=split.eval_labels,  # vault untouched
            split=split.split,  # indices/groups unchanged -> no-leakage preserved
        )

    def manifest(self) -> dict:
        return {
            "seed": self.seed,
            "fitted": self._fitted,
            "transforms": [t.config() for t in self.transforms],
            "state": [t.state_summary() for t in self.transforms],
        }


def build_pipeline(config: dict | None = None, fs: float = 100.0,
                   seed: int | None = None) -> Pipeline:
    """Construct a Pipeline from a plain dict (Hydra-friendly).

    Default pipeline: linear detrend -> (no bandpass) -> (no taper) ->
    per-channel standardization (healthy-fit) -> (no spatial PCA).
    """
    config = config or {}
    transforms: list[Transform] = []

    dt = config.get("detrend", {"kind": "linear"})
    if dt:
        transforms.append(Detrend(kind=dt.get("kind", "linear")))

    bp = config.get("bandpass")
    if bp:
        transforms.append(BandpassFilter(low=bp.get("low"), high=bp.get("high"),
                                        fs=bp.get("fs", fs), order=bp.get("order", 4)))

    tp = config.get("taper")
    if tp:
        transforms.append(Taper(kind=tp.get("kind", "none")))

    if config.get("standardize", True):
        transforms.append(PerChannelStandardizer())

    pca = config.get("spatial_pca")
    if pca and pca.get("n_components"):
        transforms.append(SpatialPCA(n_components=pca["n_components"]))

    return Pipeline(transforms, seed=seed)
