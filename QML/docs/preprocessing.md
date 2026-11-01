# Preprocessing (M3) — design & decisions

Module: [`src/qguard/preprocess/`](../src/qguard/preprocess). Every transform is
**deterministic** and the stateful ones are **fit on the healthy training set
only**. This document records each decision and its rationale so results are
reproducible and auditable.

## Pipeline contract

```
DatasetSplit (from load_z24)               # train = healthy-only & label-free
        │
        ▼
Pipeline.fit(split.train)                  # fit stateful transforms on HEALTHY TRAIN ONLY
        │
        ▼
Pipeline.transform(train), .transform(eval)   # same fitted params applied to both
        │
        ▼
cache_split(...)  ->  data/processed/z24/  # transformed arrays + manifest + sealed eval labels
```

`Pipeline.fit` accepts a `Recording`, which by construction carries **no labels**.
Therefore no statistic (scaler mean/std, PCA basis) can be influenced by
evaluation data or by labels. `fit_transform_split` fits on `split.train`, then
transforms both splits and leaves `SplitMetadata` (indices/groups) untouched, so
the group-aware, leakage-free partition from M2 is preserved (asserted by tests).

## Transforms (in default order)

| Order | Transform | Stateful? | Default | Rationale |
|------:|-----------|:---------:|---------|-----------|
| 1 | `Detrend(kind="linear")` | no | on | Remove per-window/per-channel drift; baseline wander is not damage-informative. |
| 2 | `BandpassFilter(low, high, fs)` | no | **off** | Available, but no band imposed until `fs` is validated (see OQ-fs). Zero-phase (`sosfiltfilt`). |
| 3 | `Taper(kind)` | no | **off** (`none`) | Optional Hann/Hamming taper for spectral leakage control before PSD (M4). |
| 4 | `PerChannelStandardizer` | **yes** | on | Per-sensor zero-mean/unit-var using TRAIN stats. Critical: amplitude scaling is *not* done by the mirror (see below). |
| 5 | `SpatialPCA(n_components)` | **yes** | **off** (None) | Optional Sohn-style spatial compression across the 27 sensors; unsupervised, train-fit. Kept off by default so the per-sensor structure reaches the quantum feature map unaltered. |

## Key decisions

1. **Windowing = adopt the mirror's segmentation (D39).** The Z24 mirror already
   segments each recording into fixed **6000-sample analysis windows**. We adopt
   these as the analysis window rather than re-segmenting, which avoids
   label/group bookkeeping and a leakage surface. Finer framing (if needed) is a
   *feature-extraction* concern handled inside Welch's method at M4, and a
   `Taper` is provided for spectral conditioning.

2. **Amplitude standardization is ours to do — not double-normalization (D38).**
   The mirror performed **length** normalization (trim/pad to 60,000 samples) and
   segmentation, **not amplitude** standardization (verified in the mirror README
   and confirmed empirically: raw per-channel std ≈ 2e-4). So
   `PerChannelStandardizer` is legitimate and necessary; it does not re-apply an
   existing normalization.

3. **Healthy-only fit, structurally.** Stateful transforms are fit exclusively
   inside `Pipeline.fit(split.train)`. `tests/test_preprocess.py` asserts that
   transforming eval does not change the fitted scaler stats, and that the
   `SpatialPCA` basis is reproducible from train signals alone.

4. **Spatial PCA never sees labels.** PCA is unsupervised and fit on train
   signals only; the requirement "PCA must never see evaluation labels" holds by
   construction (the fit path has no access to labels at all).

5. **Reproducibility.** No transform uses an RNG. With a fixed split seed the
   entire pipeline is bit-for-bit reproducible (`test_pipeline_is_deterministic`).
   The split `seed` and all transform configs are recorded in the cache
   `manifest.json`.

## Cache layout (`data/processed/z24/`, git-ignored)

| File | Contents |
|------|----------|
| `train_signals.npy` | preprocessed healthy-only train (label-free) |
| `eval_signals.npy` | preprocessed eval |
| `eval_labels.npy` | eval labels — **re-sealed into a `LabelVault` on load** |
| `split_meta.json` | reconstructs `SplitMetadata` (indices/groups) → no-leakage |
| `manifest.json` | timestamp, fs, source-provenance reference, shapes, pipeline config + fitted-state summary, unsupervised-contract note |

Reproduce:
```bash
python - <<'PY'
from qguard.data.loaders.z24 import load_z24
from qguard.preprocess import build_pipeline
from qguard.preprocess.cache import cache_split
split = load_z24(seed=1234)
pipe = build_pipeline({'detrend': {'kind': 'linear'}, 'standardize': True}, seed=1234)
cache_split(pipe.fit_transform_split(split), pipe, 'data/processed/z24')
PY
```

## Open items carried forward

- **OQ-fs:** `fs=100 Hz` is a documented assumption; the bandpass stays off and
  no frequency-axis claim is made until validated.
- **OQ-label-validation:** M4 EDA will check that label 0 (healthy) forms the
  tightest feature cluster, sanity-checking the PDT index assumption.
