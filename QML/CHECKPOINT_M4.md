# CHECKPOINT — End of M4 (2026-06-19)

Delta checkpoint (cost mode). For unchanged **project identity, architecture,
dataset summary, decisions D1–D39, risks** see `CHECKPOINT_M3.md`. Full decision
history in `PROGRESS_LOG.md`.

## Milestones
M0–M3 ✅; **M4 ✅ (feature extraction PSD/AR/PCA + EDA)**. Next: M5.
Git: `f5ca8ad` = M0+M1. **M2–M4 uncommitted.** Tests: **24/24 passing**.

## New since M3
- `src/qguard/features/{__init__,psd,ar,pca}.py`
  - `PSDFeatures(nperseg=256, n_bins=32)` log10 Welch, sensor-mean → (n,32) — quantum path.
  - `ARFeatures(order=8)` batched Yule-Walker, sensor-mean → (n,order) — Sohn baseline.
  - `FeaturePCA(n_components)` sklearn PCA, **fit on healthy-train only**.
- `tests/test_features.py` (4 tests: shape, determinism, PSD peak @known freq, AR1 recovery, healthy-only PCA fit).

## EDA finding (D40) — affects M5/M7
Naive sensor-averaged PSD: **intra-class-0 spread 3.07 > all damaged-centroid
distances (≤2.73)**. Label 0 remains valid healthy reference (D34 holds), but the
naive feature is weakly separable; 9 heterogeneous setups inflate within-class
variance. **OQ-feat:** revisit sensor-averaging / per-setup handling and richer
representations downstream — a core motivation for the tangent-space geometry.

## Open questions
OQ-fs (100 Hz assumed), OQ-label-validation (label 0 valid; separability weak →
OQ-feat), **OQ-feat (new)**, OQ-commit (M2–M4 uncommitted).

## Next step — M5
Classical baselines on the `NoveltyDetector` protocol: AR+SPC (Sohn) and
PCA+Fisher+SPC, + SPC decision layer; first ROC/PR numbers (experiment E01).
Regenerate progress graph at M5 (per cost-mode rule 5).
