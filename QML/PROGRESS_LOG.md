# Q-GUARD — Progress Log

Living project memory. Newest entries on top. Dates are absolute.

---

## Milestone state

| ID | Milestone | State |
|----|-----------|-------|
| M0 | Repository scaffold & tooling | ✅ done (2026-06-19) |
| M1 | Z24 acquisition & DATASET_REPORT | ✅ done (2026-06-19) |
| M2 | Loaders & canonical schema + label-quarantine + no-leakage | ✅ done (2026-06-19) |
| M3 | Preprocessing pipeline (deterministic, healthy-only fit, cached) | ✅ done (2026-06-19) |
| M4 | Feature extraction (PSD, AR, PCA) + EDA | ✅ done (2026-06-19) |
| M5 | Classical baselines + SPC decision (E01) | ✅ done |
| M6 | 1D-CNN autoencoder baseline | ✅ done |
| M7 | Quantum feature maps + fidelity kernel | ✅ done (ZZ default) |
| M8+M9 | Geometric core: tangent projector, retraction, MoM | ✅ done |
| M10 | Q-GUARD scorers (E02) + robustness (E02b) | ✅ done |
| M11 | Synthetic confound rejection (E04) | ✅ done |
| Final | Tables, figures, README, report.md, architecture fig | ✅ done (2026-06-19) |
| ~~M5(dup)~~ | (superseded by table above) | — |
| M6 | 1D-CNN autoencoder baseline | pending |
| M7 | Quantum feature maps + exact fidelity kernel | pending |
| M8 | Embeddings + UMAP visualization (NEW) | pending |
| M9 | Tangent-space core + MoM (geometry tests) | pending |
| M10 | Q-GUARD scorers (E02) | pending |
| M11 | Environmental-confound rejection on Z24 temp (E04) | pending |
| M12 | Evaluation, runner & figures | pending |
| M13 | Ablations + report | pending |
| M14 | Damage localization (stretch) | pending |

## Experiments

- E01 baselines · E02 Q-GUARD · E03 feature-map ablation · E04 Z24 environment
  · E05 shot-noise ablation · E06 localization (stretch).
- KW51-based experiments removed (single-dataset decision).

## Decision log

- **D40 (2026-06-19):** **M4 features** (`src/qguard/features/`): `PSDFeatures`
  (log10 Welch, sensor-mean, pooled to n_bins — quantum path), `ARFeatures`
  (batched Yule-Walker, sensor-mean — Sohn baseline), `FeaturePCA` (healthy-train
  fit). EDA finding: with naive sensor-averaged PSD, intra-class-0 spread (3.07) >
  all damaged-centroid distances (≤2.73) → label 0 valid as reference (D34 holds)
  but naive PSD is weakly separable; 9 heterogeneous setups inflate within-class
  variance. New **OQ-feat**: revisit sensor-averaging / per-setup handling and
  richer representations at M5/M7 (this is a core motivation for the geometry).
- **D39 (2026-06-19):** **Windowing = adopt mirror segmentation.** The mirror
  already provides 6000-sample analysis windows; we use those rather than
  re-segmenting (avoids label/group bookkeeping + a leakage surface). Optional
  `Taper` (Hann/Hamming) provided for spectral conditioning; finer framing is an
  M4 feature concern (Welch). See `docs/preprocessing.md`.
- **D38 (2026-06-19):** **Amplitude standardization is ours, not
  double-normalization.** Mirror did *length* normalization + segmentation, not
  *amplitude* standardization (confirmed: raw per-channel std ≈ 2e-4). So
  `PerChannelStandardizer` (per-sensor, healthy-train-fit) is legitimate.
- **D37 (2026-06-19):** **M3 preprocessing pipeline** (`src/qguard/preprocess/`):
  composable deterministic transforms (Detrend → optional Bandpass → optional
  Taper → PerChannelStandardizer → optional SpatialPCA), fit ONLY on healthy
  train via `Pipeline.fit`; `fit_transform_split` preserves the leakage-free
  `SplitMetadata`. Processed cache at `data/processed/z24/` with `manifest.json`
  (config + fitted-state summary + provenance ref) and sealed eval labels.
- **D36 (2026-06-19):** Added `SplitMetadata` and `DatasetSplit` schema (per user
  request) + `LabelVault`. Label quarantine is now *structural*: the fit path
  receives a label-free `Recording`; labels only via `LabelVault.reveal(consumer)`.
- **D35 (2026-06-19):** **Group-aware splitting.** The 10 segments of one Z24
  recording (group = window // 10) are kept together; healthy groups split into
  train/eval, all damaged groups eval-only. _Why:_ segment-level splitting would
  leak correlated samples across train/eval — a classic SHM evaluation error.
  New `tests/test_no_leakage.py` enforces window- AND group-level disjointness.
- **D34 (2026-06-19):** **Label semantics resolved.** Mirror labels are scenario
  indices 0–16 (PDT order); healthy/reference = label 0. The exact index→scenario
  mapping is undocumented by the mirror, so `Z24_SCENARIO_NAMES` encodes the
  canonical Z24 PDT order as a *documented assumption*, to be empirically
  validated in EDA (M4: healthy class should form the tightest feature cluster).
  Healthy label set is configurable so results never hard-depend on this.
- **D33 (2026-06-19):** Z24 mirror **confirmed to lack any temperature/environmental
  channel** (`labels.npy` = 17-class integer array, 90 windows/class). **M11/E04
  reframed** as a *controlled synthetic-confound proof-of-concept* (inject
  temperature-like modulation into healthy windows; show tangent projection
  rejects it), clearly labelled synthetic. Real long-term data from KU Leuven is
  an optional future extension.
- **D32 (2026-06-19):** Added `reports/provenance/` for machine-readable provenance
  records (per user request). Z24 provenance at `reports/provenance/z24_provenance.json`.
- **D31 (2026-06-19):** Z24 acquired via HuggingFace mirror
  `thanglexuan/Z24-dataset-processed`; provenance verified and documented; MIT
  mirror-tag vs KU Leuven research-use ownership caveat recorded in DATASET_REPORT.
- **D30 (2026-06-19):** **QML is now its own standalone git repo** (`git init -b
  main` inside QML). _Why:_ the entire home directory `C:/Users/singh` was found
  to be an uninitialized git repo, so any `git add -A` from inside QML would scan
  and risk tracking personal files. Isolating QML fixes this and is the correct
  setup for a portfolio repo. No commit made yet (will commit only when asked).
- **D29 (2026-06-19):** Add a dedicated **UMAP/embedding visualization milestone
  (M8)** *before* tangent-space modeling, to inspect classical-feature and
  quantum-state embeddings before committing to the geometry. _Why:_ de-risks the
  geometry by making manifold structure visible early.
- **D28:** Single dataset = **Z24 only**; **KW51 dropped**. Environmental-confound
  study runs on Z24's own ~1-year temperature record. _Why:_ user scope decision;
  Z24 already carries the temperature variation the novelty needs.
- **D27:** 14 milestones (M0–M14), each ≤1 day; M14 localization is stretch.
- **D26:** (KW51 size note) — obsolete after D28.
- **D25:** Primary dataset = Z24 processed (HuggingFace), after web-verification
  showed the LANL institute download endpoint unreachable on 2026-06-19.
- **D24:** `_paper_text/` is git-ignored; curated paper summaries authored as
  markdown under `docs/paper_summaries/` in a later milestone.
- **D20–D23:** Package `qguard`, src-layout, logic out of notebooks; label
  quarantine enforced by a dedicated test; MIT code license; six named experiments.
- **D13–D19:** PennyLane framework; statevector sim, exact fidelities (shot noise
  = ablation); config-driven (Hydra) + MLflow tracking; `NoveltyDetector` protocol
  unifies quantum & classical scorers; 8-layer architecture.
- **D6–D12:** Project = **Q-GUARD**; mandatory classical baselines (AR+SPC,
  PCA+Fisher+SPC, 1D-CNN AE); no quantum-advantage claims; PSD (quantum) + AR
  (Sohn baseline) features; tangent-Mahalanobis primary scorer; ZZ feature map
  primary.
- **D1–D5:** Three papers read; convergent task = unsupervised novelty detection
  on a curved manifold; constraints = simulator-only, 1–2 months, balanced
  publication+portfolio goal.

## Open questions

- OQ-fs (NEW): Z24 sampling frequency assumed **100 Hz** (benchmark value), used
  for the PSD frequency axis (M4). Mirror does not restate it; flagged in
  `loaders/z24.py` (`DEFAULT_FS_HZ`). Validate before any frequency-axis claim.
- OQ-data: RESOLVED at M1 — provenance verified & documented in DATASET_REPORT.md.
- OQ-label: RESOLVED via D34 (healthy = label 0; canonical PDT mapping documented;
  empirical validation deferred to M4 EDA).
- OQ-track: MLflow only (decided). W&B dropped per user.

## Verified facts (2026-06-19)

- Z24: processed mirror `thanglexuan/Z24-dataset-processed` (HuggingFace) =
  Progressive Damage Test only. inputs (1530,27,6000) f32; labels 17 classes,
  90 each; **no temperature channel** (confirmed M1). Long-term temperature
  campaign would require the KU Leuven original (optional future extension).
- KW51 (dropped): Zenodo DOI 10.5281/zenodo.3745914, CC-BY-NC-SA 4.0, 71.6 TB.
- LANL 3-story official endpoint `institute.lanl.gov/ei` unreachable.
