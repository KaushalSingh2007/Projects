# CHECKPOINT — End of M3 (2026-06-19)

**Purpose:** self-contained resume anchor. A future session should be able to read
*only this file* (+ `PROGRESS_LOG.md` for full decision history, `DATASET_REPORT.md`
for provenance, `docs/preprocessing.md` for M3 detail) and continue without
context loss. Supersedes `CHECKPOINT_M2.md`.

---

## 0. Project identity

**Q-GUARD** — Quantum Geometry-aware Unsupervised Anomaly detection for
vibration-based Damage detection. Curvature-aware novelty scoring on the
pure-state manifold `CP^{d-1}` for **label-free, environment-robust** SHM,
benchmarked **honestly** vs classical baselines. **No quantum-advantage claim**;
statevector-simulator only.

**Convergent thesis:** three papers describe one task — estimate a healthy
reference on a curved manifold from streaming data, flag deviations, using only
baseline data. (1) Avci 2021 (SHM ML/DL gaps: no field labels, env-confounded
features); (2) Sohn 2000 (SHM = unsupervised novelty detection, AR+SPC); (3)
Lumbreras & Tomamichel 2026 (tangent-space + symmetric retraction + var-adaptive
MoM on the Fubini–Study manifold; `1−Tr(ρA)=½‖ρ−A‖²_F` is a manifold novelty score).

**Constraints (fixed):** simulator-only (≤~12 qubits), 1–2 months, balanced
publication+portfolio. PennyLane. MLflow only.

## 1. Completed milestones

| ID | Milestone | State |
|----|-----------|-------|
| P1–P6 | research → 20 ideas → Q-GUARD → design → architecture → repo → milestones | ✅ |
| **M0** | Repo scaffold & tooling | ✅ |
| **M1** | Z24 acquisition + provenance | ✅ |
| **M2** | Loaders, schema, group-aware leakage-free split, integrity tests | ✅ |
| **M3** | Deterministic preprocessing pipeline + processed cache + validation | ✅ |

Git: commit `f5ca8ad` = M0+M1. **M2 + M3 implemented but UNCOMMITTED.**
Tests: **20/20 passing** (`python -m pytest`): smoke(3), label_quarantine(4),
no_leakage(5), preprocess(8).

## 2. Roadmap (remaining)

M4 features + EDA → M5 classical baselines (AR+SPC, PCA+Fisher+SPC) → M6 1D-CNN
autoencoder → M7 quantum feature maps + exact fidelity kernel → **M8 embeddings +
UMAP (before any quantum-geometry modeling)** → M9 tangent-space core + MoM
(geometry tests) → M10 Q-GUARD scorers (tangent-Mahalanobis primary,
subspace-fidelity, quantum-kernel SVDD) → M11 synthetic env-confound rejection →
M12 eval + Hydra runner + figures → M13 ablations (feature-map, shot-noise) +
report → M14 localization (stretch). Experiments E01–E06.

## 3. Current architecture

```mermaid
flowchart LR
  D[(Z24 raw npy)] --> LOAD[load_z24 -> DatasetSplit]
  LOAD --> TR[train Recording: healthy-only, label-free]
  LOAD --> EV[eval Recording]
  LOAD --> VA[LabelVault: sealed eval labels]
  TR --> PFIT[Pipeline.fit  — healthy-only]
  PFIT --> PP[detrend -> (bandpass) -> (taper) -> per-channel scaler -> (spatial PCA)]
  PP --> TRP[train preprocessed]
  PP --> EVP[eval preprocessed]
  TRP --> CACHE[(data/processed/z24 + manifest.json)]
  EVP --> CACHE
  VA --> CACHE
  CACHE --> FEAT[features M4: PSD / AR]
  FEAT --> QFM[quantum feature map M7] --> MAN[state on CP^d-1]
  FEAT --> UMAP[UMAP viz M8]
  MAN --> BASE[healthy baseline: base state + tangent cov M9]
  ENV[synthetic temp confound M11] -. project out .-> SCORE
  BASE --> SCORE[tangent-Mahalanobis + var-adaptive MoM M10]
  SCORE --> SPC[SPC decision] --> OUT[damage flag/index]
  FEAT --> BL[baselines M5/M6]
  OUT --> EVAL[honest benchmark M12 — ONLY module that reads VA]
  BL --> EVAL
```

Layers: data → preprocess → features → {quantum | baselines} → scorers
(`NoveltyDetector` protocol, M5) → decision → eval → viz. Config-driven (Hydra),
tracked (MLflow). **Only `qguard.eval` may read labels.**

## 4. Current file tree

```
QML/  (standalone git repo, branch main; data/ & mlruns/ gitignored)
├── README · DATASET_REPORT · PROGRESS_LOG · CHECKPOINT_M2 · CHECKPOINT_M3
├── pyproject.toml · environment.yml · Makefile · LICENSE · .gitignore
├── configs/config.yaml
├── src/qguard/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py · download_z24.py
│   │   ├── schema.py            # Recording, LabelVault, SplitMetadata, DatasetSplit
│   │   └── loaders/{__init__.py, z24.py}   # load_z24, make_split (group-aware)
│   └── preprocess/                          # ★ M3
│       ├── __init__.py
│       ├── pipeline.py          # Detrend, BandpassFilter, Taper, PerChannelStandardizer, SpatialPCA, Pipeline, build_pipeline
│       └── cache.py             # cache_split / load_cached (+ manifest, sealed eval labels)
├── tests/
│   ├── test_pipeline_smoke.py · test_label_quarantine.py · test_no_leakage.py
│   └── test_preprocess.py       # ★ M3 (8 tests)
├── docs/preprocessing.md        # ★ M3 decisions
├── reports/provenance/z24_provenance.json   [committed]
├── data/raw/z24/{inputs,labels}.npy         (gitignored)
└── data/processed/z24/{train_signals,eval_signals,eval_labels}.npy + split_meta.json + manifest.json  (gitignored) ★
```

## 5. Dataset summary (verified 2026-06-19)

- Z24 PDT via HF mirror `thanglexuan/Z24-dataset-processed`. `inputs.npy`
  (1530,27,6000) f32; `labels.npy` (1530,) int64, **17 classes × 90 windows**.
- 1530 = 17 scenarios × 9 setups × 10 segments. **No temperature channel.**
- License: mirror tags MIT; underlying © KU Leuven research-use (treat
  non-commercial; cite Maeck&DeRoeck 2003, Worden&Manson 2007, Reynders 2008).
- Mirror did length-normalization + segmentation, NOT amplitude standardization.
- After group-aware split (seed 1234, healthy=label 0): **train 60 windows
  (healthy-only)**, **eval 1470** (held-out healthy + all damaged), zero leakage.

## 6. Key design decisions (load-bearing; full D1–D39 in PROGRESS_LOG.md)

- Q-GUARD; tangent-Mahalanobis primary scorer; ZZ feature map primary (+angle,
  data-reuploading ablations); baselines AR+SPC, PCA+Fisher+SPC, 1D-CNN AE.
- Single dataset = Z24; features PSD (quantum) + AR (Sohn baseline).
- **Label quarantine structural** (Recording label-free; `LabelVault.reveal`;
  eval-only). **Group-aware split (D35)** — segments never straddle train/eval.
- **Label semantics (D34):** healthy = label 0; canonical PDT map is a documented
  assumption, validate in M4 EDA; healthy set configurable.
- **M11 reframed (D33):** no real temperature → synthetic confound PoC, clearly labelled.
- **M3 (D37–D39):** deterministic pipeline fit healthy-only; windowing = adopt
  mirror's 6000-sample windows (+optional taper); amplitude scaler is ours (not
  double-normalization); processed cache with manifest + sealed labels.
- Repo isolated (D30) — NEVER `git add -A` from `C:/Users/singh`.

## 7. Open questions

- **OQ-fs:** Z24 fs assumed 100 Hz (`loaders/z24.py:DEFAULT_FS_HZ`); bandpass off,
  no frequency-axis claim until validated vs KU Leuven docs.
- **OQ-label-validation:** confirm in M4 EDA that label 0 = tightest feature cluster.
- **OQ-commit:** M2+M3 uncommitted; decide standalone vs bundled commit.

## 8. Known risks & mitigations

| # | Risk | Mitigation |
|---|------|-----------|
| K1 | No measurable quantum benefit | Honest comparative study; geometry contribution stands. Pre-accepted. |
| K2 | No real temperature | M11 = synthetic confound PoC (D33), clearly labelled. |
| K3 | fs assumption wrong | OQ-fs; bandpass off; features valid up to axis scaling. |
| K4 | Label-0 assumption wrong | Configurable; validate M4 EDA. |
| K5 | Segment leakage | Group-aware split (D35) + tests; preserved through M3 cache (tested). |
| K6 | Double-normalization | Resolved (D38): mirror did length-norm only; amplitude scaler is ours. |
| K7 | Home-dir git hazard | QML isolated repo (D30). |
| K8 | Barren plateaus | No deep trainable circuits; kernel/geometry scoring. |
| K9 | inputs.npy ~945 MB | mmap reads; tests use synthetic arrays; processed cache. |

## 9. Next actions (resume here) — M4

**M4 features + EDA.** Implement `src/qguard/features/{psd,ar,pca}.py` (PSD via
Welch for the quantum path; AR/Yule-Walker for the Sohn baseline), each consuming
preprocessed `Recording` signals → feature matrix `X (n_windows, n_features)`.
Then EDA notebook/figures to (a) validate OQ-label (label 0 = tightest cluster),
(b) inspect feature separability healthy vs damaged. Do NOT proceed to M5 until
feature extraction has determinism + shape tests.

## 10. Environment / how to run

- Windows 11; PowerShell + Git Bash. Active `python` = Python 3.14 (AppData);
  conda env `qguard` defined in `environment.yml`.
- Install: `pip install -e ".[all]"`. Tests: `python -m pytest`.
- Data: `python -m qguard.data.download_z24` (raw) → preprocess+cache snippet in
  `docs/preprocessing.md`. Data dirs gitignored; re-create via scripts if absent.
- Always `mmap_mode="r"` for `inputs.npy` (~945 MB); tests use synthetic arrays.
