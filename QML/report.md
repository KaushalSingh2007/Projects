# Q-GUARD: Geometry-aware Quantum Novelty Detection for Vibration-based Structural Health Monitoring

*A reproducible, honest study on the Z24 bridge benchmark.*

---

## Abstract

We study whether geometric tools from quantum state learning improve unsupervised,
vibration-based structural damage detection. Vibration windows are encoded as pure
quantum states on the manifold `CP^{d-1}`; a healthy baseline is estimated and new
data scored by a curvature-aware, variance-adaptive **tangent-space Mahalanobis**
distance, ported from Lumbreras & Tomamichel (2026). On the Z24 progressive-damage
benchmark, under a deliberately hard **leak-free, group-aware split** (evaluation
contains *unseen healthy measurement setups*), all detectors are weak in absolute
terms, but the tangent-space scorer is the only method consistently above chance
(ROC-AUC 0.629 vs 0.263 for the plain quantum kernel; PR-AUC 0.988). A robustness
sweep shows this advantage is **real but fragile** (seed-/dimension-sensitive). A
synthetic environmental-confound experiment shows the tangent representation is
**intrinsically nuisance-robust**, with explicit confound projection adding little.
We make **no quantum-advantage claim**; the contribution is a rigorous,
reproducible methodology and an honest assessment.

## 1. Introduction

Structural Health Monitoring (SHM) rarely has labelled damage data, and standard
modal features are confounded by environment (Avci et al., 2021). Sohn et al.
(2000) framed SHM as **unsupervised novelty detection** from a healthy baseline.
Independently, Lumbreras & Tomamichel (2026) developed estimation on the curved
pure-state manifold (`1 − Tr(ρA) = ½‖ρ − A‖²_F` is a manifold novelty score). We
unify these: a geometry-aware quantum novelty detector for SHM.

## 2. Data

Z24 bridge progressive damage test (KU Leuven), via a processed HuggingFace mirror:
1530 windows × 27 accelerometers × 6000 samples; 17 scenarios × 9 setups × 10
segments; 17 balanced classes; **no temperature channel** (provenance and the
MIT-tag-vs-KU-Leuven-ownership caveat in `DATASET_REPORT.md`). Label 0 = reference;
canonical PDT scenario mapping is a documented assumption.

## 3. Method

- **Preprocess** (deterministic, healthy-fit): detrend → per-channel standardize;
  mirror windows are the analysis unit (`docs/preprocessing.md`).
- **Features**: log-Welch PSD (quantum path), AR/Yule-Walker (Sohn baseline).
- **Quantum encoding**: PSD → PCA(8) → `[0,π]` → **ZZ (IQP) feature map** (default;
  angle encoding as ablation) → statevector on 8 qubits.
- **Q-GUARD core** (ported): tangent projector `P_C(X)=CX(I−C)+(I−C)XC`, symmetric
  retraction (curvature-cancelling), variance-adaptive Median-of-Means.
- **Scorers**: Tangent-Mahalanobis (primary), Subspace-Fidelity, Kernel-SVDD.
- **Baselines**: AR+SPC, PCA+SPC (Hotelling-T²; replaces Sohn's *supervised* Fisher
  to keep the label-free contract), 1D-CNN autoencoder.
- **Evaluation**: ROC-AUC / PR-AUC; labels visible only to `qguard.eval`.

**Leakage control**: group-aware split keeps a recording's 10 segments on one side
of the train/eval boundary (enforced by tests).

## 4. Experiments & Results

Headline (leak-free split): Tangent-Mahalanobis 0.629 AUC / 0.988 PR — best
overall and the only method clearly above chance; plain ZZ kernel 0.263; best
classical AR+SPC 0.549; PCA+SPC and CNN-AE invert under unseen-setup negatives.
(`reports/results_table.md`, `reports/figures/fig_main_comparison.png`.)

**Robustness** (seeds × PCA dim {4,8,12} × mild noise): Tangent beats the kernel in
4/6 conditions but regresses to ~0.50 on some seeds and collapses at dim 4; AR+SPC
is the most stable (~0.55). The geometric gain is **real but fragile**
(`fig_robustness.png`).

**Synthetic confound** (injected temperature-like nuisance; the mirror lacks real
temperature): plain kernel is badly corrupted (AUC 0.133); Tangent is intrinsically
nuisance-insensitive (sensitivity 0.444 ≈ chance) and explicit confound projection
only marginally changes it (Δ 0.018) at a small damage-AUC cost (`fig_confound.png`).

## 5. Key findings

1. Tangent-space geometry is the active ingredient: it beats the plain quantum
   kernel substantially and is the only above-chance method on the hard split.
2. The honest, leakage-free evaluation is unforgiving — a valuable corrective to
   optimistic SHM-ML results that leak correlated segments.
3. The geometric advantage is fragile under resampling; absolute performance is
   modest.
4. Tangent scoring is inherently confound-robust.

## 6. Limitations

- Single dataset; processed third-party mirror; **no real environmental/temperature
  data** (confound is synthetic, clearly labelled).
- Small healthy-eval negative set (≈30) makes AUC high-variance.
- Simulator-only, ≤12 qubits, fixed feature map; no hyperparameter optimization
  (by design — validation, not performance chasing).
- `fs = 100 Hz` is an assumption; no frequency-axis claim is made.

## 7. Future work

- Real Z24 long-term environmental data (KU Leuven) for a genuine confound study.
- Larger negative sets / multiple structures; stability via the paper's full
  epoch/hot-start online estimator.
- Tighter dimensional scaling; damage localization (per-sensor tangent scores);
  hardware-noise study.

## 8. Reproducibility

`pytest` (integrity + geometry + scorer tests), `python -m qguard.data.download_z24`
(+ SHA-256 provenance), `python -m qguard.viz.plots` (all tables/figures). Configs,
seeds, and `experiments/results/*.json` logged; decisions in `PROGRESS_LOG.md`.

## References

Avci, Abdeljaber, Kiranyaz, Hussein, Gabbouj, Inman (2021), *MSSP* 147:107077.
Sohn, Czarnecki, Farrar (2000), *J. Struct. Eng.* 126(11):1356–1363.
Lumbreras, Tomamichel (2026), arXiv:2605.09019. Z24: Maeck & De Roeck (2003);
Reynders et al. (2008); Worden & Manson (2007).
