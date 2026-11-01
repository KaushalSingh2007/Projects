# Final results (Z24, leak-free split)

## Main comparison

| Method | ROC-AUC | PR-AUC |
|---|---|---|
| AR+SPC | 0.549 | 0.986 |
| PCA+SPC | 0.003 | 0.920 |
| CNN-AE | 0.007 | 0.921 |
| ZZ kernel | 0.263 | 0.948 |
| Subspace-Fid | 0.292 | 0.953 |
| Kernel-SVDD | 0.265 | 0.948 |
| Tangent-Maha | 0.629 | 0.988 |

## Synthetic confound

| Method | ROC-AUC | PR-AUC |
|---|---|---|
| ZZ kernel | 0.133 | 0.939 |
| Tangent | 0.608 | 0.985 |
| Confound-proj Tangent | 0.582 | 0.984 |

Nuisance sensitivity (0.5=insensitive): Tangent 0.444 -> projected 0.427 (Δ 0.018).