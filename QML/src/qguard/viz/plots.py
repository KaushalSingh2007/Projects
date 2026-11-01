"""Regenerate all report tables and figures from experiments/results/*.json.

Run: python -m qguard.viz.plots  (or: make figures)
Reproducible & deterministic — reads only logged results, writes to reports/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RESULTS = Path("experiments/results")
FIGS = Path("reports/figures")
PRETTY = {
    "ar_spc": "AR+SPC", "pca_spc": "PCA+SPC", "cnn_ae": "CNN-AE",
    "zz_kernel": "ZZ kernel", "tangent_mahalanobis": "Tangent-Maha",
    "subspace_fidelity": "Subspace-Fid", "kernel_svdd": "Kernel-SVDD",
}


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def fig_main(e01, e02):
    methods = {**e01, **e02}
    order = ["ar_spc", "pca_spc", "cnn_ae", "zz_kernel", "subspace_fidelity",
             "kernel_svdd", "tangent_mahalanobis"]
    order = [m for m in order if m in methods]
    auc = [methods[m]["roc_auc"] for m in order]
    names = [PRETTY[m] for m in order]
    colors = ["#7a7a7a"] * 3 + ["#4c72b0"] * (len(order) - 4) + ["#c44e52"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(names, auc, color=colors[: len(order)])
    ax.axhline(0.5, ls="--", c="k", lw=1, label="chance")
    ax.set_ylabel("ROC-AUC (damage detection)")
    ax.set_title("Q-GUARD vs baselines — Z24, leak-free split")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_main_comparison.png", dpi=150)
    plt.close(fig)


def fig_robustness(rob):
    if not rob:
        return
    conds = list(rob.keys())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for method, c in [("Tangent", "#c44e52"), ("ZZKernel", "#4c72b0"), ("AR+SPC", "#7a7a7a")]:
        ys = [rob[k][method]["roc_auc"] for k in conds]
        ax.plot(conds, ys, "o-", label=method, color=c)
    ax.axhline(0.5, ls="--", c="k", lw=1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Robustness: seeds / PCA dim / noise")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_robustness.png", dpi=150)
    plt.close(fig)


def fig_confound(e04):
    if not e04:
        return
    m = e04["metrics"]
    names = ["ZZ kernel", "Tangent", "Confound-proj\nTangent"]
    auc = [m["ZZ_kernel"]["roc_auc"], m["Tangent"]["roc_auc"],
           m["Confound_proj_Tangent"]["roc_auc"]]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(names, auc, color=["#4c72b0", "#c44e52", "#dd8452"])
    ax[0].axhline(0.5, ls="--", c="k", lw=1)
    ax[0].set_ylabel("ROC-AUC"); ax[0].set_ylim(0, 1)
    ax[0].set_title("Damage AUC under synthetic confound")
    ax[1].bar(["Tangent", "Confound-proj"], [e04["csens_tangent"], e04["csens_proj"]],
              color=["#c44e52", "#dd8452"])
    ax[1].axhline(0.5, ls="--", c="k", lw=1, label="insensitive")
    ax[1].set_ylabel("nuisance sensitivity (AUC)")
    ax[1].set_title("Confound sensitivity (lower=better)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_confound.png", dpi=150)
    plt.close(fig)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    boxes = [
        (0.5, 2.2, "Z24\nraw windows"), (2.4, 2.2, "Preprocess\n(detrend, scale)"),
        (4.3, 2.2, "Features\nPSD / AR"), (6.4, 3.1, "ZZ feature map\n|psi(x)>"),
        (8.5, 3.1, "Tangent core\n+ MoM"), (6.4, 1.3, "Classical\nbaselines"),
        (10.4, 2.2, "Eval\nROC/PR"),
    ]
    coords = {}
    for x, y, t in boxes:
        b = FancyBboxPatch((x - 0.8, y - 0.45), 1.6, 0.9, boxstyle="round,pad=0.05",
                           fc="#eef2f7", ec="#4c72b0")
        ax.add_patch(b)
        ax.text(x, y, t, ha="center", va="center", fontsize=9)
        coords[t] = (x, y)
    arrows = [(0.5, 2.2, 2.4, 2.2), (2.4, 2.2, 4.3, 2.2), (4.3, 2.2, 6.4, 3.1),
              (4.3, 2.2, 6.4, 1.3), (6.4, 3.1, 8.5, 3.1), (8.5, 3.1, 10.4, 2.2),
              (6.4, 1.3, 10.4, 2.2)]
    for x0, y0, x1, y1 in arrows:
        ax.add_patch(FancyArrowPatch((x0 + 0.8, y0), (x1 - 0.8, y1),
                     arrowstyle="->", mutation_scale=14, color="#555"))
    ax.set_xlim(-0.5, 11.7); ax.set_ylim(0.5, 4)
    ax.set_title("Q-GUARD architecture (label-free; eval-only sees labels)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_architecture.png", dpi=150)
    plt.close(fig)


def write_table(e01, e02, e04):
    rows = {**e01, **e02}
    lines = ["# Final results (Z24, leak-free split)\n",
             "## Main comparison\n", "| Method | ROC-AUC | PR-AUC |", "|---|---|---|"]
    order = ["ar_spc", "pca_spc", "cnn_ae", "zz_kernel", "subspace_fidelity",
             "kernel_svdd", "tangent_mahalanobis"]
    for m in [m for m in order if m in rows]:
        lines.append(f"| {PRETTY[m]} | {rows[m]['roc_auc']:.3f} | {rows[m]['pr_auc']:.3f} |")
    if e04:
        m = e04["metrics"]
        lines += ["\n## Synthetic confound\n", "| Method | ROC-AUC | PR-AUC |", "|---|---|---|",
                  f"| ZZ kernel | {m['ZZ_kernel']['roc_auc']:.3f} | {m['ZZ_kernel']['pr_auc']:.3f} |",
                  f"| Tangent | {m['Tangent']['roc_auc']:.3f} | {m['Tangent']['pr_auc']:.3f} |",
                  f"| Confound-proj Tangent | {m['Confound_proj_Tangent']['roc_auc']:.3f} | {m['Confound_proj_Tangent']['pr_auc']:.3f} |",
                  f"\nNuisance sensitivity (0.5=insensitive): Tangent {e04['csens_tangent']:.3f} -> "
                  f"projected {e04['csens_proj']:.3f} (Δ {e04['csens_tangent']-e04['csens_proj']:.3f})."]
    Path("reports/results_table.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    e01, e02 = _load("E01_baselines.json"), _load("E02_qguard.json")
    rob, e04 = _load("E02b_robustness.json"), _load("E04_confound.json")
    fig_main(e01, e02)
    fig_robustness(rob)
    fig_confound(e04)
    fig_architecture()
    write_table(e01, e02, e04)
    print("figures + tables written to reports/")


if __name__ == "__main__":
    main()
