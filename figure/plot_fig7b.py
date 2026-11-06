"""Fig. 7b: paired fold-wise Tm MAE of M0 and the feedback-updated model (grouped nested validation)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent / "validation" / "results" / "grouped_cv_per_fold_metrics.csv"
OUT_DIR = HERE / "output"

df = pd.read_csv(CSV_PATH)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 16,
    "font.weight": "normal",
    "axes.labelweight": "bold",
    "axes.linewidth": 2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

C_OLD = "#8F959C"
C_NEW = "#3E6F9E"
C_LINE = "#C7CCD3"
OFF_PT = 0.38
OFF_MEAN = 0.78


def paired(task, scope):
    d = df[(df["task"] == task) & (df["scope"] == scope)]
    p = d.pivot_table(index=["repeat", "fold"], columns="model", values="MAE")
    return p["M0"].to_numpy(), p["feedback_updated"].to_numpy()


def draw_pair(ax, c, m0, upd):
    x0, x1 = c - OFF_PT, c + OFF_PT
    for a, b in zip(m0, upd):
        ax.plot([x0, x1], [a, b], color=C_LINE, lw=1.1, zorder=1)
    ax.scatter(np.full(len(m0), x0), m0, s=60, c=C_OLD, edgecolors="white", linewidths=0.6, zorder=3)
    ax.scatter(np.full(len(upd), x1), upd, s=60, c=C_NEW, edgecolors="white", linewidths=0.6, zorder=3)
    for x, v, col in ((c - OFF_MEAN, m0, C_OLD), (c + OFF_MEAN, upd, C_NEW)):
        ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="D", ms=9, color="black", mfc=col, mec="black",
                    mew=1.2, capsize=4, elinewidth=1.8, capthick=1.8, zorder=4)


groups = [
    ("all_new", "All"),
    ("Tm<0", r"$\mathit{T}_{\mathrm{m}}$ < 0 °C"),
    ("Tm>=0", r"$\mathit{T}_{\mathrm{m}}$ ≥ 0 °C"),
]
centers = np.arange(len(groups)) * 2.4
fig, ax = plt.subplots(figsize=(8.0, 6.2), dpi=300)
for c, (scope, _) in zip(centers, groups):
    m0, upd = paired("Tm", scope)
    draw_pair(ax, c, m0, upd)
    print(f"{scope:8s} MAE M0 = {m0.mean():.1f} +/- {m0.std(ddof=1):.1f} | "
          f"updated = {upd.mean():.1f} +/- {upd.std(ddof=1):.1f} (n folds = {len(m0)})")
for x in (centers[:-1] + centers[1:]) / 2:
    ax.axvline(x, color="#E9ECEF", lw=1.0, zorder=0)
ax.set_xticks(centers)
ax.set_xticklabels([g[1] for g in groups])
ax.set_xlim(centers[0] - 1.2, centers[-1] + 1.2)
ax.set_ylim(0, 80)
ax.set_xlabel("Test subset", fontsize=18)
ax.set_ylabel(r"MAE of $\mathbf{\mathit{T}_{m}}$ (°C)", fontsize=18)
for s in ax.spines.values():
    s.set_visible(True)
    s.set_linewidth(2.0)
ax.tick_params(axis="y", labelsize=16, right=False)
ax.tick_params(axis="x", labelsize=16, top=False, length=0)
handles = [
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_OLD, mec="white", label="M0"),
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_NEW, mec="white", label="Updated model"),
    Line2D([], [], marker="D", ls="-", ms=8, color="black", mfc="white", label="Mean ± s.d."),
]
ax.legend(handles=handles, loc="upper right", frameon=False, handletextpad=0.3, labelspacing=0.35, prop={"size": 14})

OUT_DIR.mkdir(exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT_DIR / f"Fig7b.{ext}", dpi=600)
plt.close(fig)
print(f"saved to {OUT_DIR}")
