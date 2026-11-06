"""Fig. 7a: Tm-dHm distribution of single-peak literature and feedback records (Round-1 dataset)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent / "data" / "round1" / "round1_all_records.csv"
OUT_DIR = HERE / "output"

TM = "Tm (°C)"
DH = "ΔHm (J/g)"

df = pd.read_csv(CSV_PATH)
for c in ["single peak", TM, DH]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
is_new = df["source"].eq("feedback")
has = df[TM].notna() & df[DH].notna()

old_sp = df[~is_new & has & df["single peak"].eq(1)]
new_sp = df[is_new & has & df["single peak"].eq(1)]

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

C_OLD = "#9A9A9A"
C_NEW = "#D9541E"

fig = plt.figure(figsize=(8.0, 6.2), dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 1.0], height_ratios=[1.0, 4.2], wspace=0.0, hspace=0.0)
ax = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

ax.scatter(old_sp[TM], old_sp[DH], s=60, c=C_OLD, edgecolors="white", linewidths=0.6, alpha=0.85, zorder=3)
ax.scatter(new_sp[TM], new_sp[DH], s=70, c=C_NEW, edgecolors="white", linewidths=0.6, alpha=0.95, zorder=4)

ax.axvline(0, color="#333333", lw=1.4, ls="--", zorder=1)
ax.set_xlabel(r"$\mathbf{\mathit{T}_{m}}$ (°C)", fontsize=18)
ax.set_ylabel(r"$\mathbf{\Delta\mathit{H}_{m}}$ (J·g⁻¹)", fontsize=18)

all_tm = pd.concat([old_sp[TM], new_sp[TM]])
xlim = (np.floor(all_tm.min() / 10) * 10 - 5, np.ceil(all_tm.max() / 10) * 10 + 5)
ylim = (-10, 320)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)


def kde_line(axis, data, color, orient, grid):
    d = gaussian_kde(data.values)(grid)
    if orient == "x":
        axis.fill_between(grid, d, color=color, alpha=0.30, lw=0)
        axis.plot(grid, d, color=color, lw=2.2)
    else:
        axis.fill_betweenx(grid, d, color=color, alpha=0.30, lw=0)
        axis.plot(d, grid, color=color, lw=2.2)


gx = np.linspace(*xlim, 400)
gy = np.linspace(*ylim, 400)
kde_line(ax_top, old_sp[TM], C_OLD, "x", gx)
kde_line(ax_top, new_sp[TM], C_NEW, "x", gx)
kde_line(ax_right, old_sp[DH], C_OLD, "y", gy)
kde_line(ax_right, new_sp[DH], C_NEW, "y", gy)
ax_top.axvline(0, color="#333333", lw=1.4, ls="--")

for a in (ax_top, ax_right):
    a.tick_params(axis="both", which="both", bottom=False, labelbottom=False, top=False, labeltop=False,
                  left=False, labelleft=False, right=False, labelright=False)
    for s in a.spines.values():
        s.set_visible(False)
ax_top.set_ylim(bottom=0)
ax_right.set_xlim(left=0)
for s in ax.spines.values():
    s.set_visible(True)
    s.set_linewidth(2.0)
ax.tick_params(labelsize=16, top=True, right=True, width=2.0)

handles = [
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_OLD, mec="white", label=f"Previous, single peak (n = {len(old_sp)})"),
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_NEW, mec="white", label=f"New, single peak (n = {len(new_sp)})"),
]
ax.legend(handles=handles, loc="upper right", frameon=False, handletextpad=0.2, borderaxespad=0.3,
          labelspacing=0.3, prop={"size": 13})

OUT_DIR.mkdir(exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT_DIR / f"Fig7a.{ext}", dpi=600)
plt.close(fig)
print(f"literature single-peak n = {len(old_sp)}, feedback single-peak n = {len(new_sp)}; saved to {OUT_DIR}")
