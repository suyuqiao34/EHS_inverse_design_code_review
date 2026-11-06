"""Fig. 7c: absolute Tm error versus measured Tm for M0 and the feedback-updated model.

Errors are averaged over the three repeated outer tests of each feedback record;
trends are LOWESS fits (frac = 0.35) with 95% bands from 800 bootstrap resamples
of salt combinations.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent / "validation" / "results" / "grouped_cv_per_sample_predictions.csv"
OUT_DIR = HERE / "output"

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

C_M0 = "#8F959C"
C_UPD = "#3E6F9E"
C_NEG = "#EEF2F6"
C_ZERO = "#4F4F4F"


def lowess_on_grid(x, y, grid, frac=0.35):
    fit = lowess(y, x, frac=frac, it=0, return_sorted=True)
    fit_df = pd.DataFrame({"x": fit[:, 0], "y": fit[:, 1]}).groupby("x", as_index=False).mean()
    return np.interp(grid, fit_df["x"].to_numpy(), fit_df["y"].to_numpy())


def bootstrap_ci(data, err_col, grid, frac=0.35, n_boot=800, seed=42):
    rng = np.random.default_rng(seed)
    combos = data["combo"].unique()
    curves = np.empty((n_boot, len(grid)), dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(combos, size=len(combos), replace=True)
        boot = pd.concat([data[data["combo"] == c] for c in sampled], ignore_index=True)
        curves[i, :] = lowess_on_grid(boot["Tm_true"].to_numpy(), boot[err_col].to_numpy(), grid, frac=frac)
    return np.percentile(curves, [2.5, 97.5], axis=0)


df = pd.read_csv(CSV_PATH)
t = df[df["task"] == "Tm"].copy()
t["err_m0"] = (t["Tm_pred_M0"] - t["Tm_true"]).abs()
t["err_upd"] = (t["Tm_pred_updated"] - t["Tm_true"]).abs()
sample = (
    t.groupby("row_index", as_index=False)
    .agg(combo=("combo", "first"), Tm_true=("Tm_true", "first"), err_m0=("err_m0", "mean"), err_upd=("err_upd", "mean"))
    .sort_values("Tm_true")
)

x = sample["Tm_true"].to_numpy()
grid = np.linspace(x.min() - 1.0, x.max() + 1.0, 260)
y0_line = lowess_on_grid(x, sample["err_m0"].to_numpy(), grid, frac=0.35)
y1_line = lowess_on_grid(x, sample["err_upd"].to_numpy(), grid, frac=0.35)
y0_lo, y0_hi = bootstrap_ci(sample, "err_m0", grid, frac=0.35, n_boot=800, seed=42)
y1_lo, y1_hi = bootstrap_ci(sample, "err_upd", grid, frac=0.35, n_boot=800, seed=4242)

fig, ax = plt.subplots(figsize=(8.0, 6.2), dpi=300)
ax.axvspan(grid.min(), 0, color=C_NEG, zorder=0)
ax.axvline(0, color=C_ZERO, lw=1.5, ls="--", zorder=1)
ax.scatter(sample["Tm_true"], sample["err_m0"], s=34, c=C_M0, alpha=0.55, edgecolors="white", linewidths=0.5, zorder=2)
ax.scatter(sample["Tm_true"], sample["err_upd"], s=34, c=C_UPD, alpha=0.60, edgecolors="white", linewidths=0.5, zorder=2)
ax.fill_between(grid, y0_lo, y0_hi, color=C_M0, alpha=0.14, zorder=3)
ax.fill_between(grid, y1_lo, y1_hi, color=C_UPD, alpha=0.14, zorder=3)
ax.plot(grid, y0_line, color=C_M0, lw=3.0, zorder=4)
ax.plot(grid, y1_line, color=C_UPD, lw=3.0, zorder=4)

ax.set_xlim(-33, 68)
ax.set_ylim(0, 92)
ax.set_xlabel(r"Measured $\mathbf{\mathit{T}_{m}}$ (°C)", fontsize=18)
ax.set_ylabel(r"Absolute error of $\mathbf{\mathit{T}_{m}}$ (°C)", fontsize=18)
for s in ax.spines.values():
    s.set_linewidth(2.0)
ax.tick_params(axis="x", labelsize=16, top=False)
ax.tick_params(axis="y", labelsize=16, right=False)
handles = [Line2D([], [], color=C_M0, lw=3, label="M0"), Line2D([], [], color=C_UPD, lw=3, label="Updated model")]
ax.legend(handles=handles, loc="upper right", frameon=False, prop={"size": 14}, handlelength=2.2, handletextpad=0.5)

OUT_DIR.mkdir(exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT_DIR / f"Fig7c.{ext}", dpi=600)
plt.close(fig)
print(f"{len(sample)} feedback records with Tm; saved to {OUT_DIR}")
