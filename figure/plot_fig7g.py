"""Fig. 7g: filtered (Tm, dHm) landscapes of M0 and M1 with the validated T3 formulation.

Input: for each model, the compatibility-filtered full-space prediction files
    binary_full_predictions_filtered.csv
    ternary_full_predictions_filtered.csv
(columns tm_pred and dh_pred; about 1.05e8 rows for M0 and 3.5e7 rows for M1).
These files are not included in the repository because of their size and are
available from the corresponding authors on request. The summary statistics of
the plotted layers are provided in data/fig7g_landscape_summary.csv.

The CSV files are streamed in chunks and binned on a 0.05 degC x 0.25 J/g grid; the
binned grids are cached so that the figure can be re-styled without re-reading.

Usage:
    python plot_fig7g.py --m0-dir <folder with M0 filtered files> --m1-dir <folder with M1 filtered files>
"""
import argparse
import hashlib
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FILTERED_FILES = {
    "binary_filtered": "binary_full_predictions_filtered.csv",
    "ternary_filtered": "ternary_full_predictions_filtered.csv",
}
MODEL_STYLE = {
    "M0": {"color": "#8F959C", "alpha": 0.75, "diam_pt": 2.7, "label": "M0 filtered"},
    "M1": {"color": "#3E6F9E", "alpha": 0.70, "diam_pt": 2.7, "label": "M1 filtered"},
}
GRID_X0, GRID_X1, GRID_DX = -80.0, 140.0, 0.05
GRID_Y0, GRID_Y1, GRID_DY = -50.0, 400.0, 0.25
NX = int(round((GRID_X1 - GRID_X0) / GRID_DX))
NY = int(round((GRID_Y1 - GRID_Y0) / GRID_DY))
C_T3 = "#FFCC00"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m0-dir", required=True, help="folder containing the M0 *_filtered.csv files")
    p.add_argument("--m1-dir", required=True, help="folder containing the M1 *_filtered.csv files")
    p.add_argument("--x-col", default="tm_pred")
    p.add_argument("--y-col", default="dh_pred")
    p.add_argument("--chunk-size", type=int, default=2_000_000)
    p.add_argument("--cache-dir", default=os.path.join(SCRIPT_DIR, "_cache_fig7g"))
    p.add_argument("--rebuild-cache", action="store_true", help="ignore cache and rebuild")
    p.add_argument("--only-compute", action="store_true", help="build caches and summary only")
    p.add_argument("--xlim", default="-35,85")
    p.add_argument("--ylim", default="-10,320")
    p.add_argument("--figsize", default="20.8,6.2")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--output-stem", default=os.path.join(SCRIPT_DIR, "output", "Fig7g"))
    p.add_argument("--show-zero-line", type=int, default=1)
    p.add_argument("--show-t3", type=int, default=1)
    p.add_argument("--annotate-t3", type=int, default=1)
    p.add_argument("--t3-label", default="T3")
    p.add_argument("--t3-tm", type=float, default=-10.2)
    p.add_argument("--t3-dh", type=float, default=126.0)
    return p


def parse_pair(text):
    parts = [float(v) for v in str(text).replace("x", ",").split(",") if v.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two numbers, got: {text}")
    return parts[0], parts[1]


def locate_file(run_dir, filename):
    path = os.path.join(run_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find {filename} in {run_dir}")
    return path


def cache_path_for(cache_dir, tag, csv_path, x_col, y_col):
    st = os.stat(csv_path)
    key = "|".join([os.path.abspath(csv_path), str(st.st_size), str(int(st.st_mtime)), x_col, y_col,
                    f"{GRID_X0},{GRID_X1},{GRID_DX},{GRID_Y0},{GRID_Y1},{GRID_DY}"])
    return os.path.join(cache_dir, f"{tag}_{hashlib.md5(key.encode('utf-8')).hexdigest()[:12]}.npz")


def stream_histogram(csv_path, x_col, y_col, chunk_size, tag):
    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig")
    missing = [c for c in (x_col, y_col) if c not in header.columns]
    if missing:
        raise ValueError(f"{tag}: missing columns {missing} in {csv_path}")

    counts = np.zeros(NX * NY, dtype=np.int64)
    n_valid = n_out = n_read = 0
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    t0 = time.time()
    reader = pd.read_csv(csv_path, usecols=[x_col, y_col], chunksize=chunk_size, encoding="utf-8-sig", low_memory=True)
    for i, chunk in enumerate(reader, start=1):
        n_read += len(chunk)
        x = pd.to_numeric(chunk[x_col], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(chunk[y_col], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size == 0:
            continue
        n_valid += x.size
        xmin, xmax = min(xmin, x.min()), max(xmax, x.max())
        ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
        ix = np.floor((x - GRID_X0) / GRID_DX).astype(np.int64)
        iy = np.floor((y - GRID_Y0) / GRID_DY).astype(np.int64)
        inside = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
        n_out += int((~inside).sum())
        counts += np.bincount(iy[inside] * NX + ix[inside], minlength=NX * NY)
        if i % 10 == 0:
            print(f"    {tag}: {n_read:,} rows read, {time.time() - t0:.0f} s", flush=True)

    if n_valid == 0:
        raise RuntimeError(f"{tag}: no valid numeric rows in {csv_path}")
    if n_out:
        print(f"[warn] {tag}: {n_out:,} points outside the grid were not drawn")
    print(f"[done] {tag}: {n_valid:,} valid rows in {time.time() - t0:.0f} s")
    return {"counts": counts.reshape(NY, NX).astype(np.uint32), "n_valid": n_valid, "n_out": n_out,
            "xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)}


def load_or_build(tag, csv_path, args):
    os.makedirs(args.cache_dir, exist_ok=True)
    cpath = cache_path_for(args.cache_dir, tag, csv_path, args.x_col, args.y_col)
    if os.path.isfile(cpath) and not args.rebuild_cache:
        z = np.load(cpath, allow_pickle=False)
        print(f"[cache] {tag}: {os.path.basename(cpath)}")
        return {"counts": z["counts"], "n_valid": int(z["n_valid"]), "n_out": int(z["n_out"]),
                "xmin": float(z["xmin"]), "xmax": float(z["xmax"]), "ymin": float(z["ymin"]), "ymax": float(z["ymax"])}
    print(f"[read] {tag}: {csv_path}")
    res = stream_histogram(csv_path, args.x_col, args.y_col, args.chunk_size, tag)
    np.savez_compressed(cpath, **res)
    return res


def count_below(counts, tm):
    k = max(0, min(NX, int(round((tm - GRID_X0) / GRID_DX))))
    return int(counts[:, :k].sum(dtype=np.int64))


def dilate(mask, rx, ry):
    if rx <= 0 and ry <= 0:
        return mask
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if (dx == 0 and dy == 0) or (rx and ry and (dx / max(rx, 1)) ** 2 + (dy / max(ry, 1)) ** 2 > 1.0):
                continue
            ys_src = slice(max(0, -dy), h - max(0, dy))
            ys_dst = slice(max(0, dy), h - max(0, -dy))
            xs_src = slice(max(0, -dx), w - max(0, dx))
            xs_dst = slice(max(0, dx), w - max(0, -dx))
            out[ys_dst, xs_dst] |= mask[ys_src, xs_src]
    return out


def view_slices(xlim, ylim):
    ix0 = max(0, int(np.floor((xlim[0] - GRID_X0) / GRID_DX)))
    ix1 = min(NX, int(np.ceil((xlim[1] - GRID_X0) / GRID_DX)))
    iy0 = max(0, int(np.floor((ylim[0] - GRID_Y0) / GRID_DY)))
    iy1 = min(NY, int(np.ceil((ylim[1] - GRID_Y0) / GRID_DY)))
    extent = [GRID_X0 + ix0 * GRID_DX, GRID_X0 + ix1 * GRID_DX, GRID_Y0 + iy0 * GRID_DY, GRID_Y0 + iy1 * GRID_DY]
    return slice(iy0, iy1), slice(ix0, ix1), extent


def draw_layer(ax, counts, style, xlim, ylim, ax_w_pt, ax_h_pt, zorder):
    sy, sx, extent = view_slices(xlim, ylim)
    mask = counts[sy, sx] > 0
    if not mask.any():
        return
    rx = int(round(0.5 * style["diam_pt"] / (ax_w_pt * GRID_DX / (xlim[1] - xlim[0]))))
    ry = int(round(0.5 * style["diam_pt"] / (ax_h_pt * GRID_DY / (ylim[1] - ylim[0]))))
    mask = dilate(mask, rx, ry)
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[mask] = (*matplotlib.colors.to_rgb(style["color"]), style["alpha"])
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest", aspect="auto", zorder=zorder)


def main():
    args = build_parser().parse_args()
    xlim, ylim = parse_pair(args.xlim), parse_pair(args.ylim)
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
        "axes.unicode_minus": True,
    })

    layer_paths = {}
    try:
        for model, run_dir in (("M0", args.m0_dir), ("M1", args.m1_dir)):
            for layer, fname in FILTERED_FILES.items():
                layer_paths[(model, layer)] = locate_file(run_dir, fname)
                print(f"[path] {model} {layer}: {layer_paths[(model, layer)]}")
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)

    res = {key: load_or_build(f"{key[0]}_{key[1]}", path, args) for key, path in layer_paths.items()}

    model_counts, rows = {}, []
    for model in ("M0", "M1"):
        b, t = res[(model, "binary_filtered")], res[(model, "ternary_filtered")]
        c_all = (b["counts"].astype(np.uint64) + t["counts"].astype(np.uint64)).astype(np.uint32)
        model_counts[model] = c_all
        rows.append({"model": model, "layer": "filtered_combined",
                     "n_rows": int(b["n_valid"] + t["n_valid"]), "n_outside_grid": int(b["n_out"] + t["n_out"]),
                     "tm_min": round(min(b["xmin"], t["xmin"]), 2), "tm_max": round(max(b["xmax"], t["xmax"]), 2),
                     "dh_min": round(min(b["ymin"], t["ymin"]), 2), "dh_max": round(max(b["ymax"], t["ymax"]), 2),
                     "n_tm_below_0C": count_below(c_all, 0.0), "n_tm_below_minus10C": count_below(c_all, -10.0)})

    os.makedirs(os.path.dirname(os.path.abspath(args.output_stem)), exist_ok=True)
    summary = pd.DataFrame(rows)
    summary_path = args.output_stem + "_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"[done] summary -> {summary_path}")
    if args.only_compute:
        return

    fw, fh = parse_pair(args.figsize)
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=150)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    bbox = ax.get_position()
    ax_w_pt, ax_h_pt = bbox.width * fw * 72, bbox.height * fh * 72
    draw_layer(ax, model_counts["M0"], MODEL_STYLE["M0"], xlim, ylim, ax_w_pt, ax_h_pt, zorder=1)
    draw_layer(ax, model_counts["M1"], MODEL_STYLE["M1"], xlim, ylim, ax_w_pt, ax_h_pt, zorder=2)

    if args.show_zero_line:
        ax.axvline(0, color="#333333", lw=1.4, ls="--", zorder=4)
    if args.show_t3:
        ax.scatter([args.t3_tm], [args.t3_dh], marker="o", s=560, c="white", edgecolors="black", linewidths=0.8, zorder=5)
        ax.scatter([args.t3_tm], [args.t3_dh], marker="*", s=360, c=C_T3, edgecolors="black", linewidths=0.9, zorder=6)
        if args.annotate_t3:
            ax.annotate(args.t3_label, xy=(args.t3_tm, args.t3_dh), xytext=(10, 10), textcoords="offset points",
                        fontsize=15, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.7), zorder=7)

    ax.set_xlabel(r"$\mathbf{\mathit{T}_{m}}$ (°C)", fontsize=18)
    ax.set_ylabel(r"$\mathbf{\Delta\mathit{H}_{m}}$ (J·g⁻¹)", fontsize=18)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(2.0)
    ax.tick_params(axis="both", which="both", direction="in", width=2.0, length=6, labelsize=16, top=False, right=False)

    handles = [Line2D([0], [0], marker="o", ls="None", markersize=9, markerfacecolor=MODEL_STYLE[m]["color"],
                      markeredgecolor="none", label=MODEL_STYLE[m]["label"]) for m in ("M0", "M1")]
    if args.show_t3:
        handles.append(Line2D([0], [0], marker="*", ls="None", markersize=14, markerfacecolor=C_T3,
                              markeredgecolor="black",
                              label=(rf"{args.t3_label} ($\mathit{{T}}_{{m}}$ = {args.t3_tm:.1f} °C, "
                                     rf"$\Delta\mathit{{H}}_{{m}}$ = {args.t3_dh:.1f} J·g⁻¹)")))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=14, handletextpad=0.4, borderaxespad=0.4)

    for ext in ("png", "pdf", "svg"):
        out = f"{args.output_stem}.{ext}"
        fig.savefig(out, dpi=args.dpi)
        print(f"[done] figure -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
