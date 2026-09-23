import csv
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap
import matplotlib.patheffects as pe

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).parent
DEFAULT_CSV = BASE_DIR / "VRAM vs Inferencetime.csv"

LINESTYLES = {"CPU": ":", "GPU": "-"}
COOL = plt.get_cmap("cool")

DNF_COLORS = {"time": "grey", "oom": "black", "device_error": "black"}

NUMERIC_FIELDS = ["params", "params_M", "memory_MiB", "vram_GiB",
                  "vram_limit_mib", "min", "max", "median", "mean", "std"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot inference-time benchmark results saved by "
                    "inference_time_DL_model.py."
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"path to the results CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--width-exp-min", type=int, default=None,
        help="minimum exponent for head width to plot (width = 2**exp). "
             "Default: smallest width present in the CSV.",
    )
    parser.add_argument(
        "--width-exp-max", type=int, default=None,
        help="maximum exponent for head width to plot (width = 2**exp). "
             "Default: largest width present in the CSV.",
    )
    parser.add_argument(
        "--stages-exp-min", type=int, default=None,
        help="minimum exponent for the # head stages sweep to plot "
             "(stages = 0, 2**exp). Default: smallest stage present in the CSV.",
    )
    parser.add_argument(
        "--stages-exp-max", type=int, default=None,
        help="maximum exponent for the # head stages sweep to plot "
             "(stages = 2**exp). Default: largest stage present in the CSV.",
    )
    return parser.parse_args()


# ============================================================
# Data loading / filtering
# ============================================================

def load_results(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            r = dict(raw)
            r["head_width"] = int(r["head_width"])
            r["n_head_stages"] = int(r["n_head_stages"])
            for key in NUMERIC_FIELDS:
                try:
                    r[key] = float(r[key]) if r[key] != "" else np.nan
                except (TypeError, ValueError):
                    r[key] = np.nan
            r["dnf_reason"] = r.get("dnf_reason", "")
            rows.append(r)
    return rows


def available_stages(all_results):
    return sorted({r["n_head_stages"] for r in all_results})


def available_widths(all_results):
    return sorted({r["head_width"] for r in all_results})


def select_stages(all_results, exp_min, exp_max):
    return [
        s for s in available_stages(all_results)
        if s != 0 and exp_min <= np.log2(s) <= exp_max
    ]


def select_widths(all_results, exp_min, exp_max):
    return [
        w for w in available_widths(all_results)
        if exp_min <= np.log2(w) <= exp_max
    ]


def resolve_ranges(all_results, args):
    stages = available_stages(all_results)
    widths = available_widths(all_results)

    stages_exp_min = args.stages_exp_min
    if stages_exp_min is None:
        positive = [s for s in stages if s != 0]
        stages_exp_min = int(np.log2(min(positive))) if positive else 0

    stages_exp_max = args.stages_exp_max
    if stages_exp_max is None:
        stages_exp_max = int(np.log2(max(stages)))

    width_exp_min = args.width_exp_min
    if width_exp_min is None:
        width_exp_min = int(np.log2(min(widths)))

    width_exp_max = args.width_exp_max
    if width_exp_max is None:
        width_exp_max = int(np.log2(max(widths)))

    return (stages_exp_min, stages_exp_max), (width_exp_min, width_exp_max)


def results_for(all_results, device_name, head_width):
    return [r for r in all_results
            if r["device"] == device_name and r["head_width"] == head_width]


def dnf_for(all_results, device_name, head_width):
    res = results_for(all_results, device_name, head_width)
    if not res:
        return ""
    reasons = {r["dnf_reason"] for r in res if r["dnf_reason"]}
    return next(iter(reasons), "")


# ============================================================
# Plotting helpers
# ============================================================

def cool_color(width, widths):
    log2_min = np.log2(min(widths))
    log2_max = np.log2(max(widths))
    t = (np.log2(width) - log2_min) / (log2_max - log2_min)
    return COOL(t)


def add_colorbar(fig, ax, widths):
    norm = LogNorm(vmin=min(widths), vmax=max(widths))
    smap = ScalarMappable(cmap=COOL, norm=norm)
    smap.set_array([])
    cbar = fig.colorbar(smap, ax=ax)
    cbar.set_label("Head Conv1d width (input/output channels)", fontsize=11)
    cbar.set_ticks(widths)
    cbar.set_ticklabels([f"2^{int(round(np.log2(w)))}" for w in widths])
    cbar.ax.minorticks_off()
    return cbar


def stage_pos(s):
    return 0.5 if s == 0 else float(s)


def stage_label(s):
    """'0' for the baseline, '2^exp' for powers of two."""
    if s == 0:
        return "0"
    return f"2^{int(round(np.log2(s)))}"


def plot_curve(ax, results, stages, color, ls, dnf_marker=None):
    pos = {s: stage_pos(s) for s in stages}
    pts = [(pos[r["n_head_stages"]], r) for r in results
           if r["n_head_stages"] in pos and np.isfinite(r["median"])]
    pts.sort(key=lambda t: t[0])
    if not pts:
        return

    xs = [p[0] for p in pts]
    medians = [p[1]["median"] for p in pts]
    mins = [p[1]["min"] for p in pts]
    maxs = [p[1]["max"] for p in pts]
    stds = [p[1]["std"] for p in pts]
    n = len(pts)

    for i in range(n - 1):
        ax.fill_between(
            [xs[i], xs[i + 1]],
            [mins[i], mins[i + 1]],
            [maxs[i], maxs[i + 1]],
            color=color, alpha=0.08, linewidth=0,
        )
        ax.fill_between(
            [xs[i], xs[i + 1]],
            [medians[i] - stds[i], medians[i + 1] - stds[i + 1]],
            [medians[i] + stds[i], medians[i + 1] + stds[i + 1]],
            color=color, alpha=0.20, linewidth=0,
        )
        ax.plot([xs[i], xs[i + 1]], [medians[i], medians[i + 1]],
                color=color, linestyle=ls, linewidth=2.5)

    for i in range(n):
        ax.plot(xs[i], medians[i], "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.plot([xs[i], xs[i]],
                [medians[i] - stds[i], medians[i] + stds[i]],
                color=color, linewidth=1.2)

    if dnf_marker and n > 0:
        ax.plot(xs[-1], medians[-1], "x", color=dnf_marker, markersize=14,
                markeredgewidth=3.0, alpha=0.5, zorder=5, clip_on=False)


def finalize_ax(ax, stages, ylim=None):
    ax.set_xscale("log")
    xs = [stage_pos(s) for s in stages]
    ax.set_xticks(xs)
    ax.set_xticklabels([stage_label(s) for s in stages], fontsize=10)
    ax.set_xlim(min(xs) / 1.5, max(xs) * 1.5)
    ax.set_xlabel("# of head_width→head_width stages", fontsize=11)
    ax.set_ylabel("Inference Time [ms]", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for ref in (1000, 125, 35, 2):
        ax.axhline(y=ref, color="grey", linestyle=":", linewidth=1.2, zorder=0)
    ax.set_yticks(sorted(set(list(ax.get_yticks()) + [0, 2, 35, 125, 1000])))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    if ylim is not None:
        ax.set_ylim(ylim)


def clean_device_name(raw):
    return raw.replace("\x00", "").strip()


def device_label(raw):
    return "CPU" if clean_device_name(raw).upper() == "CPU" else "GPU"


IGPU_NAME_MARKERS = (
    "radeon(tm) graphics", "intel", "uhd graphics", "hd graphics",
    "iris", "integrated", "vega",
)


def device_short_name(raw):
    """Short title for a device subplot: CPU, GPU or iGPU."""
    clean = clean_device_name(raw)
    if clean.upper() == "CPU":
        return "CPU"
    low = clean.lower()
    if any(marker in low for marker in IGPU_NAME_MARKERS):
        return "iGPU"
    return "GPU"


def device_slug(raw):
    clean = clean_device_name(raw)
    if clean.upper() == "CPU":
        return "CPU"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", clean).strip("_")
    return slug or "GPU"


def device_entries(all_results):
    entries = []
    for raw in sorted({r["device"] for r in all_results}):
        clean = clean_device_name(raw)
        entries.append((raw, device_label(raw), clean))
    return entries


def device_proxies(entries):
    return [
        Line2D([0], [0], color="0.3", linestyle=LINESTYLES[label],
               linewidth=2.5, label=clean)
        for raw, label, clean in entries
    ]


def dnf_proxies():
    return [
        Line2D([0], [0], marker="x", color=DNF_COLORS["time"], linestyle="None",
               markersize=12, markeredgewidth=3.0,
               label="DNF: inference > 1 s"),
        Line2D([0], [0], marker="x", color=DNF_COLORS["oom"], linestyle="None",
               markersize=12, markeredgewidth=3.0,
               label="DNF: OOM / VRAM / device error"),
    ]


def draw_curves(ax, all_results, entries, stages, widths):
    for raw, label, clean in entries:
        for width in widths:
            res = results_for(all_results, raw, width)
            marker = DNF_COLORS.get(dnf_for(all_results, raw, width))
            plot_curve(ax, res, stages, cool_color(width, widths),
                       LINESTYLES[label], dnf_marker=marker)


# ============================================================
# Time heatmap (x = # stages, y = head width, color = time)
# ============================================================

THRESHOLDS_MS = (2, 35, 125, 1000)
TIME_MAX_MS = 1000.0


def cell_reason_by(all_results, raw_device, stage_i, width_i):
    return {
        (r["head_width"], r["n_head_stages"]): (r["dnf_reason"], r["vram_limit_mib"])
        for r in all_results
        if r["device"] == raw_device
        and r["n_head_stages"] in stage_i
        and r["head_width"] in width_i
    }


def heatmap_overlays(cell_info, stages, widths, time_grid):
    stage_i = {s: i for i, s in enumerate(stages)}
    width_i = {w: j for j, w in enumerate(widths)}
    overlays = {}
    for w in widths:
        j = width_i[w]
        for i in range(len(stages)):
            if not np.isfinite(time_grid[j, i]):
                reason, _ = cell_info.get((w, stages[i]), ("", np.nan))
                if reason == "time":
                    overlays[(j, i)] = DNF_COLORS["time"]
                elif reason in ("oom", "device_error"):
                    overlays[(j, i)] = DNF_COLORS["oom"]
    return overlays


def format_limit(mib):
    if np.isfinite(mib) and mib >= 1024:
        return f"{mib / 1024:g} GB"
    return f"{mib:g} MB"


def quantize_to_thresholds(time_grid):
    bounds = [0.0] + list(THRESHOLDS_MS)

    idx = np.searchsorted(bounds, time_grid, side="right") - 1
    idx = np.clip(idx, 0, len(bounds) - 2)
    qgrid = np.where(np.isfinite(time_grid), np.array(bounds)[idx], np.nan)

    norm = LogNorm(vmin=1.0, vmax=TIME_MAX_MS)
    cmap = ListedColormap([COOL(norm(max(b, 1.0))) for b in bounds[:-1]])
    bnorm = BoundaryNorm(bounds, len(bounds) - 1, clip=True)
    return qgrid, cmap, bnorm, bounds


def draw_heatmap_ax(ax, all_results, raw_device, stages, widths):
    stage_i = {s: i for i, s in enumerate(stages)}
    width_i = {w: j for j, w in enumerate(widths)}
    n_stages = len(stages)
    n_widths = len(widths)

    time_grid = np.full((n_widths, n_stages), np.nan)
    for r in all_results:
        if r["device"] != raw_device:
            continue
        if r["n_head_stages"] not in stage_i or r["head_width"] not in width_i:
            continue
        i = stage_i[r["n_head_stages"]]
        j = width_i[r["head_width"]]
        time_grid[j, i] = r["mean"]

    cell_info = cell_reason_by(all_results, raw_device, stage_i, width_i)
    overlays = heatmap_overlays(cell_info, stages, widths, time_grid)

    qgrid, cmap, bnorm, bounds = quantize_to_thresholds(time_grid)
    xe = np.arange(-0.5, n_stages + 0.5)
    ye = np.arange(-0.5, n_widths + 0.5)
    pc = ax.pcolormesh(xe, ye, qgrid, cmap=cmap, norm=bnorm,
                       edgecolors="white", linewidth=1.0)

    for (j, i), color in overlays.items():
        ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color,
                               edgecolor="white", linewidth=0.6, zorder=3))

    for j in range(n_widths):
        for i in range(n_stages):
            v = time_grid[j, i]
            if np.isfinite(v):
                ax.text(i, j, f"{v:.3g}", ha="center", va="center",
                        fontsize=6.0, color="white", zorder=7,
                        path_effects=[pe.withStroke(linewidth=1.4,
                                                    foreground="black")])
            else:
                color = overlays.get((j, i))
                txt = ""
                if color == DNF_COLORS["time"]:
                    txt = "> 1s"
                elif color == DNF_COLORS["oom"]:
                    _, limit = cell_info.get((widths[j], stages[i]), ("", np.nan))
                    txt = f"> {format_limit(limit)}"
                if txt:
                    ax.text(i, j, txt, ha="center", va="center",
                            fontsize=6.0, color="white", zorder=7,
                            path_effects=[pe.withStroke(linewidth=1.4,
                                                        foreground="black")])

    ax.set_xticks(range(n_stages))
    ax.set_xticklabels([stage_label(s) for s in stages], fontsize=10)
    ax.set_yticks(range(n_widths))
    ax.set_yticklabels([f"2^{int(round(np.log2(w)))}" for w in widths], fontsize=9)
    ax.set_xlabel("Conv1D width", fontsize=11)
    ax.set_ylabel("number of layers", fontsize=11)
    ax.set_xlim(-0.5, n_stages - 0.5)
    ax.set_ylim(-0.5, n_widths - 0.5)
    return pc


def time_heatmap_colorbar(fig, pc, ax, orientation="vertical", pad=0.05):
    cbar = fig.colorbar(pc, ax=ax, orientation=orientation, pad=pad)
    cbar.set_label("Inference Time [ms] (mean)", fontsize=11, rotation=0,
                   ha="center", va="top")
    ticks = list(THRESHOLDS_MS)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:g}" for t in ticks])
    cbar.ax.minorticks_off()
    return cbar


def plot_time_heatmap(all_results, raw_device, label, stages, widths, out_path):
    fig, ax = plt.subplots(figsize=(11, 8))
    pc = draw_heatmap_ax(ax, all_results, raw_device, stages, widths)
    time_heatmap_colorbar(fig, pc, ax)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_time_heatmap_grid(all_results, entries, stages, widths, ncols, out_path):
    n = len(entries)
    nrows = (n + ncols - 1) // ncols
    stacked = ncols == 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.6 * ncols, 9.0 * nrows),
                             squeeze=False)
    pc = None
    for i, (raw, label, clean) in enumerate(entries):
        ax = axes[i // ncols, i % ncols]
        pc = draw_heatmap_ax(ax, all_results, raw, stages, widths)
        ax.set_title(device_short_name(raw), fontsize=12)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    if stacked:
        time_heatmap_colorbar(fig, pc, axes.ravel().tolist(),
                              orientation="horizontal", pad=0.04)
    else:
        time_heatmap_colorbar(fig, pc, axes.ravel().tolist(),
                              orientation="vertical", pad=0.03)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_curves_grid(all_results, entries, stages, widths, ncols, out_path):
    n = len(entries)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.8 * nrows),
                             squeeze=False)
    for i, (raw, label, clean) in enumerate(entries):
        ax = axes[i // ncols, i % ncols]
        draw_curves(ax, all_results, [(raw, label, clean)], stages, widths)
        finalize_ax(ax, stages, ylim=(0, 1000))
        ax.set_title(device_short_name(raw), fontsize=12)
        ax.legend(handles=dnf_proxies(), loc="lower right", fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    norm = LogNorm(vmin=min(widths), vmax=max(widths))
    smap = ScalarMappable(cmap=COOL, norm=norm)
    smap.set_array([])
    cbar = fig.colorbar(smap, ax=axes.ravel().tolist(), pad=0.02)
    cbar.set_label("Head Conv1d width (input/output channels)", fontsize=11)
    cbar.set_ticks(widths)
    cbar.set_ticklabels([f"2^{int(round(np.log2(w)))}" for w in widths])
    cbar.ax.minorticks_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    all_results = load_results(args.csv)
    if not all_results:
        print(f"No data found in {args.csv}")
        return

    (stages_exp_min, stages_exp_max), (width_exp_min, width_exp_max) = \
        resolve_ranges(all_results, args)

    stages = select_stages(all_results, stages_exp_min, stages_exp_max)
    widths = select_widths(all_results, width_exp_min, width_exp_max)

    device_names = sorted({r["device"] for r in all_results})
    device_entries_list = device_entries(all_results)

    print(f"CSV: {args.csv}")
    print(f"# head stages shown: {stages}")
    print(f"head widths shown:   {widths}")

    if not stages or not widths:
        print("Selected ranges contain no data; nothing to plot")
        return

    filtered = [r for r in all_results
                if r["n_head_stages"] in stages and r["head_width"] in widths]
    ylim = (0, 1000)

    # ============================================================
    # Individual plots (one per device; x = #stages, color = width)
    # ============================================================

    for raw, label, clean in device_entries_list:
        fig, ax = plt.subplots(figsize=(10, 6))
        draw_curves(ax, all_results, [(raw, label, clean)], stages, widths)
        finalize_ax(ax, stages, ylim=ylim)
        add_colorbar(fig, ax, widths)
        proxy = [Line2D([0], [0], color="0.3", linestyle=LINESTYLES[label],
                        linewidth=2.5, label=clean)]
        ax.legend(handles=proxy + dnf_proxies(), loc="lower right", fontsize=9)
        plt.tight_layout()
        out = BASE_DIR / f"VRAM vs Inferencetime_{device_slug(raw)}.png"
        fig.savefig(out, dpi=300)
        print(f"Plot saved: {out}")

        heatmap_out = BASE_DIR / f"Time heatmap_{device_slug(raw)}.png"
        plot_time_heatmap(all_results, raw, label, stages, widths, heatmap_out)

    # grid of heatmaps: all devices side by side, and stacked
    if len(device_entries_list) > 1:
        out = BASE_DIR / "Time heatmap_side-by-side.png"
        plot_time_heatmap_grid(all_results, device_entries_list, stages, widths,
                               3, out)

        out = BASE_DIR / "Time heatmap_stacked.png"
        plot_time_heatmap_grid(all_results, device_entries_list, stages, widths,
                               1, out)

    # ============================================================
    # Combined plot (x = #stages, color = width, style = device)
    # ============================================================

    if len(device_entries_list) > 1:
        out = BASE_DIR / "VRAM vs Inferencetime_side-by-side.png"
        plot_curves_grid(all_results, device_entries_list, stages, widths, 3, out)

        out = BASE_DIR / "VRAM vs Inferencetime_stacked.png"
        plot_curves_grid(all_results, device_entries_list, stages, widths, 1, out)

        fig, ax = plt.subplots(figsize=(10, 6))
        draw_curves(ax, all_results, device_entries_list, stages, widths)
        finalize_ax(ax, stages, ylim=ylim)
        add_colorbar(fig, ax, widths)
        ax.legend(handles=device_proxies(device_entries_list) + dnf_proxies(),
                  loc="lower right", fontsize=9, ncol=2)
        plt.tight_layout()
        out = BASE_DIR / "VRAM vs Inferencetime_combined.png"
        fig.savefig(out, dpi=300)
        print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()