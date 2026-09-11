#!/usr/bin/env python3
"""Draw Figure 19 from its clean incremental-overhead CSV."""

from __future__ import annotations

import argparse
import io
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox


ACC_ORDER = ["FP16", "INT32"]
METHOD_ORDER = ["SCNA-8", "SCNA-16", "OneSA", "FuseMax", "FSA", "NN-LUT", "T-LUT", "PICACHU"]
HATCHES = {
    "SCNA-8": "///",
    "SCNA-16": "\\\\\\",
    "OneSA": "xx",
    "FuseMax": "..",
    "FSA": "++",
    "NN-LUT": "oo",
    "T-LUT": "--",
    "PICACHU": "**",
}
BAR_WIDTH = 0.15
BAR_PITCH = 0.21
GROUP_PAD = 0.20
GROUP_GAP = 0.10


def gradient(start_hex: str, end_hex: str, steps: int) -> list[str]:
    start, end = to_rgb(start_hex), to_rgb(end_hex)
    return [
        to_hex(tuple(start[c] + (end[c] - start[c]) * i / (steps - 1) for c in range(3)))
        for i in range(steps)
    ]


COLORS = dict(zip(METHOD_ORDER, gradient("#99d98c", "#1a759f", len(METHOD_ORDER)), strict=True))


def load(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["acc_type"], {})[row["method"]] = {
                "area": float(row["area_um2"]),
                "power": float(row["power_mw"]),
            }
    return result


def plot_metric(ax: plt.Axes, data: dict, metric: str, ylabel: str) -> set[str]:
    spans, centers = [], []
    cursor = 0.0
    for acc_type in ACC_ORDER:
        count = sum(method in data.get(acc_type, {}) for method in METHOD_ORDER)
        span = (count - 1) * BAR_PITCH + BAR_WIDTH + 2 * GROUP_PAD
        spans.append((cursor, cursor + span))
        centers.append(cursor + span / 2)
        cursor += span + GROUP_GAP
    ax.axvline((spans[0][1] + spans[1][0]) / 2, color="#4C4C4C", linewidth=1.1)

    used, maximum, minimum = set(), 0.0, float("inf")
    for group, acc_type in enumerate(ACC_ORDER):
        available = [(method, data[acc_type][method][metric]) for method in METHOD_ORDER if method in data.get(acc_type, {})]
        baseline = data[acc_type]["SCNA-8"][metric]
        start = centers[group] - BAR_PITCH * (len(available) - 1) / 2
        for index, (method, value) in enumerate(available):
            x = start + index * BAR_PITCH
            ax.bar(x, value, width=BAR_WIDTH, color=COLORS[method], edgecolor="black", linewidth=1, hatch=HATCHES[method])
            used.add(method)
            maximum, minimum = max(maximum, value), min(minimum, value)
            ax.text(
                x,
                value * 1.08,
                f"{value / baseline:.1f}x",
                ha="center",
                va="bottom",
                fontsize=20,
                rotation=90,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
            )

    ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xticks(centers, ACC_ORDER)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.9, color="#707070")
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.6, alpha=0.8, color="#8A8A8A")
    ax.set_axisbelow(True)
    ax.set_xlim(spans[0][0] - 0.08, spans[-1][1] + 0.08)
    ax.set_ylim(max(minimum * 0.75, 1e-3), maximum * 11)
    ax.tick_params(axis="y", labelrotation=90, pad=2)
    return used


def align_paper_style(fig: plt.Figure) -> Bbox:
    """Apply reference styling while retaining published axes and export bounds."""
    fig.savefig(io.BytesIO(), format="png", dpi=300, bbox_inches="tight")
    fig.savefig(io.BytesIO(), format="pdf", bbox_inches="tight")
    fig.set_layout_engine(None)
    first_panel = fig.axes[0].get_position()
    left = first_panel.x0 * fig.get_figwidth() - 57.4969 / 72
    top = first_panel.y1 * fig.get_figheight() + 77.983 / 72
    bounds = Bbox.from_bounds(left, top - 394.6 / 72, 669.5 / 72, 394.6 / 72)
    paper_scale = 239.4 / 669.5
    for ax in fig.axes:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.35 / paper_scale)
        ax.tick_params(width=0.35 / paper_scale)
        ax.grid(False, which="minor", axis="both")
        ax.grid(True, which="major", axis="y", linestyle="--",
                linewidth=0.34 / paper_scale, color="#b0b0b0", alpha=0.22)
        # Preserve category dividers, but make them as quiet as the grid.
        for line in ax.lines:
            line.set_color("#b0b0b0")
            line.set_alpha(0.22)
            line.set_linestyle("--")
            line.set_linewidth(0.34 / paper_scale)
    for legend in fig.legends:
        legend.set_frame_on(False)
    return bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.family": "sans-serif",
            "font.sans-serif": ["Libertinus Sans"],
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.linewidth": 1.5,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 20,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    data = load(args.input)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True)
    used = plot_metric(axes[0], data, "area", r"Area ($\mu$m$^2$)")
    used |= plot_metric(axes[1], data, "power", "Power (mW)")
    handles = [Patch(facecolor=COLORS[name], edgecolor="black", hatch=HATCHES[name], label=name) for name in METHOD_ORDER if name in used]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.6, 0.97), columnspacing=0.9, handlelength=1.6)
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.2, top=0.8, hspace=0.12)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bounds = align_paper_style(fig)
    for suffix in ("png", "pdf"):
        path = args.output_dir / f"figure19.{suffix}"
        with plt.rc_context({"pdf.fonttype": 3}):
            fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches=bounds, pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
