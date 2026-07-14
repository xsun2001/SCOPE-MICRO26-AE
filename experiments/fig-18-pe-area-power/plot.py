#!/usr/bin/env python3
"""Draw Figure 18 from the fitted per-PE CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator


TYPE_ORDER = ["FP16-FP16", "FP8-FP16", "INT16-INT32", "INT8-INT32"]
DESIGN_ORDER = ["Baseline", "SCOPE", "OneSA", "FuseMax", "FSA"]
HATCHES = {"Baseline": "///", "SCOPE": "\\\\\\", "OneSA": "xx", "FuseMax": "..", "FSA": "++"}
BAR_WIDTH = 0.11
BAR_PITCH = 0.16
GROUP_PAD = 0.08
GROUP_GAP = 0.10


def gradient(start_hex: str, end_hex: str, steps: int) -> list[str]:
    start, end = to_rgb(start_hex), to_rgb(end_hex)
    return [
        to_hex(tuple(start[c] + (end[c] - start[c]) * i / (steps - 1) for c in range(3)))
        for i in range(steps)
    ]


COLORS = dict(zip(DESIGN_ORDER, gradient("#99d98c", "#1a759f", len(DESIGN_ORDER)), strict=True))


def load(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            type_name = f"{row['data_type']}-{row['acc_type']}"
            result.setdefault(type_name, {})[row["design"]] = {
                "area": float(row["area_per_pe_um2"]),
                "power": float(row["power_per_pe_mw"]),
            }
    return result


def plot_metric(ax: plt.Axes, data: dict, metric: str, ylabel: str) -> set[str]:
    spans, centers = [], []
    cursor = 0.0
    for type_name in TYPE_ORDER:
        count = sum(design in data.get(type_name, {}) for design in DESIGN_ORDER)
        span = (max(count, 1) - 1) * BAR_PITCH + BAR_WIDTH + 2 * GROUP_PAD
        spans.append((cursor, cursor + span))
        centers.append(cursor + span / 2)
        cursor += span + GROUP_GAP

    for index in range(len(spans) - 1):
        ax.axvline((spans[index][1] + spans[index + 1][0]) / 2, color="#4C4C4C", linewidth=0.9)

    used, maximum = set(), 0.0
    for group, type_name in enumerate(TYPE_ORDER):
        available = [(name, data[type_name][name][metric]) for name in DESIGN_ORDER if name in data.get(type_name, {})]
        baseline = data[type_name]["Baseline"][metric]
        start = centers[group] - BAR_PITCH * (len(available) - 1) / 2
        for index, (design, value) in enumerate(available):
            x = start + index * BAR_PITCH
            ax.bar(x, value, width=BAR_WIDTH, color=COLORS[design], edgecolor="black", linewidth=1, hatch=HATCHES[design])
            maximum = max(maximum, value)
            used.add(design)
            ax.text(
                x,
                value + maximum * 0.05,
                f"{value / baseline:.2f}x",
                ha="center",
                va="bottom",
                fontsize=20 if design == "SCOPE" else 18,
                fontweight="bold" if design == "SCOPE" else "normal",
                rotation=90,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
            )

    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xticks(centers, TYPE_ORDER)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.9, color="#707070")
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.6, alpha=0.8, color="#8A8A8A")
    ax.set_axisbelow(True)
    ax.set_xlim(spans[0][0] - 0.08, spans[-1][1] + 0.08)
    ax.set_ylim(0, maximum * 1.5)
    ax.tick_params(axis="y", labelrotation=90, pad=2)
    return used


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
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    used = plot_metric(axes[0], data, "area", r"Area ($\mu$m$^2$/PE)")
    used |= plot_metric(axes[1], data, "power", "Power (mW/PE)")
    handles = [Patch(facecolor=COLORS[name], edgecolor="black", hatch=HATCHES[name], label=name) for name in DESIGN_ORDER if name in used]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.6, 0.95), columnspacing=0.9, handlelength=1.6)
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.11, top=0.82, hspace=0.12)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = args.output_dir / f"figure18.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
