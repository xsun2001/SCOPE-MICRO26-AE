from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FUNCTIONS = ["exp", "rsqrt", "sigmoid", "erf", "tanh", "sin", "softsign", "arctan"]
WIDTHS = [4, 8, 16, 32]
COLORS = {4: "#94a3b8", 8: "#60a5fa", 16: "#2563eb", 32: "#172554"}


def history(path: Path, limit: int) -> tuple[list[int], list[float], list[float]]:
    rows = [row for row in csv.DictReader(path.open()) if int(row["epoch"]) <= limit]
    step = max(1, len(rows) // 1000)
    rows = rows[::step]
    return [int(r["epoch"]) for r in rows], [float(r["mse"]) for r in rows], [float(r["avg_loss"]) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=10000)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8), sharex=True)
    for ax, func in zip(axes.flat, FUNCTIONS):
        for width in WIDTHS:
            epoch, mse, loss = history(args.runs_dir / f"{func}_{width}_exp/history.csv", args.max_epochs)
            ax.plot(epoch, mse, color=COLORS[width], linewidth=1.2, label=f"{width} neurons")
            ax.plot(epoch, loss, color=COLORS[width], linewidth=0.8, linestyle="--", alpha=0.75)
        ax.set_yscale("log"); ax.set_title(func.capitalize()); ax.grid(True, which="major", linestyle=":", alpha=0.4)
    axes[0, 0].set_ylabel("Loss / MSE"); axes[1, 0].set_ylabel("Loss / MSE")
    for ax in axes[1]: ax.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"): fig.savefig(args.output_dir / f"figure17.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__": raise SystemExit(main())
