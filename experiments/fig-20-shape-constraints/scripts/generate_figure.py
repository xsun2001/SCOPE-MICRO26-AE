from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FUNCTIONS = ["exp", "rsqrt", "sigmoid", "erf", "tanh", "sin", "softsign", "arctan"]


def history(path: Path, limit: int) -> tuple[list[int], list[float], list[float]]:
    rows = [row for row in csv.DictReader(path.open()) if int(row["epoch"]) <= limit]
    step = max(1, len(rows) // 1000)
    rows = rows[::step]
    return [int(r["epoch"]) for r in rows], [float(r["mse"]) for r in rows], [float(r["avg_loss"]) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=5000)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8), sharex=True)
    for ax, func in zip(axes.flat, FUNCTIONS):
        gain = None
        for mode, color, label in (("exp", "#2563eb", "SCNA"), ("none", "#94a3b8", "Unconstrained NN")):
            run = args.runs_dir / f"{func}_16_{mode}"
            epoch, mse, loss = history(run / "history.csv", args.max_epochs)
            ax.plot(epoch, mse, color=color, linewidth=1.3, label=label)
            ax.plot(epoch, loss, color=color, linewidth=0.8, linestyle="--", alpha=0.75)
            value = float(json.loads((run / "summary.json").read_text())["best_mse"])
            gain = value if mode == "exp" else value / gain
        ax.set_yscale("log"); ax.set_title(f"{func.capitalize()} — {gain:.1f}x"); ax.grid(True, which="major", linestyle=":", alpha=0.4)
    axes[0, 0].set_ylabel("Loss / MSE"); axes[1, 0].set_ylabel("Loss / MSE")
    for ax in axes[1]: ax.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"): fig.savefig(args.output_dir / f"figure20.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__": raise SystemExit(main())
