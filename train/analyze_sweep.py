from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


PLOT_FONT_FAMILY = ["DejaVu Sans"]


def register_libertinus_sans() -> None:
    preferred_family = "Libertinus Sans"
    try:
        font_manager.findfont(font_manager.FontProperties(family=preferred_family), fallback_to_default=False)
        PLOT_FONT_FAMILY[:] = [preferred_family, "DejaVu Sans"]
    except ValueError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate train.py sweep outputs and generate comparison figures."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing per-run subdirectories such as exp_4_exp.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated CSV and figures. Defaults to <run-dir>/analysis.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure/report title prefix.",
    )
    return parser.parse_args()


def load_runs(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_dir():
            continue

        summary_path = path / "summary.json"
        config_path = path / "config.json"
        if not summary_path.is_file() or not config_path.is_file():
            continue

        func, width, reparam = path.name.rsplit("_", 2)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        bounds = config.get("resolved_output_bounds", {})

        rows.append(
            {
                "run_name": path.name,
                "func": func,
                "num_units": int(width),
                "reparam": reparam,
                "best_mse": float(summary["best_mse"]),
                "last_mse": float(summary["last_mse"]),
                "best_epoch": int(summary["best_epoch"]),
                "last_epoch": int(summary["last_epoch"]),
                "l_range": float(config["l_range"]),
                "r_range": float(config["r_range"]),
                "y_min": float(bounds["y_min"]),
                "y_max": float(bounds["y_max"]),
                "output_dir": str(path),
            }
        )

    if not rows:
        raise ValueError(f"No runs with config.json and summary.json found under {run_dir}.")
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "run_name",
        "func",
        "num_units",
        "reparam",
        "best_mse",
        "last_mse",
        "best_epoch",
        "last_epoch",
        "l_range",
        "r_range",
        "y_min",
        "y_max",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pair_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["func"]), int(row["num_units"]))
        grouped.setdefault(key, {})[str(row["reparam"])] = row

    pair_rows: list[dict[str, object]] = []
    for (func, num_units), pair in sorted(grouped.items()):
        exp_row = pair.get("exp")
        none_row = pair.get("none")
        if exp_row is None or none_row is None:
            continue

        exp_mse = float(exp_row["best_mse"])
        none_mse = float(none_row["best_mse"])
        pair_rows.append(
            {
                "func": func,
                "num_units": num_units,
                "exp_best_mse": exp_mse,
                "none_best_mse": none_mse,
                "ratio_exp_to_none": exp_mse / none_mse if none_mse else math.inf,
                "better": "exp" if exp_mse < none_mse else "none" if none_mse < exp_mse else "tie",
                "exp_best_epoch": int(exp_row["best_epoch"]),
                "none_best_epoch": int(none_row["best_epoch"]),
            }
        )
    return pair_rows


def write_pair_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "func",
        "num_units",
        "exp_best_mse",
        "none_best_mse",
        "ratio_exp_to_none",
        "better",
        "exp_best_epoch",
        "none_best_epoch",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_best_mse(rows: list[dict[str, object]], output_path: Path, title_prefix: str) -> None:
    register_libertinus_sans()
    funcs = sorted({str(row["func"]) for row in rows})
    widths = sorted({int(row["num_units"]) for row in rows})
    reparams = ["none", "exp"]

    with plt.rc_context({"font.family": PLOT_FONT_FAMILY, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(
            len(widths),
            1,
            figsize=(14, 4.4 * len(widths)),
            sharex=True,
            constrained_layout=True,
        )
        if len(widths) == 1:
            axes = [axes]

        x = np.arange(len(funcs))
        bar_width = 0.36
        colors = {"none": "#94a3b8", "exp": "#2563eb"}

        for axis, width in zip(axes, widths):
            width_rows = [row for row in rows if int(row["num_units"]) == width]
            for idx, reparam in enumerate(reparams):
                values = []
                for func in funcs:
                    match = next(
                        (
                            float(row["best_mse"])
                            for row in width_rows
                            if str(row["func"]) == func and str(row["reparam"]) == reparam
                        ),
                        math.nan,
                    )
                    values.append(match)

                offset = (idx - 0.5) * bar_width
                axis.bar(
                    x + offset,
                    values,
                    width=bar_width,
                    label=reparam,
                    color=colors[reparam],
                    alpha=0.9,
                )

            axis.set_yscale("log")
            axis.set_ylabel(f"Best MSE\n{width} units")
            axis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
            axis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
            axis.legend(loc="upper right")

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(funcs, rotation=25, ha="right")
        fig.suptitle(f"{title_prefix} Best MSE by function and width", fontsize=15, fontweight="bold")
        fig.savefig(output_path, dpi=220)
        plt.close(fig)


def plot_ratio(pair_rows: list[dict[str, object]], output_path: Path, title_prefix: str) -> None:
    register_libertinus_sans()
    widths = sorted({int(row["num_units"]) for row in pair_rows})
    funcs = sorted({str(row["func"]) for row in pair_rows})

    with plt.rc_context({"font.family": PLOT_FONT_FAMILY, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(
            len(widths),
            1,
            figsize=(13, 4.0 * len(widths)),
            sharex=True,
            constrained_layout=True,
        )
        if len(widths) == 1:
            axes = [axes]

        x = np.arange(len(funcs))
        for axis, width in zip(axes, widths):
            width_rows = [row for row in pair_rows if int(row["num_units"]) == width]
            ratios = []
            colors = []
            for func in funcs:
                match = next((row for row in width_rows if str(row["func"]) == func), None)
                ratio = float(match["ratio_exp_to_none"]) if match is not None else math.nan
                ratios.append(ratio)
                colors.append("#059669" if ratio < 1.0 else "#dc2626")

            axis.bar(x, ratios, color=colors, alpha=0.9)
            axis.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
            axis.set_yscale("log")
            axis.set_ylabel(f"exp / none\n{width} units")
            axis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
            axis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(funcs, rotation=25, ha="right")
        fig.suptitle(
            f"{title_prefix} Relative best MSE (lower than 1.0 favors exp)",
            fontsize=15,
            fontweight="bold",
        )
        fig.savefig(output_path, dpi=220)
        plt.close(fig)


def write_report(path: Path, run_dir: Path, rows: list[dict[str, object]], pair_rows: list[dict[str, object]]) -> None:
    exp_better = sum(1 for row in pair_rows if str(row["better"]) == "exp")
    none_better = sum(1 for row in pair_rows if str(row["better"]) == "none")
    tie_count = len(pair_rows) - exp_better - none_better

    lines = [
        f"# Sweep Analysis: `{run_dir.name}`",
        "",
        f"- Total runs parsed: {len(rows)}",
        f"- Paired comparisons (`exp` vs `none`): {len(pair_rows)}",
        f"- `exp` better: {exp_better}",
        f"- `none` better: {none_better}",
        f"- Ties: {tie_count}",
        "",
        "## Paired comparisons",
        "",
        "| Function | Units | exp best MSE | none best MSE | exp/none | Better |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for row in pair_rows:
        lines.append(
            "| "
            f"{row['func']} | {row['num_units']} | "
            f"{float(row['exp_best_mse']):.6e} | {float(row['none_best_mse']):.6e} | "
            f"{float(row['ratio_exp_to_none']):.4f} | {row['better']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    register_libertinus_sans()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_runs(run_dir)
    pair_rows = build_pair_rows(rows)
    title_prefix = args.title or run_dir.name

    write_summary_csv(output_dir / "summary.csv", rows)
    write_pair_csv(output_dir / "paired_summary.csv", pair_rows)
    plot_best_mse(rows, output_dir / "best_mse_by_function.png", title_prefix)
    plot_ratio(pair_rows, output_dir / "exp_vs_none_ratio.png", title_prefix)
    write_report(output_dir / "report.md", run_dir, rows, pair_rows)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "total_runs": len(rows),
                "paired_runs": len(pair_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
