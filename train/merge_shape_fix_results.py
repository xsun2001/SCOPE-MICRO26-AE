from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator

from analyze_sweep import build_pair_rows, load_runs, plot_best_mse, plot_ratio, write_pair_csv, write_summary_csv
from plot_history import downsample, infer_smooth_span, load_history, smooth_metric


CANONICAL_FUNC_ORDER = [
    "exp",
    "exp2",
    "sigmoid",
    "erf",
    "rsqrt",
    "recip",
    "sin",
    "tanh",
    "softsign",
    "arctan",
]

PAPER_PANEL_ORDER = [
    "exp",
    "rsqrt",
    "sigmoid",
    "erf",
    "tanh",
    "sin",
    "softsign",
    "arctan",
]
PAPER_OMITTED_FUNCS = {"exp2"}

DISPLAY_NAMES = {
    "exp": "Exp",
    "exp2": "Exp2",
    "sigmoid": "Sigmoid",
    "erf": "Erf",
    "rsqrt": "Rsqrt",
    "recip": "Recip",
    "sin": "Sin",
    "tanh": "Tanh",
    "softsign": "Softsign",
    "arctan": "Arctan",
}

ANNOTATION_HEADROOM = {
    "softsign": 12.0,
    "arctan": 8.0,
}

PLOT_FONT_FAMILY = ["DejaVu Sans"]
DEFAULT_FULL_WIDTH_SCNA_MAX_EPOCHS = [10_000, 20_000, 30_000]
GRID_ROWS = 2
GRID_COLS = 4
GRID_EXPECTED_PANELS = GRID_ROWS * GRID_COLS
GRID_FIGSIZE = (3.52, 2.05)
GRID_YTICK_LABEL_SIZE = 5.0
GRID_LEGEND_Y = 1.005
GRID_SUBPLOTS_LEFT = 0.11
GRID_SUBPLOTS_RIGHT = 0.995
GRID_SUBPLOTS_BOTTOM = 0.12
GRID_SUBPLOTS_TOP = 0.865
GRID_SUBPLOTS_WSPACE = 0.10
GRID_SUBPLOTS_HSPACE = 0.08
GRID_LEGEND_HANDLE_LENGTH = 1.8
GRID_LEGEND_HANDLE_TEXT_PAD = 0.45
GRID_LEGEND_COLUMN_SPACING = 0.85
GRID_LEGEND_BORDERPAD = 0.35
GRID_ANNOTATION_FONT_SIZE = 5.2
GRID_AXIS_LABEL_SIZE = 6.4
GRID_SAVE_PAD_INCHES = 0.015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a baseline sweep with an overlay sweep and render merged analysis artifacts."
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--corrected-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will contain the merged sweep symlinks and generated analysis.",
    )
    parser.add_argument(
        "--convergence-width",
        type=int,
        default=8,
        help="Width used by the all-in-one 2x4 convergence figure.",
    )
    parser.add_argument(
        "--convergence-max-epochs",
        type=int,
        default=5_000,
        help="Maximum epoch shown in the all-in-one convergence figure.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title prefix for generated figures and reports.",
    )
    parser.add_argument(
        "--subplot-box-aspect",
        type=float,
        default=1.0,
        help="Height-to-width ratio used for each panel in the all-in-one grid figures.",
    )
    parser.add_argument(
        "--full-width-scna-max-epochs",
        type=int,
        nargs="+",
        default=DEFAULT_FULL_WIDTH_SCNA_MAX_EPOCHS,
        help=(
            "Epoch windows used for the all-width SCNA comparison figures. "
            "Each value emits a separate suffixed artifact such as `_12k`."
        ),
    )
    parser.add_argument(
        "--full-width-scna-x-axis-limit-overrides",
        nargs="*",
        default=[],
        metavar="DATA_EPOCHS:AXIS_LIMIT",
        help=(
            "Optional display-limit overrides for full-width SCNA figures. "
            "For example, `10000:12000` keeps the last labeled tick at 10k while drawing the curves out to 12k."
        ),
    )
    return parser.parse_args()


def geometric_mean(values: list[float]) -> float:
    if not values:
        return math.nan
    if any(value <= 0.0 for value in values):
        raise ValueError("geometric_mean requires positive values.")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_epoch_limit_overrides(specs: list[str]) -> dict[int, int]:
    overrides: dict[int, int] = {}
    for spec in specs:
        try:
            data_epochs_raw, axis_limit_raw = spec.split(":", 1)
            data_epochs = int(data_epochs_raw)
            axis_limit = int(axis_limit_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid SCNA axis override `{spec}`. Expected DATA_EPOCHS:AXIS_LIMIT, e.g. 10000:12000."
            ) from exc
        if data_epochs <= 0 or axis_limit <= 0:
            raise ValueError("SCNA axis overrides must use positive integers.")
        if axis_limit < data_epochs:
            raise ValueError("SCNA axis override limit must be greater than or equal to the data epoch cap.")
        overrides[data_epochs] = axis_limit
    return overrides


def func_sort_key(func: str) -> tuple[int, str]:
    if func in CANONICAL_FUNC_ORDER:
        return (CANONICAL_FUNC_ORDER.index(func), func)
    return (len(CANONICAL_FUNC_ORDER), func)


def pair_sort_key(row: dict[str, object]) -> tuple[int, str, int]:
    return (*func_sort_key(str(row["func"])), int(row["num_units"]))


def paper_panel_sort_key(row: dict[str, object]) -> tuple[int, str]:
    func = str(row["func"])
    if func in PAPER_PANEL_ORDER:
        return (PAPER_PANEL_ORDER.index(func), func)
    return (len(PAPER_PANEL_ORDER), func)


def format_epoch_tick(value: float, _: float) -> str:
    if value <= 0:
        return "0"
    if value >= 1000:
        scaled = value / 1000.0
        if abs(scaled - round(scaled)) < 1e-9:
            return f"{int(round(scaled))}k"
        return f"{scaled:.1f}k"
    return str(int(round(value)))


def compute_axis_limits(series: list[np.ndarray]) -> tuple[float, float]:
    positive = [values[np.isfinite(values) & (values > 0.0)] for values in series]
    positive = [values for values in positive if values.size > 0]
    if not positive:
        return (1e-6, 1e0)

    all_values = np.concatenate(positive)
    burn_in = min(25, max(3, all_values.size // 60))
    focused = [values[burn_in:] if values.size > burn_in else values for values in positive]
    focused_values = np.concatenate([values for values in focused if values.size > 0])

    y_min = float(np.min(all_values))
    y_focus_upper = float(np.quantile(focused_values, 0.985))
    y_upper = max(y_focus_upper, y_min * 20.0)

    lower_decade = math.floor(math.log10(y_min))
    upper_decade = math.ceil(math.log10(y_upper))
    if upper_decade - lower_decade < 2:
        upper_decade = lower_decade + 2

    return (10.0 ** lower_decade, 10.0 ** upper_decade)


def register_libertinus_sans() -> None:
    preferred_family = "Libertinus Sans"
    try:
        font_manager.findfont(font_manager.FontProperties(family=preferred_family), fallback_to_default=False)
        PLOT_FONT_FAMILY[:] = [preferred_family, "DejaVu Sans"]
        return
    except ValueError:
        pass

    font_dir = Path(__file__).resolve().parent / "assets" / "fonts"
    found_font = False
    for font_name in [
        "LibertinusSans-Regular.otf",
        "LibertinusSans-Bold.otf",
        "LibertinusSans-Italic.otf",
    ]:
        font_path = font_dir / font_name
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            found_font = True
    if found_font:
        PLOT_FONT_FAMILY[:] = [preferred_family, "DejaVu Sans"]


def style_grid_y_ticks(axis: matplotlib.axes.Axes, *, show_labels: bool) -> None:
    axis.tick_params(axis="y", labelleft=show_labels, labelrotation=90, pad=2 if show_labels else 0)
    for label in axis.get_yticklabels():
        label.set_horizontalalignment("center")
        label.set_verticalalignment("center")


def discover_run_dirs(run_dir: Path) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for path in sorted(run_dir.iterdir()):
        if not path.is_dir():
            continue
        if (path / "config.json").is_file() and (path / "summary.json").is_file():
            runs[path.name] = path
    return runs


def reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def symlink_relative(source: Path, dest: Path) -> None:
    if dest.is_symlink() or dest.exists():
        reset_path(dest)
    rel_source = os.path.relpath(source, start=dest.parent)
    dest.symlink_to(rel_source)


def merge_runs(
    baseline_dir: Path,
    corrected_dir: Path,
    output_dir: Path,
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    baseline_runs = discover_run_dirs(baseline_dir)
    corrected_runs = discover_run_dirs(corrected_dir)
    if not baseline_runs:
        raise ValueError(f"No train.py run directories found under {baseline_dir}.")
    if not corrected_runs:
        raise ValueError(f"No overlay train.py run directories found under {corrected_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_names = sorted(set(baseline_runs) | set(corrected_runs))
    replaced = sorted(set(baseline_runs) & set(corrected_runs))
    added = sorted(set(corrected_runs) - set(baseline_runs))
    source_rows: list[dict[str, str]] = []

    for run_name in selected_names:
        source_group = "overlay" if run_name in corrected_runs else "baseline"
        source_dir = corrected_runs.get(run_name, baseline_runs.get(run_name))
        if source_dir is None:
            continue
        symlink_relative(source_dir.resolve(), output_dir / run_name)
        source_rows.append(
            {
                "run_name": run_name,
                "source_group": source_group,
                "source_dir": str(source_dir.resolve()),
            }
        )

    return source_rows, replaced, added


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pick_convergence_funcs(
    pair_rows: list[dict[str, object]],
    width: int,
    *,
    preferred_order: list[str] | None = None,
    omitted_funcs: set[str] | None = None,
    max_funcs: int | None = None,
) -> list[str]:
    omitted = omitted_funcs or set()
    available = {
        str(row["func"])
        for row in pair_rows
        if int(row["num_units"]) == width and str(row["func"]) not in omitted
    }
    order = preferred_order or CANONICAL_FUNC_ORDER
    ordered = [func for func in order if func in available]
    extra = sorted(available - set(ordered))
    selected = ordered + extra
    if max_funcs is not None:
        selected = selected[:max_funcs]
    return selected


def build_convergence_rows(
    merged_dir: Path,
    pair_rows: list[dict[str, object]],
    width: int,
    funcs: list[str],
) -> list[dict[str, object]]:
    pair_map = {
        (str(row["func"]), int(row["num_units"])): row
        for row in pair_rows
    }
    rows: list[dict[str, object]] = []
    for func in funcs:
        pair = pair_map.get((func, width))
        if pair is None:
            continue

        exp_history = load_history(merged_dir / f"{func}_{width}_exp" / "history.csv")
        none_history = load_history(merged_dir / f"{func}_{width}_none" / "history.csv")
        exp_epochs = exp_history["epoch"].astype(int)
        exp_mse = exp_history["mse"]
        none_epochs = none_history["epoch"].astype(int)
        none_mse = none_history["mse"]

        exp_best_mse = float(np.min(exp_mse))
        exp_best_epoch = int(exp_epochs[int(np.argmin(exp_mse))])
        none_best_mse = float(np.min(none_mse))
        none_best_epoch = int(none_epochs[int(np.argmin(none_mse))])

        reach_candidates = exp_epochs[exp_mse <= none_best_mse]
        exp_reaches_none_best_epoch = int(reach_candidates[0]) if len(reach_candidates) else None
        speedup = (
            none_best_epoch / exp_reaches_none_best_epoch
            if exp_reaches_none_best_epoch and exp_reaches_none_best_epoch > 0
            else math.nan
        )

        rows.append(
            {
                "func": func,
                "num_units": width,
                "exp_best_mse": exp_best_mse,
                "none_best_mse": none_best_mse,
                "none_over_exp_best_mse": none_best_mse / exp_best_mse,
                "exp_best_epoch": exp_best_epoch,
                "none_best_epoch": none_best_epoch,
                "exp_reaches_none_best_epoch": exp_reaches_none_best_epoch,
                "speedup_to_none_best": speedup,
                "ratio_exp_to_none": float(pair["ratio_exp_to_none"]),
            }
        )

    return rows


def build_scna_width_rows(
    merged_dir: Path,
    funcs: list[str],
    widths: list[int],
) -> list[dict[str, object]]:
    narrowest_width = min(widths)
    widest_width = max(widths)
    rows: list[dict[str, object]] = []
    for func in funcs:
        best_by_width: dict[int, float] = {}
        for width in widths:
            history = load_history(merged_dir / f"{func}_{width}_exp" / "history.csv")
            best_by_width[width] = float(np.min(history["mse"]))
        rows.append(
            {
                "func": func,
                "widths": list(widths),
                "best_mse_by_width": best_by_width,
                "narrowest_width": narrowest_width,
                "widest_width": widest_width,
                "gain_widest_over_narrowest": best_by_width[narrowest_width] / best_by_width[widest_width],
            }
        )
    return rows


def build_scna_width_csv_rows(rows: list[dict[str, object]]) -> tuple[list[str], list[dict[str, object]]]:
    if not rows:
        return [], []

    widths = list(rows[0]["widths"])
    fieldnames = ["func", "narrowest_width", "widest_width", "gain_widest_over_narrowest"] + [
        f"best_mse_w{width}" for width in widths
    ]
    csv_rows: list[dict[str, object]] = []
    for row in rows:
        csv_row = {
            "func": row["func"],
            "narrowest_width": row["narrowest_width"],
            "widest_width": row["widest_width"],
            "gain_widest_over_narrowest": row["gain_widest_over_narrowest"],
        }
        for width in widths:
            csv_row[f"best_mse_w{width}"] = row["best_mse_by_width"][width]
        csv_rows.append(csv_row)
    return fieldnames, csv_rows


def width_slug(widths: list[int]) -> str:
    return "_".join(str(width) for width in widths)


def epoch_slug(max_epochs: int) -> str:
    if max_epochs >= 1000 and max_epochs % 1000 == 0:
        return f"{max_epochs // 1000}k"
    return str(max_epochs)


def build_width_colors(widths: list[int]) -> dict[int, object]:
    preset = {
        4: "#7fd3c1",
        8: "#2f8fce",
        16: "#174ea6",
        32: "#0f2c73",
    }
    if all(width in preset for width in widths):
        return {width: preset[width] for width in widths}

    color_positions = np.linspace(0.18, 0.92, len(widths))
    cmap = plt.get_cmap("viridis")
    return {width: cmap(position) for width, position in zip(widths, color_positions)}


def plot_convergence_grid(
    merged_dir: Path,
    rows: list[dict[str, object]],
    output_path: Path,
    title: str,
    max_epochs: int,
    *,
    box_aspect: float,
) -> None:
    rows = sorted(rows, key=paper_panel_sort_key)
    total = len(rows)
    if total == 0:
        raise ValueError("No rows provided for convergence grid.")
    if total != GRID_EXPECTED_PANELS:
        raise ValueError(f"Expected {GRID_EXPECTED_PANELS} subplots for a 2x4 convergence grid, got {total}.")
    if box_aspect <= 0.0:
        raise ValueError("box_aspect must be positive.")
    with plt.rc_context(
        {
            "font.family": PLOT_FONT_FAMILY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.0,
            "axes.titlesize": 6.4,
            "axes.labelsize": GRID_AXIS_LABEL_SIZE,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": GRID_YTICK_LABEL_SIZE,
            "legend.fontsize": 6.0,
        }
    ):
        fig, axes = plt.subplots(
            GRID_ROWS,
            GRID_COLS,
            figsize=GRID_FIGSIZE,
            sharex=True,
            subplot_kw={"box_aspect": box_aspect},
        )
        fig.subplots_adjust(
            left=GRID_SUBPLOTS_LEFT,
            right=GRID_SUBPLOTS_RIGHT,
            bottom=GRID_SUBPLOTS_BOTTOM,
            top=GRID_SUBPLOTS_TOP,
            wspace=GRID_SUBPLOTS_WSPACE,
            hspace=GRID_SUBPLOTS_HSPACE,
        )
        axes = axes.flatten()

        colors = {"exp": "#1d4ed8", "none": "#94a3b8"}
        line_styles = {"mse": "-", "avg_loss": (0, (2.2, 1.6))}
        xticks = None
        if max_epochs > 0:
            xticks = [0, max_epochs / 2.0, max_epochs]

        metric_handles = [
            Line2D([0], [0], color=colors["exp"], linewidth=1.6, linestyle="-", label="SCNA MSE"),
            Line2D([0], [0], color=colors["exp"], linewidth=1.2, linestyle=line_styles["avg_loss"], label="SCNA loss"),
            Line2D([0], [0], color=colors["none"], linewidth=1.6, linestyle="-", label="Naive MSE"),
            Line2D([0], [0], color=colors["none"], linewidth=1.2, linestyle=line_styles["avg_loss"], label="Naive loss"),
        ]

        for index, (axis, row) in enumerate(zip(axes, rows)):
            row_idx = index // GRID_COLS
            col_idx = index % GRID_COLS
            func = str(row["func"])
            width = int(row["num_units"])
            histories = {
                "exp": load_history(merged_dir / f"{func}_{width}_exp" / "history.csv"),
                "none": load_history(merged_dir / f"{func}_{width}_none" / "history.csv"),
            }

            axis.set_facecolor("#fbfcfe")
            for spine in axis.spines.values():
                spine.set_color("#cbd5e1")
                spine.set_linewidth(0.75)

            panel_series: list[np.ndarray] = []
            for reparam, history in histories.items():
                epochs = history["epoch"]
                if max_epochs > 0:
                    mask = epochs <= max_epochs
                    if np.any(mask):
                        history = {key: value[mask] for key, value in history.items()}
                        epochs = history["epoch"]
                span = infer_smooth_span(len(epochs))
                mse = smooth_metric(history["mse"], span)
                avg_loss = smooth_metric(history["avg_loss"], span)
                mse_x, mse_y = downsample(epochs, mse, 1600)
                loss_x, loss_y = downsample(epochs, avg_loss, 1600)
                panel_series.extend([mse_y, loss_y])

                axis.plot(
                    mse_x,
                    mse_y,
                    color=colors[reparam],
                    linestyle=line_styles["mse"],
                    linewidth=1.55,
                    alpha=0.98,
                    solid_capstyle="round",
                    zorder=2,
                )
                axis.plot(
                    loss_x,
                    loss_y,
                    color=colors[reparam],
                    linestyle=line_styles["avg_loss"],
                    linewidth=1.15,
                    alpha=0.84,
                    solid_capstyle="round",
                    zorder=2,
                )

            axis.set_yscale("log")
            ymin, ymax = compute_axis_limits(panel_series)
            ymax *= ANNOTATION_HEADROOM.get(func, 1.8)
            axis.set_ylim(ymin, ymax)
            axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=4))
            axis.yaxis.set_major_formatter(LogFormatterMathtext())
            axis.minorticks_off()
            axis.grid(True, axis="y", which="major", linestyle=":", linewidth=0.4, alpha=0.32, color="#64748b")
            axis.grid(True, axis="x", which="major", linestyle=":", linewidth=0.28, alpha=0.14, color="#94a3b8")
            axis.tick_params(axis="both", length=1.8, width=0.5, pad=0.7)
            style_grid_y_ticks(axis, show_labels=col_idx == 0)
            if max_epochs > 0:
                axis.set_xlim(0, max_epochs)
                axis.set_xticks(xticks)
                axis.xaxis.set_major_formatter(FuncFormatter(format_epoch_tick))

            best_gain = float(row["none_over_exp_best_mse"])
            axis.text(
                0.97,
                0.97,
                f"{DISPLAY_NAMES.get(func, func)}\nMSE {best_gain:.1f}x",
                transform=axis.transAxes,
                ha="right",
                va="top",
                linespacing=1.15,
                fontsize=GRID_ANNOTATION_FONT_SIZE,
                fontweight="bold",
                color="#0f172a",
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "edgecolor": "#dbe4f0",
                    "linewidth": 0.5,
                    "alpha": 0.95,
                },
            )

            if col_idx == 0:
                axis.set_ylabel("Loss / MSE", fontweight="bold", fontsize=GRID_AXIS_LABEL_SIZE)
            if row_idx == GRID_ROWS - 1:
                axis.set_xlabel("Epoch", fontweight="bold", fontsize=GRID_AXIS_LABEL_SIZE)

        fig.legend(
            handles=metric_handles,
            loc="upper center",
            bbox_to_anchor=(0.53, GRID_LEGEND_Y),
            ncol=4,
            frameon=True,
            handlelength=GRID_LEGEND_HANDLE_LENGTH,
            handletextpad=GRID_LEGEND_HANDLE_TEXT_PAD,
            columnspacing=GRID_LEGEND_COLUMN_SPACING,
            borderpad=GRID_LEGEND_BORDERPAD,
            fancybox=True,
            edgecolor="#cbd5e1",
            facecolor="white",
            framealpha=1.0,
        )

        fig.savefig(output_path, dpi=320, bbox_inches="tight", pad_inches=GRID_SAVE_PAD_INCHES, facecolor="white")
        if output_path.suffix.lower() == ".png":
            fig.savefig(
                output_path.with_suffix(".pdf"),
                bbox_inches="tight",
                pad_inches=GRID_SAVE_PAD_INCHES,
                facecolor="white",
            )
        plt.close(fig)


def plot_scna_width_grid(
    merged_dir: Path,
    rows: list[dict[str, object]],
    output_path: Path,
    max_epochs: int,
    *,
    box_aspect: float,
    x_axis_limit: int | None = None,
) -> None:
    rows = sorted(rows, key=paper_panel_sort_key)
    if len(rows) != GRID_EXPECTED_PANELS:
        raise ValueError(f"Expected {GRID_EXPECTED_PANELS} subplots for a 2x4 grid, got {len(rows)}.")
    widths = list(rows[0]["widths"])
    if any(list(row["widths"]) != widths for row in rows):
        raise ValueError("All SCNA width rows must share the same width set.")
    if x_axis_limit is not None and x_axis_limit < max_epochs:
        raise ValueError("x_axis_limit must be greater than or equal to max_epochs.")
    if box_aspect <= 0.0:
        raise ValueError("box_aspect must be positive.")

    with plt.rc_context(
        {
            "font.family": PLOT_FONT_FAMILY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.0,
            "axes.titlesize": 6.4,
            "axes.labelsize": GRID_AXIS_LABEL_SIZE,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": GRID_YTICK_LABEL_SIZE,
            "legend.fontsize": 6.0,
        }
    ):
        fig, axes = plt.subplots(
            GRID_ROWS,
            GRID_COLS,
            figsize=GRID_FIGSIZE,
            sharex=True,
            subplot_kw={"box_aspect": box_aspect},
        )
        fig.subplots_adjust(
            left=GRID_SUBPLOTS_LEFT,
            right=GRID_SUBPLOTS_RIGHT,
            bottom=GRID_SUBPLOTS_BOTTOM,
            top=GRID_SUBPLOTS_TOP,
            wspace=GRID_SUBPLOTS_WSPACE,
            hspace=GRID_SUBPLOTS_HSPACE,
        )
        axes = axes.flatten()

        width_colors = build_width_colors(widths)
        line_styles = {"mse": "-", "avg_loss": (0, (2.2, 1.6))}
        xticks = [0, max_epochs / 2.0, max_epochs] if max_epochs > 0 else None
        data_limit = x_axis_limit if x_axis_limit is not None else max_epochs
        x_limit = data_limit

        metric_handles = [
            Line2D([0], [0], color=width_colors[width], linewidth=1.6, label=f"{width} units")
            for width in widths
        ] + [
            Line2D([0], [0], color="#0f172a", linewidth=1.35, linestyle="-", label="MSE"),
            Line2D([0], [0], color="#0f172a", linewidth=1.1, linestyle=line_styles["avg_loss"], label="loss"),
        ]

        for index, (axis, row) in enumerate(zip(axes, rows)):
            row_idx = index // GRID_COLS
            col_idx = index % GRID_COLS
            func = str(row["func"])
            axis.set_facecolor("#fbfcfe")
            for spine in axis.spines.values():
                spine.set_color("#cbd5e1")
                spine.set_linewidth(0.75)

            panel_series: list[np.ndarray] = []
            for width_index, width in enumerate(widths):
                history = load_history(merged_dir / f"{func}_{width}_exp" / "history.csv")
                epochs = history["epoch"]
                if data_limit > 0:
                    mask = epochs <= data_limit
                    if np.any(mask):
                        history = {key: value[mask] for key, value in history.items()}
                        epochs = history["epoch"]
                span = infer_smooth_span(len(epochs))
                mse = smooth_metric(history["mse"], span)
                avg_loss = smooth_metric(history["avg_loss"], span)
                mse_x, mse_y = downsample(epochs, mse, 1600)
                loss_x, loss_y = downsample(epochs, avg_loss, 1600)
                panel_series.extend([mse_y, loss_y])

                axis.plot(
                    mse_x,
                    mse_y,
                    color=width_colors[width],
                    linestyle=line_styles["mse"],
                    linewidth=1.4 if width_index + 1 < len(widths) else 1.6,
                    alpha=0.98,
                    solid_capstyle="round",
                    zorder=2,
                )
                axis.plot(
                    loss_x,
                    loss_y,
                    color=width_colors[width],
                    linestyle=line_styles["avg_loss"],
                    linewidth=1.05,
                    alpha=0.82,
                    solid_capstyle="round",
                    zorder=2,
                )

            axis.set_yscale("log")
            ymin, ymax = compute_axis_limits(panel_series)
            ymax *= ANNOTATION_HEADROOM.get(func, 1.8)
            axis.set_ylim(ymin, ymax)
            axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=4))
            axis.yaxis.set_major_formatter(LogFormatterMathtext())
            axis.minorticks_off()
            axis.grid(True, axis="y", which="major", linestyle=":", linewidth=0.4, alpha=0.32, color="#64748b")
            axis.grid(True, axis="x", which="major", linestyle=":", linewidth=0.28, alpha=0.14, color="#94a3b8")
            axis.tick_params(axis="both", length=1.8, width=0.5, pad=0.7)
            style_grid_y_ticks(axis, show_labels=col_idx == 0)
            if x_limit > 0:
                axis.set_xlim(0, x_limit)
            if max_epochs > 0:
                axis.set_xticks(xticks)
                axis.xaxis.set_major_formatter(FuncFormatter(format_epoch_tick))

            # gain_widest_over_narrowest = float(row["gain_widest_over_narrowest"])
            # widest_width = int(row["widest_width"])
            # narrowest_width = int(row["narrowest_width"])
            axis.text(
                0.97,
                0.97,
                f"{DISPLAY_NAMES.get(func, func)}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                linespacing=1.15,
                fontsize=GRID_ANNOTATION_FONT_SIZE,
                fontweight="bold",
                color="#0f172a",
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "edgecolor": "#dbe4f0",
                    "linewidth": 0.5,
                    "alpha": 0.95,
                },
            )

            if col_idx == 0:
                axis.set_ylabel("Loss / MSE", fontweight="bold", fontsize=GRID_AXIS_LABEL_SIZE)
            if row_idx == GRID_ROWS - 1:
                axis.set_xlabel("Epoch", fontweight="bold", fontsize=GRID_AXIS_LABEL_SIZE)

        fig.legend(
            handles=metric_handles,
            loc="upper center",
            bbox_to_anchor=(0.515, GRID_LEGEND_Y),
            ncol=min(len(metric_handles), 6),
            frameon=True,
            handlelength=GRID_LEGEND_HANDLE_LENGTH,
            handletextpad=GRID_LEGEND_HANDLE_TEXT_PAD,
            columnspacing=GRID_LEGEND_COLUMN_SPACING,
            borderpad=GRID_LEGEND_BORDERPAD,
            fancybox=True,
            edgecolor="#cbd5e1",
            facecolor="white",
            framealpha=1.0,
        )

        fig.savefig(output_path, dpi=320, bbox_inches="tight", pad_inches=GRID_SAVE_PAD_INCHES, facecolor="white")
        if output_path.suffix.lower() == ".png":
            fig.savefig(
                output_path.with_suffix(".pdf"),
                bbox_inches="tight",
                pad_inches=GRID_SAVE_PAD_INCHES,
                facecolor="white",
            )
        plt.close(fig)


def write_report(
    path: Path,
    *,
    baseline_dir: Path,
    corrected_dir: Path,
    merged_dir: Path,
    replaced_runs: list[str],
    added_runs: list[str],
    pair_rows: list[dict[str, object]],
    convergence_rows: list[dict[str, object]],
    convergence_width: int,
    convergence_max_epochs: int,
) -> None:
    exp_better = sum(1 for row in pair_rows if str(row["better"]) == "exp")
    none_better = sum(1 for row in pair_rows if str(row["better"]) == "none")
    tie_count = len(pair_rows) - exp_better - none_better

    width_stats = []
    for width in sorted({int(row["num_units"]) for row in pair_rows}):
        width_rows = [row for row in pair_rows if int(row["num_units"]) == width]
        ratios = [float(row["ratio_exp_to_none"]) for row in width_rows]
        width_stats.append(
            {
                "num_units": width,
                "pairs": len(width_rows),
                "exp_better": sum(float(row["ratio_exp_to_none"]) < 1.0 for row in width_rows),
                "none_better": sum(float(row["ratio_exp_to_none"]) > 1.0 for row in width_rows),
                "geom_exp_to_none": geometric_mean(ratios),
                "geom_none_to_exp": geometric_mean([1.0 / ratio for ratio in ratios]),
            }
        )

    convergence_speedups = [
        float(row["speedup_to_none_best"])
        for row in convergence_rows
        if math.isfinite(float(row["speedup_to_none_best"]))
    ]
    convergence_wins = sum(float(row["speedup_to_none_best"]) > 1.0 for row in convergence_rows)
    convergence_exp_wins = sum(float(row["ratio_exp_to_none"]) < 1.0 for row in convergence_rows)
    convergence_ratios = [float(row["ratio_exp_to_none"]) for row in convergence_rows]
    convergence_geom_ratio = geometric_mean(convergence_ratios)
    convergence_geom_speedup = geometric_mean(convergence_speedups)
    none_wins = [row for row in pair_rows if str(row["better"]) == "none"]
    available_widths = sorted({int(row["num_units"]) for row in pair_rows})
    replaced_funcs = ", ".join(sorted({name.rsplit("_", 2)[0] for name in replaced_runs})) or "none"
    added_funcs = ", ".join(sorted({name.rsplit("_", 2)[0] for name in added_runs})) or "none"

    lines = [
        f"# Merged Sweep Analysis: `{merged_dir.name}`",
        "",
        "## Merge inputs",
        "",
        f"- Baseline sweep: `{baseline_dir}`",
        f"- Overlay sweep: `{corrected_dir}`",
        f"- Merged sweep dir: `{merged_dir}`",
        f"- Replaced runs from overlay sweep: {len(replaced_runs)}",
        f"- Added runs from overlay sweep: {len(added_runs)}",
        f"- Replaced functions: {replaced_funcs}",
        f"- Added functions: {added_funcs}",
        "",
        "## Overall result",
        "",
        f"- Paired comparisons (`exp` vs `none`): {len(pair_rows)}",
        f"- Available widths: {', '.join(str(width) for width in available_widths)}",
        f"- `exp` better: {exp_better}",
        f"- `none` better: {none_better}",
        f"- Ties: {tie_count}",
        f"- Geometric-mean `exp/none` best-MSE ratio across all pairs: {geometric_mean([float(row['ratio_exp_to_none']) for row in pair_rows]):.4f}",
        "",
        "## By width",
        "",
        "| Units | Pairs | exp better | none better | Geometric mean exp/none | Geometric mean none/exp |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in width_stats:
        lines.append(
            "| "
            f"{row['num_units']} | {row['pairs']} | {row['exp_better']} | {row['none_better']} | "
            f"{row['geom_exp_to_none']:.4f} | {row['geom_none_to_exp']:.2f} |"
        )

    lines.extend(
        [
            "",
            f"## 2x4 convergence slice ({convergence_width} units)",
            "",
            f"- Final MSE: `exp` wins {sum(float(row['ratio_exp_to_none']) < 1.0 for row in convergence_rows)}/{len(convergence_rows)} functions.",
            f"- Speed to reach the final `none` best MSE: `exp` wins {convergence_wins}/{len(convergence_rows)} functions.",
            f"- Geometric-mean final-MSE gain (`none/exp`): {1.0 / convergence_geom_ratio:.2f}x",
            f"- Geometric-mean speedup to the `none` best-MSE target: {convergence_geom_speedup:.2f}x",
            "",
            "| Function | exp best MSE | none best MSE | none/exp gain | exp <= none-best epoch | none best epoch | Speedup |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in convergence_rows:
        reach = row["exp_reaches_none_best_epoch"]
        speedup = float(row["speedup_to_none_best"])
        lines.append(
            "| "
            f"{row['func']} | "
            f"{float(row['exp_best_mse']):.6e} | "
            f"{float(row['none_best_mse']):.6e} | "
            f"{float(row['none_over_exp_best_mse']):.2f} | "
            f"{reach if reach is not None else 'NA'} | "
            f"{int(row['none_best_epoch'])} | "
            f"{speedup:.2f} |"
        )

    slow_cases = [row for row in convergence_rows if float(row["speedup_to_none_best"]) < 1.0]
    if none_wins:
        none_heading = "## Remaining none-favored case" if len(none_wins) == 1 else "## Remaining none-favored cases"
        lines.extend(
            [
                "",
                none_heading,
                "",
                "| Function | Units | exp best MSE | none best MSE | exp/none |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in none_wins:
            lines.append(
                "| "
                f"{row['func']} | {row['num_units']} | "
                f"{float(row['exp_best_mse']):.6e} | "
                f"{float(row['none_best_mse']):.6e} | "
                f"{float(row['ratio_exp_to_none']):.6f} |"
            )

    if slow_cases:
        if convergence_exp_wins == len(convergence_rows):
            convergence_note = (
                f"- The convergence figure uses the {convergence_width}-unit slice because it is the cleanest "
                "all-function demonstration: every displayed function ends with a lower best MSE under `exp`."
            )
        else:
            convergence_note = (
                f"- The convergence figure uses the requested {convergence_width}-unit slice; "
                f"`exp` wins {convergence_exp_wins}/{len(convergence_rows)} displayed functions on final best MSE."
            )
        lines.extend(
            [
                "",
                "## Notes",
                "",
                convergence_note,
                "- The 2x4 panel order omits `exp2`; it remains in the merged sweep summaries but is not "
                "shown in the all-in-one grids.",
                f"- The plotted window is limited to the first {convergence_max_epochs:,} epochs to emphasize convergence behavior rather than the long flat tail.",
                "- The slower crossover cases in this slice are those where `exp` keeps descending to a much lower final floor after `none` plateaus early.",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if any(max_epochs <= 0 for max_epochs in args.full_width_scna_max_epochs):
        raise ValueError("--full-width-scna-max-epochs values must be positive.")
    if args.subplot_box_aspect <= 0.0:
        raise ValueError("--subplot-box-aspect must be positive.")
    full_width_scna_x_axis_limit_overrides = parse_epoch_limit_overrides(
        args.full_width_scna_x_axis_limit_overrides
    )
    register_libertinus_sans()
    baseline_dir = args.baseline_dir.resolve()
    corrected_dir = args.corrected_dir.resolve()
    output_dir = args.output_dir.resolve()
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    source_rows, replaced_runs, added_runs = merge_runs(baseline_dir, corrected_dir, output_dir)
    # write_csv(analysis_dir / "run_sources.csv", ["run_name", "source_group", "source_dir"], source_rows)
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(
            {
                "added_runs": added_runs,
                "baseline_dir": str(baseline_dir),
                "corrected_dir": str(corrected_dir),
                "merged_dir": str(output_dir),
                "replaced_runs": replaced_runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_runs(output_dir)
    pair_rows = sorted(build_pair_rows(rows), key=pair_sort_key)
    title_prefix = args.title or output_dir.name
    available_widths = sorted({int(row["num_units"]) for row in pair_rows})
    if args.convergence_width not in available_widths:
        raise ValueError(
            f"Requested convergence width {args.convergence_width} is unavailable; found widths {available_widths}."
        )

    # write_summary_csv(analysis_dir / "summary.csv", rows)
    # write_pair_csv(analysis_dir / "paired_summary.csv", pair_rows)
    # plot_best_mse(rows, analysis_dir / "best_mse_by_function.png", title_prefix)
    # plot_ratio(pair_rows, analysis_dir / "exp_vs_none_ratio.png", title_prefix)

    convergence_funcs = pick_convergence_funcs(
        pair_rows,
        args.convergence_width,
        preferred_order=PAPER_PANEL_ORDER,
        omitted_funcs=PAPER_OMITTED_FUNCS,
        max_funcs=GRID_EXPECTED_PANELS,
    )
    convergence_rows = build_convergence_rows(output_dir, pair_rows, args.convergence_width, convergence_funcs)
    convergence_fieldnames = [
        "func",
        "num_units",
        "exp_best_mse",
        "none_best_mse",
        "none_over_exp_best_mse",
        "exp_best_epoch",
        "none_best_epoch",
        "exp_reaches_none_best_epoch",
        "speedup_to_none_best",
        "ratio_exp_to_none",
    ]
    ordered_convergence_widths = [args.convergence_width] + [
        width for width in available_widths if width != args.convergence_width
    ]
    for width in ordered_convergence_widths:
        width_funcs = pick_convergence_funcs(
            pair_rows,
            width,
            preferred_order=PAPER_PANEL_ORDER,
            omitted_funcs=PAPER_OMITTED_FUNCS,
            max_funcs=GRID_EXPECTED_PANELS,
        )
        width_rows = build_convergence_rows(output_dir, pair_rows, width, width_funcs)
        write_csv(
            analysis_dir / f"convergence_width{width}.csv",
            convergence_fieldnames,
            width_rows,
        )
        plot_convergence_grid(
            output_dir,
            width_rows,
            analysis_dir / f"all_in_one_convergence_width{width}.png",
            f"{title_prefix} width {width}",
            args.convergence_max_epochs,
            box_aspect=args.subplot_box_aspect,
        )

    legacy_widths = [4, 8, 16]
    scna_width_sets: list[tuple[list[int], Path, Path, list[int], dict[int, int]]] = []
    if all(width in available_widths for width in legacy_widths):
        scna_width_sets.append(
            (
                legacy_widths,
                analysis_dir / "scna_width_comparison.csv",
                analysis_dir / "all_in_one_scna_widths_4_8_16.png",
                [args.convergence_max_epochs],
                {},
            )
        )
    if available_widths != legacy_widths:
        full_width_slug = width_slug(available_widths)
        scna_width_sets.append(
            (
                available_widths,
                analysis_dir / f"scna_width_comparison_{full_width_slug}.csv",
                analysis_dir / f"all_in_one_scna_widths_{full_width_slug}.png",
                args.full_width_scna_max_epochs,
                full_width_scna_x_axis_limit_overrides,
            )
        )

    for widths, csv_path, figure_path, figure_max_epochs, figure_x_axis_limit_overrides in scna_width_sets:
        scna_width_rows = build_scna_width_rows(output_dir, convergence_funcs, widths)
        fieldnames, csv_rows = build_scna_width_csv_rows(scna_width_rows)
        write_csv(csv_path, fieldnames, csv_rows)
        for max_epochs in figure_max_epochs:
            target_path = figure_path
            if len(figure_max_epochs) > 1:
                target_path = figure_path.with_name(
                    f"{figure_path.stem}_{epoch_slug(max_epochs)}{figure_path.suffix}"
                )
                plot_scna_width_grid(
                    output_dir,
                    scna_width_rows,
                    target_path,
                    max_epochs,
                    box_aspect=args.subplot_box_aspect,
                    x_axis_limit=figure_x_axis_limit_overrides.get(max_epochs),
                )

    write_report(
        analysis_dir / "report.md",
        baseline_dir=baseline_dir,
        corrected_dir=corrected_dir,
        merged_dir=output_dir,
        replaced_runs=replaced_runs,
        added_runs=added_runs,
        pair_rows=pair_rows,
        convergence_rows=convergence_rows,
        convergence_width=args.convergence_width,
        convergence_max_epochs=args.convergence_max_epochs,
    )

    print(
        json.dumps(
            {
                "baseline_dir": str(baseline_dir),
                "corrected_dir": str(corrected_dir),
                "merged_dir": str(output_dir),
                "replaced_runs": len(replaced_runs),
                "paired_runs": len(pair_rows),
                "convergence_width": args.convergence_width,
                "convergence_max_epochs": args.convergence_max_epochs,
                "convergence_funcs": convergence_funcs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
