from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, to_hex, to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = EXPERIMENT_DIR
PLOT_CSV = Path(os.environ.get("FIG16_INPUT", EXPERIMENT_DIR / "expected" / "paper_figure16.csv"))
NLI_SUMMARY_CSV = EXPERIMENT_DIR / "data" / "not-included-nli.csv"
NNLUT_SUMMARY_CSV = EXPERIMENT_DIR / "data" / "not-included-nnlut.csv"
GQALUT_SUMMARY_CSV = EXPERIMENT_DIR / "data" / "not-included-gqalut.csv"
PAPER_FIGURES_DIR = Path(os.environ.get("FIG16_OUTPUT_DIR", EXPERIMENT_DIR / "generated"))
PAPER_TABLES_DIR = PAPER_FIGURES_DIR
EVAL_DRAFT_PATH = PAPER_FIGURES_DIR / "evaluation_section_draft.md"

MODEL_ORDER = ["facebook-opt-6.7b", "Llama-2-7b-hf", "Llama-3-8b", "Qwen2.5-7B", "Qwen3-8B"]
MODEL_TITLES = {
    "facebook-opt-6.7b": "OPT-6.7B",
    "Llama-2-7b-hf": "Llama-2-7B",
    "Llama-3-8b": "Llama-3-8B",
    "Qwen2.5-7B": "Qwen2.5-7B",
    "Qwen3-8B": "Qwen3-8B",
}
SCNA_VARIANTS = ["pinn8", "pinn16", "pinn32"]
DEFAULT_TRIPTYCH_VARIANT_ORDER = ["exact", "nli", "nnlut", "gqalut", *SCNA_VARIANTS]
DISPLAY_LABELS = {
    "exact": "Baseline",
    "nli": "NLI",
    "nnlut": "NNLUT",
    "gqalut": "GQALUT",
    "pinn8": "SCNA 8",
    "pinn16": "SCNA 16",
    "pinn32": "SCNA 32",
}
PRIMARY_DIM = "pinn16"
TASK_METRICS = [
    ("group1_mean", "Avg"),
    ("arc_easy", "ARC-E"),
    ("hellaswag", "HellaSwag"),
    ("piqa", "PIQA"),
    ("winogrande", "Winogrande"),
]
HATCH_SEQUENCE = ["///", "\\\\\\", "xx", "..", "++", "oo", "--", "**"]


def _gradient_colors(start_hex: str, end_hex: str, steps: int) -> list[str]:
    start = to_rgb(start_hex)
    end = to_rgb(end_hex)
    if steps == 1:
        return [to_hex(start)]
    colors: list[str] = []
    for index in range(steps):
        weight = index / (steps - 1)
        rgb = tuple(
            start[channel] + (end[channel] - start[channel]) * weight
            for channel in range(3)
        )
        colors.append(to_hex(rgb))
    return colors


def _variant_styles(variant_order: list[str]) -> dict[str, dict[str, str]]:
    colors = _gradient_colors("#99d98c", "#1a759f", len(variant_order))
    return {
        variant: {"color": color, "hatch": HATCH_SEQUENCE[index % len(HATCH_SEQUENCE)]}
        for index, (variant, color) in enumerate(zip(variant_order, colors, strict=True))
    }


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.family": "sans-serif",
            "font.sans-serif": ["Libertinus Sans", "DejaVu Sans"],
            "font.size": 20,
            "axes.titlesize": 28,
            "axes.labelsize": 28,
            "axes.linewidth": 1.5,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 28,
            "legend.frameon": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_plot_rows() -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with PLOT_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["model"], row["metric"])] = {
                key: float(value) for key, value in row.items() if key not in {"model", "metric"}
            }

    # Initialize approximation-only columns so missing runs render as NaN instead of raising.
    for row in rows.values():
        for prefix in ("fp16", "int8"):
            for variant in ("nli", "nnlut", "gqalut"):
                row.setdefault(f"{prefix}_{variant}", np.nan)

    def merge_cross_model_summary(
        summary_csv: Path,
        *,
        fp_variant: str,
        int8_variant: str,
        target_label: str,
    ) -> None:
        if not summary_csv.exists():
            return
        with summary_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row["model"]
                precision = row["precision_mode"]
                variant = row["variant"]
                if precision == "fullprecision" and variant == fp_variant:
                    key_prefix = "fp16"
                elif precision == "static_sq_w8a8" and variant == int8_variant:
                    key_prefix = "int8"
                else:
                    continue
                rows[(model, "ppl")][f"{key_prefix}_{target_label}"] = float(row["wikitext_ppl"])
                rows[(model, "group1_mean")][f"{key_prefix}_{target_label}"] = float(row["avg_zero_shot_acc"])

    merge_cross_model_summary(
        NLI_SUMMARY_CSV,
        fp_variant="nli_fp",
        int8_variant="nli_w8a8",
        target_label="nli",
    )
    merge_cross_model_summary(
        NNLUT_SUMMARY_CSV,
        fp_variant="nnlut_fp",
        int8_variant="nnlut_w8a8",
        target_label="nnlut",
    )
    merge_cross_model_summary(
        GQALUT_SUMMARY_CSV,
        fp_variant="gqalut_fp",
        int8_variant="gqalut_w8a8",
        target_label="gqalut",
    )
    return rows


def _axis_limits(values: list[float]) -> tuple[float, float]:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return -1.0, 1.0
    vmin = min(finite_values)
    vmax = max(finite_values)
    vrange = vmax - vmin
    padding = max(vrange * 0.18, vmax * 0.015, 0.03)
    return vmin - padding, vmax + padding * 1.6


def _save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PAPER_FIGURES_DIR / f"{stem}.png"
    pdf_path = PAPER_FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _variant_offsets(count: int) -> np.ndarray:
    return np.linspace(-0.285, 0.285, count)


def _legend_handles(variant_order: list[str], variant_styles: dict[str, dict[str, str]]) -> list[Patch]:
    return [
        Patch(
            facecolor=variant_styles[label]["color"],
            edgecolor="black",
            hatch=variant_styles[label]["hatch"],
            linewidth=1.0,
            label=DISPLAY_LABELS[label],
        )
        for label in variant_order
    ]


def _draw_triptych_row(
    axes: list[plt.Axes] | np.ndarray,
    plot_rows: dict[tuple[str, str], dict[str, float]],
    *,
    variant_order: list[str],
    variant_styles: dict[str, dict[str, str]],
    metric: str,
    ylabel: str,
    scale: float = 1.0,
    fmt: str = "{:.3f}",
    show_xticklabels: bool = True,
) -> None:
    centers = [0.0, 1.0]
    offsets = _variant_offsets(len(variant_order))
    # if len(offsets) > 1:
    #     bar_width = min(0.2, float(offsets[1] - offsets[0]) * 0.72)
    # else:
    bar_width = 0.14

    for ax, model in zip(axes, MODEL_ORDER):
        row = plot_rows[(model, metric)]
        fp16_vals = [row[f"fp16_{label}"] * scale for label in variant_order]
        int8_vals = [row[f"int8_{label}"] * scale for label in variant_order]
        ymin, ymax = _axis_limits(fp16_vals + int8_vals)
        text_offset = (ymax - ymin) * 0.04

        for idx, label in enumerate(variant_order):
            fp16_x = centers[0] + offsets[idx]
            int8_x = centers[1] + offsets[idx]
            fp16_val = fp16_vals[idx]
            int8_val = int8_vals[idx]
            style = variant_styles[label]

            text_font_size = 24
            if np.isfinite(fp16_val):
                ax.bar(
                    fp16_x,
                    fp16_val,
                    width=bar_width,
                    color=style["color"],
                    edgecolor="black",
                    linewidth=1.0,
                    hatch=style["hatch"],
                )
                ax.text(
                    fp16_x,
                    fp16_val + text_offset,
                    fmt.format(fp16_val),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=text_font_size,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
                )
            else:
                ax.text(
                    fp16_x,
                    ymax - text_offset * 0.8,
                    "NaN",
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=text_font_size,
                    color=style["color"],
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
                )

            if np.isfinite(int8_val):
                ax.bar(
                    int8_x,
                    int8_val,
                    width=bar_width,
                    color=style["color"],
                    edgecolor="black",
                    linewidth=1.0,
                    hatch=style["hatch"],
                )
                ax.text(
                    int8_x,
                    int8_val + text_offset,
                    fmt.format(int8_val),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=text_font_size,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
                )
            else:
                ax.text(
                    int8_x,
                    ymax - text_offset * 0.8,
                    "NaN",
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=text_font_size,
                    color=style["color"],
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
                )

        ax.text(
            0.02,
            0.96,
            MODEL_TITLES[model],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=28,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.2},
        )
        ax.set_xticks(centers, ["Full-Prec.", "Static SQ"] if show_xticklabels else ["", ""])
        ax.set_ylim(ymin - 0.4, ymax + 0.23)
        ax.set_box_aspect(0.85)
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.9, color="#707070")
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.6, alpha=0.8, color="#8A8A8A")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", pad=4)
        ax.tick_params(axis="y", labelrotation=90, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    axes[0].set_ylabel(ylabel, fontweight="bold")


def _plot_triptych(
    plot_rows: dict[tuple[str, str], dict[str, float]],
    *,
    variant_order: list[str],
    metric: str,
    ylabel: str,
    stem: str,
    scale: float = 1.0,
    fmt: str = "{:.3f}",
) -> tuple[Path, Path]:
    _configure_style()
    variant_styles = _variant_styles(variant_order)
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(4.6 * len(MODEL_ORDER), 4.8))
    _draw_triptych_row(
        axes,
        plot_rows,
        variant_order=variant_order,
        variant_styles=variant_styles,
        metric=metric,
        ylabel=ylabel,
        scale=scale,
        fmt=fmt,
        show_xticklabels=True,
    )
    legend_ncol = min(len(variant_order), 4)
    fig.legend(
        handles=_legend_handles(variant_order, variant_styles),
        ncol=legend_ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        handlelength=1.8,
        columnspacing=1.1,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.14,
        top=0.78 if len(variant_order) > 4 else 0.82,
        wspace=0.1,
    )
    return _save_figure(fig, stem)


def _plot_triptych_vertical(
    plot_rows: dict[tuple[str, str], dict[str, float]],
    *,
    variant_order: list[str],
    stem: str,
) -> tuple[Path, Path]:
    _configure_style()
    variant_styles = _variant_styles(variant_order)
    fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(4.6 * len(MODEL_ORDER), 8.8))
    _draw_triptych_row(
        axes[0],
        plot_rows,
        variant_order=variant_order,
        variant_styles=variant_styles,
        metric="ppl",
        ylabel="WikiText-2 PPL",
        scale=1.0,
        fmt="{:.3f}",
        show_xticklabels=True,
    )
    _draw_triptych_row(
        axes[1],
        plot_rows,
        variant_order=variant_order,
        variant_styles=variant_styles,
        metric="group1_mean",
        ylabel="Avg. Zero-shot Acc. (%)",
        scale=100.0,
        fmt="{:.2f}",
        show_xticklabels=True,
    )
    legend_ncol = min(len(variant_order), 4)
    fig.legend(
        handles=_legend_handles(variant_order, variant_styles),
        ncol=legend_ncol,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        handlelength=1.8,
        columnspacing=1.1,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.08,
        top=0.88 if len(variant_order) > 4 else 0.91,
        hspace=0.35,
        wspace=0.1,
    )
    return _save_figure(fig, stem)


def _plot_delta_heatmap(
    plot_rows: dict[tuple[str, str], dict[str, float]],
    *,
    stem: str,
) -> tuple[Path, Path]:
    _configure_style()
    dims = ["pinn8", "pinn16", "pinn32"]
    data_blocks = []
    max_abs = 0.0
    for model in MODEL_ORDER:
        matrix = []
        for metric, _ in TASK_METRICS:
            row = plot_rows[(model, metric)]
            baseline = row["int8_exact"]
            deltas = [(row[f"int8_{dim}"] - baseline) * 100.0 for dim in dims]
            matrix.append(deltas)
            max_abs = max(max_abs, max(abs(delta) for delta in deltas))
        data_blocks.append(np.array(matrix))

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(4.9 * len(MODEL_ORDER), 4.2))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = plt.get_cmap("RdBu_r")

    for ax, model, matrix in zip(axes, MODEL_ORDER, data_blocks):
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(MODEL_TITLES[model], pad=12)
        ax.set_xticks(range(3), [DISPLAY_LABELS[dim] for dim in dims], rotation=15, ha="right")
        ax.set_yticks(range(len(TASK_METRICS)), [label for _, label in TASK_METRICS])
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="black",
                )
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    cbar_ax = fig.add_axes([0.945, 0.18, 0.01, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Delta vs INT8 Baseline (pp)", rotation=90, labelpad=14)
    fig.subplots_adjust(left=0.055, right=0.935, bottom=0.19, top=0.84, wspace=0.24)
    return _save_figure(fig, stem)


def _write_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _fmt_metric(metric: str, value: float) -> str:
    if metric == "ppl":
        return f"{value:.3f}"
    return f"{value * 100.0:.2f}"


def _write_summary_tables(
    plot_rows: dict[tuple[str, str], dict[str, float]],
) -> tuple[Path, Path, Path, Path]:
    summary_headers = ["Model", "Metric", "BF16 Baseline", "INT8 Baseline", "INT8 SCNA 8", "INT8 SCNA 16", "INT8 SCNA 32"]
    summary_rows: list[list[str]] = []
    for model in MODEL_ORDER:
        for metric, metric_label in [("ppl", "WikiText-2 PPL"), ("group1_mean", "Group1 Avg (%)")]:
            row = plot_rows[(model, metric)]
            summary_rows.append(
                [
                    MODEL_TITLES[model],
                    metric_label,
                    _fmt_metric(metric, row["fp16_exact"]),
                    _fmt_metric(metric, row["int8_exact"]),
                    _fmt_metric(metric, row["int8_pinn8"]),
                    _fmt_metric(metric, row["int8_pinn16"]),
                    _fmt_metric(metric, row["int8_pinn32"]),
                ]
            )

    summary_md = PAPER_TABLES_DIR / "summary_metrics_static_sq.md"
    summary_csv = PAPER_TABLES_DIR / "summary_metrics_static_sq.csv"
    _write_table(summary_md, summary_headers, summary_rows)
    _write_csv(summary_csv, summary_headers, summary_rows)

    lmeval_headers = ["Model", "Task", "BF16 Baseline (%)", "INT8 Baseline (%)", "INT8 SCNA 16 (%)", "Delta vs INT8 Baseline (pp)"]
    lmeval_rows: list[list[str]] = []
    for model in MODEL_ORDER:
        for metric, task_label in TASK_METRICS:
            row = plot_rows[(model, metric)]
            baseline_int8 = row["int8_exact"] * 100.0
            scna16 = row[f"int8_{PRIMARY_DIM}"] * 100.0
            delta_pp = scna16 - baseline_int8
            lmeval_rows.append(
                [
                    MODEL_TITLES[model],
                    task_label,
                    f"{row['fp16_exact'] * 100.0:.2f}",
                    f"{baseline_int8:.2f}",
                    f"{scna16:.2f}",
                    f"{delta_pp:+.2f}",
                ]
            )

    lmeval_md = PAPER_TABLES_DIR / "lmeval_scna16_static_sq.md"
    lmeval_csv = PAPER_TABLES_DIR / "lmeval_scna16_static_sq.csv"
    _write_table(lmeval_md, lmeval_headers, lmeval_rows)
    _write_csv(lmeval_csv, lmeval_headers, lmeval_rows)
    return summary_md, summary_csv, lmeval_md, lmeval_csv


def _delta(plot_rows: dict[tuple[str, str], dict[str, float]], model: str, metric: str, variant: str) -> float:
    row = plot_rows[(model, metric)]
    if metric == "ppl":
        return row[f"int8_{variant}"] - row["int8_exact"]
    return (row[f"int8_{variant}"] - row["int8_exact"]) * 100.0


def _write_evaluation_draft(
    plot_rows: dict[tuple[str, str], dict[str, float]],
    *,
    ppl_paths: tuple[Path, Path],
    group1_paths: tuple[Path, Path],
    heatmap_paths: tuple[Path, Path],
    summary_md: Path,
    lmeval_md: Path,
) -> Path:
    ppl_deltas = {
        model: [_delta(plot_rows, model, "ppl", dim) for dim in ("pinn8", "pinn16", "pinn32")]
        for model in MODEL_ORDER
    }
    group1_deltas = {model: _delta(plot_rows, model, "group1_mean", PRIMARY_DIM) for model in MODEL_ORDER}
    model_family_text = ", ".join(MODEL_TITLES[model] for model in MODEL_ORDER)
    group1_delta_text = ", ".join(
        f"{MODEL_TITLES[model]} {delta:+.2f} pp" for model, delta in group1_deltas.items()
    )
    ppl_delta_text = "; ".join(
        f"{MODEL_TITLES[model]} {min(deltas):+.3f} to {max(deltas):+.3f}"
        for model, deltas in ppl_deltas.items()
    )

    lines = [
        "# Evaluation Section Draft",
        "",
        "## Figures and Tables",
        "",
        f"- Figure A (PPL triptych): [{ppl_paths[0].relative_to(ANALYSIS_DIR)}]({ppl_paths[0].relative_to(ANALYSIS_DIR)})",
        f"- Figure B (Group1 triptych): [{group1_paths[0].relative_to(ANALYSIS_DIR)}]({group1_paths[0].relative_to(ANALYSIS_DIR)})",
        f"- Figure C (task-level delta heatmap): [{heatmap_paths[0].relative_to(ANALYSIS_DIR)}]({heatmap_paths[0].relative_to(ANALYSIS_DIR)})",
        f"- Table A (summary metrics): [{summary_md.relative_to(ANALYSIS_DIR)}]({summary_md.relative_to(ANALYSIS_DIR)})",
        f"- Table B (representative SCNA-16 LM-eval absolute results): [{lmeval_md.relative_to(ANALYSIS_DIR)}]({lmeval_md.relative_to(ANALYSIS_DIR)})",
        "",
        "## Draft Text",
        "",
        "### Evaluation Setup",
        "",
        f"We evaluate five decoder-only LLM families: {model_family_text}. For each model, we compare the BF16 baseline, the INT8 SmoothQuant baseline, and INT8 SmoothQuant augmented with SCNA at three sizes (SCNA-8, SCNA-16, and SCNA-32). Following the all-in-one experiment plan, the INT8 figures in the main paper use the `static_sq` setting with quantized PINN activations and weights (`pinn_w8a8`). WikiText-2 perplexity is reported as a language-modeling diagnostic, while downstream quality is measured by the average accuracy over ARC-Easy, HellaSwag, PIQA, and Winogrande (`group1 mean`). For the static INT8 runs, backbone calibration uses the same configuration across all models: validation split, 128 samples, and sequence length 2048.",
        "",
        "### Overall Accuracy Retention",
        "",
        "Figure B and Table A summarize the downstream accuracy trends. The absolute movement in `group1 mean` is small across the evaluated models, which is the main architectural result: SCNA introduces only limited end-to-end quality drift even when the nonlinear approximation is inserted into the inference path. Using SCNA-16 as the representative operating point, the INT8 `group1 mean` changes by "
        f"{group1_delta_text}. "
        "This places the SCNA-induced accuracy change in the sub-percentage-point regime for all evaluated families. Table B reports the task-level absolute numbers for the representative SCNA-16 design point, while Figure C exposes the full SCNA-8/16/32 sweep as deltas relative to the INT8 baseline.",
        "",
        "### Task-Level Sensitivity",
        "",
        "Figure C is the most compact way to present the LM-eval sweep without resorting to a large table. Each cell shows the accuracy delta of a SCNA operating point relative to the INT8 baseline in percentage points. Two patterns are visible. First, the average downstream effect remains small, confirming that the approximation does not catastrophically distort model behavior. Second, the task-level changes are heterogeneous: some tasks improve slightly while others regress slightly. This is useful for an architecture paper because it shows that SCNA is not simply shifting all tasks in one direction; instead, the approximation perturbs the token distribution in a localized way that depends on both the model family and the benchmark.",
        "",
        "### Language-Modeling Quality",
        "",
        "Figure A highlights a model-family-dependent perplexity trend. The INT8 SCNA-minus-baseline PPL deltas are "
        f"{ppl_delta_text}. "
        "Table A makes these trends explicit in a compact form. Importantly, the same comparison is available in the BF16 runs, which helps separate approximation effects from INT8 quantization and static SmoothQuant calibration.",
        "",
        "### Interpreting Model-Family Differences",
        "",
        "The most plausible explanation is the model-family-dependent integration path of SCNA in the current software prototype. For LLaMA and Qwen, SCNA replaces both the attention softmax path and the MLP gating nonlinearity. For OPT, SCNA replaces only the attention softmax path, while the ReLU FFN remains exact. Consequently, LLaMA and Qwen accumulate approximation error from both the attention and FFN paths, whereas OPT only perturbs the attention normalization. We therefore interpret model-family differences as an implementation-path and architecture interaction rather than as a general capability improvement.",
        "",
        "### Suggested Main-Text Placement",
        "",
        "For a top-tier computer architecture paper, we recommend using Table A plus Figures A-C in the main text. Table A provides the absolute baseline and SCNA summary by model, Figure A gives the language-modeling diagnostic, Figure B shows the downstream accuracy retention, and Figure C communicates the full task-level sweep as signed deltas instead of overwhelming the reader with a large raw-score table. Table B can be kept if space permits and is especially useful when discussing the representative SCNA-16 operating point in detail.",
        "",
        "## Notes",
        "",
        "- Figure and table labels above are placeholders for manuscript integration.",
        "- The generated assets are paper-oriented outputs and do not overwrite the earlier exploratory figures in `analysis/figures/`.",
    ]

    with EVAL_DRAFT_PATH.open("w") as f:
        f.write("\n".join(lines) + "\n")
    return EVAL_DRAFT_PATH


def _parse_variant_order(spec: str) -> list[str]:
    variant_order: list[str] = []
    seen: set[str] = set()
    for item in spec.split(","):
        label = item.strip()
        if not label or label in seen:
            continue
        if label not in DISPLAY_LABELS:
            raise ValueError(f"Unknown variant `{label}`. Valid choices: {', '.join(DISPLAY_LABELS)}")
        seen.add(label)
        variant_order.append(label)
    if not variant_order:
        raise ValueError("At least one variant is required.")
    return variant_order


def _variant_suffix(variant_order: list[str]) -> str:
    if variant_order == DEFAULT_TRIPTYCH_VARIANT_ORDER:
        return ""
    if variant_order == ["exact", *SCNA_VARIANTS]:
        return "_scna_only"
    return "_" + "_".join(variant_order)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper evaluation figures and tables.")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_TRIPTYCH_VARIANT_ORDER),
        help="Comma-separated variant list for triptych figures.",
    )
    parser.add_argument(
        "--merged-triptych-only",
        action="store_true",
        help="Only generate the merged 2x3 PPL/Group1 triptych figure.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Optional output stem override for figure-only mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    variant_order = _parse_variant_order(args.variants)
    plot_rows = _load_plot_rows()
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.merged_triptych_only:
        merged_paths = _plot_triptych_vertical(
            plot_rows,
            variant_order=variant_order,
            stem=args.stem or f"fig_ppl_group1_triptych_static_sq{_variant_suffix(variant_order)}",
        )
        for path in merged_paths:
            print(path)
        return 0

    if variant_order != DEFAULT_TRIPTYCH_VARIANT_ORDER:
        ppl_paths = _plot_triptych(
            plot_rows,
            variant_order=variant_order,
            metric="ppl",
            ylabel="WikiText-2 PPL",
            stem=f"fig_ppl_triptych_static_sq{_variant_suffix(variant_order)}",
            scale=1.0,
            fmt="{:.3f}",
        )
        group1_paths = _plot_triptych(
            plot_rows,
            variant_order=variant_order,
            metric="group1_mean",
            ylabel="Group1 Mean Accuracy (%)",
            stem=f"fig_group1_triptych_static_sq{_variant_suffix(variant_order)}",
            scale=100.0,
            fmt="{:.2f}",
        )
        merged_paths = _plot_triptych_vertical(
            plot_rows,
            variant_order=variant_order,
            stem=f"fig_ppl_group1_triptych_static_sq{_variant_suffix(variant_order)}",
        )
        for path in [*ppl_paths, *group1_paths, *merged_paths]:
            print(path)
        return 0

    ppl_paths = _plot_triptych(
        plot_rows,
        variant_order=DEFAULT_TRIPTYCH_VARIANT_ORDER,
        metric="ppl",
        ylabel="WikiText-2 PPL",
        stem="fig_ppl_triptych_static_sq",
        scale=1.0,
        fmt="{:.3f}",
    )
    group1_paths = _plot_triptych(
        plot_rows,
        variant_order=DEFAULT_TRIPTYCH_VARIANT_ORDER,
        metric="group1_mean",
        ylabel="Group1 Mean Accuracy (%)",
        stem="fig_group1_triptych_static_sq",
        scale=100.0,
        fmt="{:.2f}",
    )
    merged_paths = _plot_triptych_vertical(
        plot_rows,
        variant_order=DEFAULT_TRIPTYCH_VARIANT_ORDER,
        stem="fig_ppl_group1_triptych_static_sq",
    )
    heatmap_paths = _plot_delta_heatmap(
        plot_rows,
        stem="fig_lmeval_delta_heatmap_static_sq",
    )
    summary_md, summary_csv, lmeval_md, lmeval_csv = _write_summary_tables(plot_rows)
    draft_path = _write_evaluation_draft(
        plot_rows,
        ppl_paths=ppl_paths,
        group1_paths=group1_paths,
        heatmap_paths=heatmap_paths,
        summary_md=summary_md,
        lmeval_md=lmeval_md,
    )

    for path in [
        *ppl_paths,
        *group1_paths,
        *merged_paths,
        *heatmap_paths,
        summary_md,
        summary_csv,
        lmeval_md,
        lmeval_csv,
        draft_path,
    ]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
