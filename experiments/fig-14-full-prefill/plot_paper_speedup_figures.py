import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

DEFAULT_FP16_MAIN_CSV = (
    RESULTS_DIR
    / "2026-06-10_23-56-10_b200_awsv4_tpuv6e_speedup_512k_figure"
    / "customsa_flash_speedups.csv"
)
DEFAULT_INT8_MAIN_CSV = (
    RESULTS_DIR
    / "2026-06-11_19-36-02_b300_b200_awsv4_tpuv6e_int8_speedup_512k_figure"
    / "customsa_flash_speedups.csv"
)
DEFAULT_B300_FP16_CSV = (
    RESULTS_DIR
    / "2026-06-11_19-00-25_llama3_8b_full_model_b300_512k"
    / "full_model_speedup.csv"
)
DEFAULT_B300_INT8_CSV = (
    RESULTS_DIR
    / "2026-06-11_19-34-59_llama3_8b_full_model_b300_int8_512k"
    / "full_model_speedup.csv"
)

CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
SPARSE_CONTEXT_TICKS = [2048, 8192, 32768, 131072, 524288]
MAIN_DEVICES = ["b200", "awsv4", "tpuv6e"]
DEVICE_LABELS = {
    "b200": "B200",
    "awsv4": "AWSv4",
    "tpuv6e": "TPUv6e",
}
DEVICE_COLORS = {
    "b200": "#355070",
    "awsv4": "#0a9396",
    "tpuv6e": "#8b1e3f",
}
DEVICE_MARKERS = {
    "b200": "o",
    "awsv4": "s",
    "tpuv6e": "D",
}
METRIC_COLORS = {
    "attention": "#0a9396",
    "e2e": "#355070",
}
METRIC_MARKERS = {
    "attention": "o",
    "e2e": "s",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 13.5,
            "axes.labelsize": 14,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_context(value, _pos=None) -> str:
    value = int(value)
    if value >= 1024:
        return f"{value // 1024}k"
    return str(value)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)


def write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def setup_speedup_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(SPARSE_CONTEXT_TICKS)
    ax.xaxis.set_major_formatter(FuncFormatter(format_context))
    ax.set_xlim(CONTEXT_LENGTHS[0] * 0.86, CONTEXT_LENGTHS[-1] * 1.20)
    ax.grid(True, axis="y", linestyle="--", alpha=0.22)
    ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--", zorder=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Context Length", fontweight="bold", fontsize=14, labelpad=7)
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(0)
        tick_label.set_ha("center")


def set_panel_ylim(ax: plt.Axes, values: List[float]) -> None:
    y_min = min(values + [1.0])
    y_max = max(values + [1.0])
    span = max(y_max - y_min, 0.05)
    ax.set_ylim(y_min - 0.16 * span - 0.01, y_max + 0.20 * span + 0.02)


def add_dtype_label(ax: plt.Axes, dtype_name: str) -> None:
    ax.text(
        0.035,
        0.94,
        dtype_name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
        zorder=5,
    )


def add_shared_legend(fig: plt.Figure, handles: List[plt.Line2D], ncol: int) -> None:
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 1.08),
        columnspacing=1.45,
        handlelength=2.5,
    )


def make_device_handles() -> List[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            color=DEVICE_COLORS[device],
            marker=DEVICE_MARKERS[device],
            linewidth=2.2,
            markersize=5.4,
            label=DEVICE_LABELS[device],
        )
        for device in MAIN_DEVICES
    ]


def make_metric_handles() -> List[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            color=METRIC_COLORS[key],
            marker=METRIC_MARKERS[key],
            linewidth=2.2,
            markersize=5.4,
            label=label,
        )
        for key, label in [("attention", "Attention"), ("e2e", "End-to-End")]
    ]


def read_main_speedups(fp16_csv: Path, int8_csv: Path) -> pd.DataFrame:
    fp16 = pd.read_csv(fp16_csv)
    fp16["dtype"] = "FP16"
    int8 = pd.read_csv(int8_csv)
    int8["dtype"] = "INT8"
    combined = pd.concat([fp16, int8], ignore_index=True)
    combined = combined[combined["device_key"].isin(MAIN_DEVICES)].copy()
    combined["device_label"] = combined["device_key"].map(DEVICE_LABELS)
    return combined.sort_values(["dtype", "device_key", "context_length"])


def plot_main_e2e_speedup(
    fp16_csv: Path,
    int8_csv: Path,
    output_dir: Path,
) -> None:
    data = read_main_speedups(fp16_csv, int8_csv)
    rows = [
        {
            "dtype": row["dtype"],
            "device_key": row["device_key"],
            "device_label": row["device_label"],
            "context_length": int(row["context_length"]),
            "end_to_end_speedup_x": float(row["end_to_end_speedup_x"]),
            "flashattention_model_ms": float(row["flashattention_model_ms"]),
            "customsa_model_ms": float(row["customsa_model_ms"]),
        }
        for _, row in data.iterrows()
    ]
    write_rows(output_dir / "paper_main_e2e_speedups.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.95), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.005, h_pad=0.01, wspace=0.012, hspace=0.01)
    endpoint_offsets = {
        "b200": (0, 9),
        "awsv4": (0, -15),
        "tpuv6e": (0, 9),
    }
    for ax, dtype_name in zip(axes, ["FP16", "INT8"]):
        panel_data = data[data["dtype"] == dtype_name]
        y_values = []
        setup_speedup_axis(ax)
        for device in MAIN_DEVICES:
            device_data = panel_data[panel_data["device_key"] == device].sort_values(
                "context_length"
            )
            xs = device_data["context_length"].to_numpy(dtype=float)
            ys = device_data["end_to_end_speedup_x"].to_numpy(dtype=float)
            y_values.extend(ys.tolist())
            ax.plot(
                xs,
                ys,
                color=DEVICE_COLORS[device],
                marker=DEVICE_MARKERS[device],
                linewidth=2.2,
                markersize=4.8,
                label=DEVICE_LABELS[device],
            )
            if len(xs):
                dx, dy = endpoint_offsets[device]
                ax.annotate(
                    f"{ys[-1]:.2f}x",
                    (xs[-1], ys[-1]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    ha="right",
                    va="center",
                    fontsize=11.2,
                    color=DEVICE_COLORS[device],
                    clip_on=False,
                )
        set_panel_ylim(ax, y_values)
        add_dtype_label(ax, dtype_name)
        ax.set_box_aspect(0.67)
    axes[0].set_ylabel(
        "E2E Speedup\nover FlashAttention",
        fontweight="bold",
        fontsize=14,
        labelpad=9,
    )
    add_shared_legend(fig, make_device_handles(), ncol=3)
    save_figure(fig, output_dir, "figure_paper_e2e_speedup_main_devices")
    plt.close(fig)


def read_b300_speedup(path: Path, dtype_name: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "customsa_vs_flashattention_attention_core_x" not in data.columns:
        flash = data[data["variant"] == "flashattention"].set_index("context_length")
        customsa = data[data["variant"] == "customsa"].set_index("context_length")
        contexts = sorted(set(flash.index).intersection(customsa.index))
        return pd.DataFrame(
            {
                "dtype": dtype_name,
                "context_length": contexts,
                "attention_speedup_x": [
                    float(flash.loc[length, "variant_attention_core_ms"])
                    / float(customsa.loc[length, "variant_attention_core_ms"])
                    for length in contexts
                ],
                "end_to_end_speedup_x": [
                    float(flash.loc[length, "variant_model_ms"])
                    / float(customsa.loc[length, "variant_model_ms"])
                    for length in contexts
                ],
                "flashattention_model_ms": [
                    float(flash.loc[length, "variant_model_ms"]) for length in contexts
                ],
                "customsa_model_ms": [
                    float(customsa.loc[length, "variant_model_ms"]) for length in contexts
                ],
            }
        )
    return pd.DataFrame(
        {
            "dtype": dtype_name,
            "context_length": data["context_length"].astype(int),
            "attention_speedup_x": data[
                "customsa_vs_flashattention_attention_core_x"
            ].astype(float),
            "end_to_end_speedup_x": data["customsa_vs_flashattention_x"].astype(float),
            "flashattention_model_ms": data["flashattention_model_ms"].astype(float),
            "customsa_model_ms": data["customsa_model_ms"].astype(float),
        }
    )


def plot_b300_speedup(
    fp16_csv: Path,
    int8_csv: Path,
    output_dir: Path,
) -> None:
    data = pd.concat(
        [
            read_b300_speedup(fp16_csv, "FP16"),
            read_b300_speedup(int8_csv, "INT8"),
        ],
        ignore_index=True,
    )
    write_rows(output_dir / "paper_b300_speedups.csv", data.to_dict("records"))

    fig, axes = plt.subplots(1, 2, figsize=(7.7, 2.86), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.005, h_pad=0.01, wspace=0.012, hspace=0.01)
    metric_specs = [
        ("attention_speedup_x", "Attention", "attention", 9),
        ("end_to_end_speedup_x", "End-to-End", "e2e", -15),
    ]
    for ax, dtype_name in zip(axes, ["FP16", "INT8"]):
        panel_data = data[data["dtype"] == dtype_name].sort_values("context_length")
        setup_speedup_axis(ax)
        y_values = []
        for column, label, key, endpoint_dy in metric_specs:
            xs = panel_data["context_length"].to_numpy(dtype=float)
            ys = panel_data[column].to_numpy(dtype=float)
            y_values.extend(ys.tolist())
            ax.plot(
                xs,
                ys,
                color=METRIC_COLORS[key],
                marker=METRIC_MARKERS[key],
                linewidth=2.2,
                markersize=4.8,
                label=label,
            )
            ax.annotate(
                f"{ys[-1]:.2f}x",
                (xs[-1], ys[-1]),
                textcoords="offset points",
                xytext=(0, endpoint_dy),
                ha="right",
                va="center",
                fontsize=11.2,
                color=METRIC_COLORS[key],
                clip_on=False,
            )
        set_panel_ylim(ax, y_values)
        add_dtype_label(ax, dtype_name)
        ax.set_box_aspect(0.68)
    axes[0].set_ylabel(
        "B300 Speedup\nover FlashAttention",
        fontweight="bold",
        fontsize=14,
        labelpad=9,
    )
    add_shared_legend(fig, make_metric_handles(), ncol=2)
    save_figure(fig, output_dir, "figure_paper_b300_speedup")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures",
        choices=["all", "main", "b300"],
        default="all",
        help="Select the paper panel(s) to render.",
    )
    parser.add_argument("--fp16-main-csv", default=str(DEFAULT_FP16_MAIN_CSV))
    parser.add_argument("--int8-main-csv", default=str(DEFAULT_INT8_MAIN_CSV))
    parser.add_argument("--b300-fp16-csv", default=str(DEFAULT_B300_FP16_CSV))
    parser.add_argument("--b300-int8-csv", default=str(DEFAULT_B300_INT8_CSV))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output_dir = Path(args.output_dir).resolve()
    if args.figures in ("all", "main"):
        plot_main_e2e_speedup(
            Path(args.fp16_main_csv).resolve(),
            Path(args.int8_main_csv).resolve(),
            output_dir,
        )
    if args.figures in ("all", "b300"):
        plot_b300_speedup(
            Path(args.b300_fp16_csv).resolve(),
            Path(args.b300_int8_csv).resolve(),
            output_dir,
        )
    print(f"[outputs] {output_dir}")


if __name__ == "__main__":
    main()
