import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


VARIANT_COLORS = {
    "baseline": "#8b1e3f",
    "flashattention": "#0a9396",
    "customsa": "#355070",
}

CONVERSION_STYLES = {
    "with_conversion": {"linestyle": "-", "alpha": 1.0},
    "no_conversion": {"linestyle": "--", "alpha": 0.9},
}

PLATFORM_ORDER = {
    "gpu": ["a100", "h100", "b200"],
    "aws": ["awsv2", "awsv3", "awsv4"],
    "tpu": ["tpuv3", "tpuv4", "tpuv5e", "tpuv5p", "tpuv6e"],
}

ALL_DEVICE_ORDER = PLATFORM_ORDER["gpu"] + PLATFORM_ORDER["aws"] + PLATFORM_ORDER["tpu"]
SPEEDUP_DEVICE_ORDER = ALL_DEVICE_ORDER

DEVICE_DISPLAY_NAMES = {
    "a100": "A100",
    "h100": "H100",
    "b200": "B200",
    "awsv2": "AWSv2",
    "awsv3": "AWSv3",
    "awsv4": "AWSv4",
    "tpuv3": "TPUv3",
    "tpuv4": "TPUv4",
    "tpuv5e": "TPUv5e",
    "tpuv5p": "TPUv5p",
    "tpuv6e": "TPUv6e",
}

SEQUENCE_LENGTHS = [2048, 4096, 8192, 16384, 32768]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite-dir",
        help="Path to a completed attention result suite.",
    )
    parser.add_argument(
        "--gpu-suite-dir",
        help="Optional suite directory to use for gpu_conv and gpu_no_conv.",
    )
    parser.add_argument(
        "--tpu-suite-dir",
        help="Optional suite directory to use for tpu_conv and tpu_no_conv.",
    )
    parser.add_argument(
        "--aws-suite-dir",
        help="Optional suite directory to use for aws_conv and aws_no_conv.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to <suite-dir>/paper_figures.",
    )
    parser.add_argument(
        "--requested-only",
        action="store_true",
        help="Generate only the figures requested in figure_list.md.",
    )
    parser.add_argument(
        "--devices",
        help="Comma-separated device keys to include, e.g. h100,b200,awsv3,awsv4,tpuv3,tpuv6e.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def format_seq_length(value, _):
    if value >= 1024:
        return f"{int(value / 1024)}k"
    return str(int(value))


def tighten_square_layout(
    fig: plt.Figure,
    *,
    w_pad: float = 0.02,
    h_pad: float = 0.02,
    wspace: float = 0.03,
    hspace: float = 0.015,
) -> None:
    fig.set_constrained_layout_pads(
        w_pad=w_pad,
        h_pad=h_pad,
        wspace=wspace,
        hspace=hspace,
    )


def device_key(case_name: str) -> str:
    for suffix in ("_fp16", "_int8"):
        if case_name.endswith(suffix):
            return case_name[: -len(suffix)]
    return case_name


def condition_to_conversion(condition: str) -> str:
    return "no_conversion" if "_no_conv" in condition else "with_conversion"


def read_metadata_flag(suite_dir: Path, condition: str, field: str) -> bool | None:
    metadata_path = suite_dir / condition / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open() as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    value = payload.get(field)
    if value is None:
        return None
    return bool(value)


def prefer_no_onchip_io_gpu_suite(default_suite_dir: Path) -> Path:
    if read_metadata_flag(default_suite_dir, "gpu_conv", "ignore_onchip_io_bottleneck"):
        return default_suite_dir

    results_root = default_suite_dir.parent
    candidates = []
    for child in results_root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        if read_metadata_flag(child, "gpu_conv", "ignore_onchip_io_bottleneck"):
            candidates.append(child.resolve())
    if not candidates:
        return default_suite_dir
    return max(candidates, key=lambda path: path.name)


def resolve_condition_suite_dirs(args) -> dict[str, Path]:
    if not args.suite_dir and not (args.gpu_suite_dir and args.tpu_suite_dir):
        raise ValueError(
            "Provide --suite-dir or both --gpu-suite-dir and --tpu-suite-dir."
        )

    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else None
    gpu_suite_dir = (
        Path(args.gpu_suite_dir).resolve()
        if args.gpu_suite_dir
        else (
            prefer_no_onchip_io_gpu_suite(suite_dir) if suite_dir is not None else None
        )
    )
    tpu_suite_dir = (
        Path(args.tpu_suite_dir).resolve() if args.tpu_suite_dir else suite_dir
    )
    aws_suite_dir = (
        Path(args.aws_suite_dir).resolve() if args.aws_suite_dir else suite_dir
    )
    if gpu_suite_dir is None or tpu_suite_dir is None:
        raise ValueError("Could not resolve GPU/TPU suite directories.")

    condition_dirs = {
        "gpu_conv": gpu_suite_dir,
        "gpu_no_conv": gpu_suite_dir,
        "tpu_conv": tpu_suite_dir,
        "tpu_no_conv": tpu_suite_dir,
    }
    if aws_suite_dir is not None and (aws_suite_dir / "aws_conv").exists():
        condition_dirs["aws_conv"] = aws_suite_dir
        condition_dirs["aws_no_conv"] = aws_suite_dir
    return condition_dirs


def load_suite_dataframe(condition_suite_dirs: dict[str, Path]) -> pd.DataFrame:
    records = []
    for condition, suite_dir in condition_suite_dirs.items():
        latency_path = suite_dir / condition / "attention_latency.csv"
        df = pd.read_csv(latency_path)
        df["condition"] = condition
        df["platform"] = condition.split("_", 1)[0]
        df["conversion"] = condition_to_conversion(condition)
        df["device_key"] = df["case_name"].map(device_key)
        df["source_suite_dir"] = str(suite_dir)
        records.append(df)
    combined = pd.concat(records, ignore_index=True)
    return combined


def available_platforms(all_df: pd.DataFrame) -> list[str]:
    present = set(all_df["platform"].unique())
    return [platform for platform in PLATFORM_ORDER if platform in present]


def ordered_devices(all_df: pd.DataFrame, platform: str | None = None) -> list[str]:
    present = set(all_df["device_key"].unique())
    order = PLATFORM_ORDER[platform] if platform is not None else ALL_DEVICE_ORDER
    return [device for device in order if device in present]


def fp16_dataframe(all_df: pd.DataFrame) -> pd.DataFrame:
    fp16 = all_df[all_df["data_type"] == "fp16"].copy()
    fp16 = fp16.sort_values(
        ["platform", "device_key", "prefill_length", "variant", "condition"]
    )
    return fp16.drop_duplicates(["platform", "device_key", "prefill_length", "variant"])


def int8_dataframe(all_df: pd.DataFrame) -> pd.DataFrame:
    return all_df[all_df["data_type"] == "int8"].copy()


def case_label_map(all_df: pd.DataFrame) -> dict[str, str]:
    labels = (
        all_df.sort_values(["device_key", "case_label"])
        .drop_duplicates(["device_key"])[["device_key", "case_label"]]
        .set_index("device_key")["case_label"]
        .to_dict()
    )
    return labels


def display_name(device_key_name: str) -> str:
    return DEVICE_DISPLAY_NAMES.get(device_key_name, device_key_name.upper())


def parse_device_filter(args) -> list[str] | None:
    if not args.devices:
        return None
    devices = [
        device.strip().lower() for device in args.devices.split(",") if device.strip()
    ]
    invalid = [device for device in devices if device not in ALL_DEVICE_ORDER]
    if invalid:
        raise ValueError(f"Unknown device keys in --devices: {', '.join(invalid)}")
    return devices


def filter_devices(
    all_df: pd.DataFrame, selected_devices: list[str] | None
) -> pd.DataFrame:
    if selected_devices is None:
        return all_df
    filtered = all_df[all_df["device_key"].isin(selected_devices)].copy()
    if filtered.empty:
        raise ValueError("No rows remain after applying --devices filter.")
    return filtered


def with_attention_throughput(
    all_df: pd.DataFrame, latency_ms_col: str, output_col: str
) -> pd.DataFrame:
    df = all_df.copy()
    heads_total = df["batch_size"] * df["num_heads"]
    q_len = df["prefill_length"]
    kv_len = df["prefill_length"]
    score_flops = 2.0 * heads_total * q_len * kv_len * df["head_dim"]
    value_flops = score_flops
    df[output_col] = (score_flops + value_flops) / (df[latency_ms_col] / 1000.0) / 1e12
    return df


def requested_int8_dataframe(all_df: pd.DataFrame) -> pd.DataFrame:
    int8 = int8_dataframe(all_df)
    baseline = int8[int8["variant"] == "baseline"].sort_values(
        ["device_key", "prefill_length", "condition"]
    )
    baseline = baseline.drop_duplicates(["device_key", "prefill_length", "variant"])
    flash = int8[
        (int8["variant"] == "flashattention")
        & (int8["conversion"] == "with_conversion")
    ]
    customsa = int8[
        (int8["variant"] == "customsa") & (int8["conversion"] == "no_conversion")
    ]
    requested = pd.concat([baseline, flash, customsa], ignore_index=True)
    requested = requested.sort_values(["device_key", "prefill_length", "variant"])
    return requested


def make_handles_for_scaling_legend():
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["baseline"],
            linewidth=2.2,
            marker="o",
            linestyle="--",
            label="Baseline",
        ),
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["flashattention"],
            linewidth=2.2,
            marker="o",
            linestyle="-",
            label="FlashAttention",
        ),
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["flashattention"],
            linewidth=2.2,
            marker="o",
            linestyle="-",
            label="FlashAttention",
        ),
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["customsa"],
            linewidth=2.2,
            marker="o",
            linestyle="-",
            label="SCNA",
        ),
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["customsa"],
            linewidth=2.2,
            marker="o",
            linestyle="-",
            label="SCNA",
        ),
    ]
    return handles


def make_requested_latency_handles():
    return [
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["baseline"],
            linewidth=2.2,
            marker="o",
            label="Baseline",
        ),
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["flashattention"],
            linewidth=2.2,
            marker="o",
            linestyle="-",
            label="FlashAttention",
        ),
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["customsa"],
            linewidth=2.2,
            marker="o",
            linestyle="--",
            label="SCNA",
        ),
    ]


def make_requested_throughput_handles():
    return [
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["baseline"],
            linewidth=2.2,
            marker="o",
            linestyle="--",
            label="Baseline",
        ),
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["flashattention"],
            linewidth=2.2,
            marker="o",
            linestyle="-",
            label="FlashAttention",
        ),
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["customsa"],
            linewidth=2.2,
            marker="o",
            linestyle="-",
            label="SCNA",
        ),
    ]


def plot_scaling_panel(ax, subset: pd.DataFrame, row_name: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(SEQUENCE_LENGTHS)
    ax.xaxis.set_major_formatter(FuncFormatter(format_seq_length))
    ax.grid(True, which="major", linestyle="--", alpha=0.22)
    ax.grid(True, which="minor", linestyle=":", alpha=0.12)

    if row_name == "fp16":
        for variant in ["baseline", "flashattention", "customsa"]:
            variant_df = subset[subset["variant"] == variant].sort_values(
                "prefill_length"
            )
            ax.plot(
                variant_df["prefill_length"],
                variant_df["total_latency_ms"],
                color=VARIANT_COLORS[variant],
                linewidth=2.2,
                marker="o",
                markersize=4.6,
            )
        return

    baseline_df = subset[subset["variant"] == "baseline"].sort_values("prefill_length")
    ax.plot(
        baseline_df["prefill_length"],
        baseline_df["total_latency_ms"],
        color=VARIANT_COLORS["baseline"],
        linewidth=2.2,
        marker="o",
        markersize=4.6,
    )
    for variant in ["flashattention", "customsa"]:
        for conversion in ["with_conversion", "no_conversion"]:
            variant_df = subset[
                (subset["variant"] == variant) & (subset["conversion"] == conversion)
            ].sort_values("prefill_length")
            if variant_df.empty:
                continue
            style = CONVERSION_STYLES[conversion]
            ax.plot(
                variant_df["prefill_length"],
                variant_df["total_latency_ms"],
                color=VARIANT_COLORS[variant],
                linewidth=2.2,
                marker="o",
                markersize=4.6,
                linestyle=style["linestyle"],
                alpha=style["alpha"],
            )


def setup_latency_axis(ax, log_y: bool, box_aspect: float = 1.0) -> None:
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(SEQUENCE_LENGTHS)
    ax.xaxis.set_major_formatter(FuncFormatter(format_seq_length))
    ax.grid(True, which="major", linestyle="--", alpha=0.22)
    if log_y:
        ax.grid(True, which="minor", linestyle=":", alpha=0.12)
    ax.set_box_aspect(box_aspect)


def plot_requested_latency_panel(ax, subset: pd.DataFrame, log_y: bool) -> None:
    setup_latency_axis(ax, log_y=log_y)
    for variant, linestyle in [
        ("baseline", "-"),
        ("flashattention", "-"),
        ("customsa", "--"),
    ]:
        variant_df = subset[subset["variant"] == variant].sort_values("prefill_length")
        ax.plot(
            variant_df["prefill_length"],
            variant_df["total_latency_ms"],
            color=VARIANT_COLORS[variant],
            linewidth=2.2,
            marker="o",
            markersize=4.6,
            linestyle=linestyle,
        )


def plot_requested_throughput_panel(
    ax, subset: pd.DataFrame, *, annotate_speedup: bool = True, box_aspect: float = 1.0
) -> None:
    setup_latency_axis(ax, log_y=False, box_aspect=box_aspect)
    for variant, linestyle in [
        ("baseline", "--"),
        ("flashattention", "-"),
        ("customsa", "-"),
    ]:
        variant_df = subset[subset["variant"] == variant].sort_values("prefill_length")
        ax.plot(
            variant_df["prefill_length"],
            variant_df["effective_attention_tflops"],
            color=VARIANT_COLORS[variant],
            linewidth=2.2,
            marker="o",
            markersize=4.6,
            linestyle=linestyle,
        )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top * (1.20 if annotate_speedup else 1.06))
    if not annotate_speedup:
        return
    flash_df = subset[subset["variant"] == "flashattention"][
        ["prefill_length", "effective_attention_tflops"]
    ].rename(columns={"effective_attention_tflops": "flashattention_tflops"})
    custom_df = subset[subset["variant"] == "customsa"][
        ["prefill_length", "effective_attention_tflops"]
    ].rename(columns={"effective_attention_tflops": "customsa_tflops"})
    speedup_df = custom_df.merge(flash_df, on="prefill_length", how="inner")
    annotation_offsets = [(-8, 10), (0, 14), (8, 10), (0, 14), (-8, 10)]
    for idx, row in enumerate(speedup_df.itertuples(index=False)):
        if (
            not np.isfinite(row.customsa_tflops)
            or not np.isfinite(row.flashattention_tflops)
            or row.flashattention_tflops <= 0
        ):
            continue
        dx, dy = annotation_offsets[idx % len(annotation_offsets)]
        annotation = ax.annotate(
            f"{row.customsa_tflops / row.flashattention_tflops:.2f}x",
            (row.prefill_length, row.customsa_tflops),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="semibold",
            color=VARIANT_COLORS["customsa"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
        )
        annotation.set_clip_on(False)


def speedup_vs_flash_dataframe(throughput_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        throughput_df.pivot_table(
            index=["device_key", "prefill_length"],
            columns="variant",
            values="effective_attention_tflops",
        )
        .reset_index()
        .copy()
    )
    pivot["speedup_x"] = pivot["customsa"] / pivot["flashattention"]
    return pivot[["device_key", "prefill_length", "speedup_x"]]


def plot_requested_speedup_mini_panel(
    ax, subset: pd.DataFrame, *, box_aspect: float = 0.3
) -> None:
    setup_latency_axis(ax, log_y=False, box_aspect=box_aspect)
    ax.grid(True, axis="y", linestyle="--", alpha=0.22)
    ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--")
    values = subset["speedup_x"].to_numpy(dtype=float)
    x = subset["prefill_length"].to_numpy(dtype=float)
    if values.size:
        bars = ax.bar(
            x,
            values,
            width=x * 0.20,
            color=VARIANT_COLORS["customsa"],
            alpha=0.92,
            zorder=3,
        )
        ymin = min(0.95, np.nanmin(values) * 0.94)
        ymax = np.nanmax(values) * 1.18
        if np.isclose(ymax, ymin):
            ymax = ymin + 0.2
        ax.set_ylim(ymin, ymax)
        label_offset = (ymax - ymin) * 0.03
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + label_offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                color=VARIANT_COLORS["customsa"],
                fontweight="semibold",
                clip_on=False,
            )
    else:
        ax.set_ylim(0.9, 1.1)


def plot_requested_end_to_end(
    all_df: pd.DataFrame, output_dir: Path, log_y: bool
) -> None:
    fp16 = fp16_dataframe(all_df)
    int8 = requested_int8_dataframe(all_df)
    devices = ordered_devices(all_df)
    fig, axes = plt.subplots(
        2,
        len(devices),
        figsize=(2.95 * len(devices), 6.15),
        sharex=True,
        sharey="row" if log_y else False,
        constrained_layout=True,
    )
    tighten_square_layout(fig, wspace=0.035, hspace=0.01)

    for col, dev in enumerate(devices):
        fp16_ax = axes[0, col]
        int8_ax = axes[1, col]
        plot_requested_latency_panel(
            fp16_ax,
            fp16[(fp16["device_key"] == dev)],
            log_y=log_y,
        )
        plot_requested_latency_panel(
            int8_ax,
            int8[(int8["device_key"] == dev)],
            log_y=log_y,
        )
        fp16_ax.set_title(display_name(dev))
        int8_ax.set_xlabel("Prefill Length")

    axes[0, 0].set_ylabel("FP16 Latency (ms)")
    axes[1, 0].set_ylabel("INT8 Latency (ms)")
    title_suffix = "Log-Y" if log_y else "Linear-Y"
    fig.legend(
        handles=make_requested_latency_handles(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.04),
        columnspacing=1.5,
        handlelength=2.6,
    )
    # fig.suptitle(
    #     f"End-to-End Attention Latency Across Devices ({title_suffix})",
    #     y=1.10,
    #     fontsize=13,
    # )
    stem = (
        "figure_paper_requested_e2e_latency_logy"
        if log_y
        else "figure_paper_requested_e2e_latency_lineary"
    )
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def plot_requested_throughput(all_df: pd.DataFrame, output_dir: Path) -> None:
    fp16 = with_attention_throughput(
        fp16_dataframe(all_df), "total_latency_ms", "effective_attention_tflops"
    )
    int8 = with_attention_throughput(
        requested_int8_dataframe(all_df),
        "total_latency_ms",
        "effective_attention_tflops",
    )
    devices = ordered_devices(all_df)
    fig, axes = plt.subplots(
        2,
        len(devices),
        figsize=(2.95 * len(devices), 6.15),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    tighten_square_layout(fig, wspace=0.035, hspace=0.01)

    for col, dev in enumerate(devices):
        fp16_ax = axes[0, col]
        int8_ax = axes[1, col]
        plot_requested_throughput_panel(
            fp16_ax,
            fp16[(fp16["device_key"] == dev)],
        )
        plot_requested_throughput_panel(
            int8_ax,
            int8[(int8["device_key"] == dev)],
        )
        fp16_ax.set_title(display_name(dev))
        int8_ax.set_xlabel("Prefill Length")

    axes[0, 0].set_ylabel("FP16 Throughput (TFLOPS)")
    axes[1, 0].set_ylabel("INT8 Throughput (TFLOPS)")
    fig.legend(
        handles=make_requested_throughput_handles(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.04),
        columnspacing=1.5,
        handlelength=2.6,
    )
    # fig.suptitle(
    #     "QK+AV Throughput Across Devices (Linear-Y)",
    #     y=1.10,
    #     fontsize=13,
    # )
    save_figure(fig, output_dir, "figure_paper_requested_e2e_throughput")
    plt.close(fig)


def plot_requested_throughput_with_speedup(
    all_df: pd.DataFrame, output_dir: Path
) -> None:
    fp16 = with_attention_throughput(
        fp16_dataframe(all_df), "total_latency_ms", "effective_attention_tflops"
    )
    int8 = with_attention_throughput(
        requested_int8_dataframe(all_df),
        "total_latency_ms",
        "effective_attention_tflops",
    )
    fp16_speedup = speedup_vs_flash_dataframe(fp16)
    int8_speedup = speedup_vs_flash_dataframe(int8)
    devices = ordered_devices(all_df)

    fig = plt.figure(figsize=(2.8 * len(devices), 6.75), constrained_layout=True)
    tighten_square_layout(fig, wspace=0.03, hspace=0.001, h_pad=0.008)
    grid = fig.add_gridspec(
        5,
        len(devices),
        height_ratios=[0.8, 0.3, 0.1, 0.8, 0.3],
    )
    left_label_axes = []

    for col, dev in enumerate(devices):
        fp16_ax = fig.add_subplot(grid[0, col])
        fp16_speed_ax = fig.add_subplot(grid[1, col], sharex=fp16_ax)
        int8_ax = fig.add_subplot(grid[3, col])
        int8_speed_ax = fig.add_subplot(grid[4, col], sharex=int8_ax)

        plot_requested_throughput_panel(
            fp16_ax,
            fp16[fp16["device_key"] == dev],
            annotate_speedup=False,
            box_aspect=0.7,
        )
        plot_requested_speedup_mini_panel(
            fp16_speed_ax,
            fp16_speedup[fp16_speedup["device_key"] == dev].sort_values(
                "prefill_length"
            ),
            box_aspect=0.3,
        )
        plot_requested_throughput_panel(
            int8_ax,
            int8[int8["device_key"] == dev],
            annotate_speedup=False,
            box_aspect=0.7,
        )
        plot_requested_speedup_mini_panel(
            int8_speed_ax,
            int8_speedup[int8_speedup["device_key"] == dev].sort_values(
                "prefill_length"
            ),
            box_aspect=0.3,
        )

        fp16_ax.set_title(display_name(dev))
        plt.setp(fp16_ax.get_xticklabels(), visible=False)
        plt.setp(int8_ax.get_xticklabels(), visible=False)
        fp16_speed_ax.set_xlabel("")
        int8_speed_ax.set_xlabel("Prefill Length", fontweight="bold", fontsize=14)
        for axis in [fp16_ax, fp16_speed_ax, int8_ax, int8_speed_ax]:
            axis.tick_params(axis="y", labelrotation=90, pad=8)
            plt.setp(
                axis.get_yticklabels(),
                rotation_mode="anchor",
                va="center",
                ha="center",
            )
        if col == 0:
            fp16_ax.set_ylabel("Throughput", fontweight="bold", fontsize=14)
            fp16_speed_ax.set_ylabel("Speedup", fontweight="bold", fontsize=14)
            int8_ax.set_ylabel("Throughput", fontweight="bold", fontsize=14)
            int8_speed_ax.set_ylabel("Speedup", fontweight="bold", fontsize=14)
            left_label_axes.extend([fp16_ax, fp16_speed_ax, int8_ax, int8_speed_ax])
    for ax in left_label_axes:
        ax.yaxis.set_label_coords(-0.18, 0.5)
    fig.legend(
        handles=make_requested_throughput_handles(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=1.5,
        handlelength=2.6,
    )
    save_figure(fig, output_dir, "figure_paper_requested_e2e_throughput_with_speedup")
    plt.close(fig)


def plot_platform_scaling(
    all_df: pd.DataFrame, platform: str, output_dir: Path
) -> None:
    labels = case_label_map(all_df)
    devices = ordered_devices(all_df, platform)
    if not devices:
        return
    fp16 = fp16_dataframe(all_df)
    int8 = int8_dataframe(all_df)

    fig, axes = plt.subplots(
        2,
        len(devices),
        figsize=(4.0 * len(devices), 6.4),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    if len(devices) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, dev in enumerate(devices):
        fp16_ax = axes[0, col]
        int8_ax = axes[1, col]
        plot_scaling_panel(
            fp16_ax,
            fp16[(fp16["platform"] == platform) & (fp16["device_key"] == dev)],
            "fp16",
        )
        plot_scaling_panel(
            int8_ax,
            int8[(int8["platform"] == platform) & (int8["device_key"] == dev)],
            "int8",
        )
        fp16_ax.set_title(labels.get(dev, dev.upper()))
        int8_ax.set_xlabel("Prefill Length")

    axes[0, 0].set_ylabel("FP16 Latency (ms)")
    axes[1, 0].set_ylabel("INT8 Latency (ms)")
    fig.legend(
        handles=make_handles_for_scaling_legend(),
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.04),
        columnspacing=1.2,
        handlelength=2.5,
    )
    if platform == "gpu":
        title = "GPU End-to-End Attention Latency"
    elif platform == "aws":
        title = "AWS Trainium / Inferentia End-to-End Attention Latency"
    else:
        title = "TPU End-to-End Attention Latency"
    fig.suptitle(title, y=1.08, fontsize=13)
    save_figure(fig, output_dir, f"figure_paper_{platform}_latency_scaling")
    plt.close(fig)


def grouped_barh(
    ax,
    df: pd.DataFrame,
    device_labels: list[str],
    title: str,
    show_legend: bool = False,
) -> None:
    y = np.arange(len(device_labels))
    bar_h = 0.22
    offsets = {
        "baseline": -bar_h,
        "flashattention": 0.0,
        "customsa": bar_h,
    }
    variant_labels = {
        "baseline": "Baseline",
        "flashattention": "FlashAttention",
        "customsa": "CustomSA",
    }
    for variant in ["baseline", "flashattention", "customsa"]:
        variant_df = (
            df[df["variant"] == variant].set_index("case_label").reindex(device_labels)
        )
        ax.barh(
            y + offsets[variant],
            variant_df["total_latency_ms"],
            height=bar_h * 0.92,
            color=VARIANT_COLORS[variant],
            label=variant_labels[variant] if show_legend else None,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(device_labels)
    ax.set_xscale("log")
    ax.grid(True, axis="x", which="both", linestyle="--", alpha=0.22)
    ax.set_title(title)
    ax.invert_yaxis()


def plot_longest_context_summary(all_df: pd.DataFrame, output_dir: Path) -> None:
    longest = int(all_df["prefill_length"].max())
    dedup_fp16 = fp16_dataframe(all_df)
    no_conv_int8 = int8_dataframe(all_df)
    no_conv_int8 = no_conv_int8[no_conv_int8["conversion"] == "no_conversion"]

    platforms = available_platforms(all_df)
    panel_specs = []
    for platform in platforms:
        prefix = "GPU" if platform == "gpu" else "AWS" if platform == "aws" else "TPU"
        panel_specs.append(
            (
                platform,
                dedup_fp16[dedup_fp16["platform"] == platform],
                f"{prefix} FP16 @ 32k",
            )
        )
    for platform in platforms:
        prefix = "GPU" if platform == "gpu" else "AWS" if platform == "aws" else "TPU"
        panel_specs.append(
            (
                platform,
                no_conv_int8[no_conv_int8["platform"] == platform],
                f"{prefix} INT8 @ 32k (No Conversion)",
            )
        )

    fig, axes = plt.subplots(
        2, len(platforms), figsize=(4.8 * len(platforms), 8.2), constrained_layout=True
    )
    if len(platforms) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for ax, (platform, df, title) in zip(axes.flat, panel_specs):
        panel_df = df[df["prefill_length"] == longest].copy()
        order = ordered_devices(panel_df, platform)
        label_map = case_label_map(panel_df)
        labels = [label_map[key] for key in order if key in label_map]
        grouped_barh(ax, panel_df, labels, title, show_legend=ax is axes.flat[0])
        ax.set_xlabel("Latency (ms)")

    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Longest-Context End-to-End Latency Summary", y=1.05, fontsize=13)
    save_figure(fig, output_dir, "figure_paper_longest_context_latency")
    plt.close(fig)


def plot_requested_speedup_vs_flash(all_df: pd.DataFrame, output_dir: Path) -> None:
    fp16 = fp16_dataframe(all_df)
    int8 = requested_int8_dataframe(all_df)
    fp16_pivot = fp16.pivot_table(
        index=["device_key", "prefill_length"],
        columns="variant",
        values="total_latency_ms",
    ).reset_index()
    fp16_pivot["speedup_x"] = fp16_pivot["flashattention"] / fp16_pivot["customsa"]
    fp16_pivot["dtype"] = "fp16"

    int8_pivot = int8.pivot_table(
        index=["device_key", "prefill_length"],
        columns="variant",
        values="total_latency_ms",
    ).reset_index()
    int8_pivot["speedup_x"] = int8_pivot["flashattention"] / int8_pivot["customsa"]
    int8_pivot["dtype"] = "int8"

    speedup_df = pd.concat(
        [
            fp16_pivot[["device_key", "prefill_length", "dtype", "speedup_x"]],
            int8_pivot[["device_key", "prefill_length", "dtype", "speedup_x"]],
        ],
        ignore_index=True,
    )

    dtype_colors = {"fp16": "#355070", "int8": "#5e7ea3"}
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=dtype_colors["fp16"], label="FP16"),
        plt.Rectangle((0, 0), 1, 1, color=dtype_colors["int8"], label="INT8"),
    ]
    fig, axes = plt.subplots(
        1,
        len(ordered_devices(all_df)),
        figsize=(3.0 * len(ordered_devices(all_df)), 3.95),
        sharey=True,
        constrained_layout=True,
    )
    tighten_square_layout(fig, wspace=0.04, hspace=0.01)
    for ax, dev in zip(np.atleast_1d(axes), ordered_devices(all_df)):
        ax.set_xscale("log", base=2)
        ax.set_xticks(SEQUENCE_LENGTHS)
        ax.xaxis.set_major_formatter(FuncFormatter(format_seq_length))
        ax.grid(True, axis="y", linestyle="--", alpha=0.22)
        ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_box_aspect(1.0)
        ax.set_title(display_name(dev))

        subset = speedup_df[speedup_df["device_key"] == dev]
        for dtype_name, factor in [("fp16", 1 / 1.18), ("int8", 1.18)]:
            dtype_df = subset[subset["dtype"] == dtype_name].sort_values(
                "prefill_length"
            )
            centers = dtype_df["prefill_length"].to_numpy(dtype=float) * factor
            widths = dtype_df["prefill_length"].to_numpy(dtype=float) * 0.18
            ax.bar(
                centers,
                dtype_df["speedup_x"],
                width=widths,
                align="center",
                color=dtype_colors[dtype_name],
                alpha=0.95,
            )
        ax.set_xlabel("Prefill Length")

    np.atleast_1d(axes)[0].set_ylabel("CustomSA Speedup vs FlashAttention (x)")
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07)
    )
    fig.suptitle("CustomSA Speedup vs FlashAttention", y=1.13, fontsize=13)
    save_figure(fig, output_dir, "figure_paper_requested_speedup_vs_flashattention")
    plt.close(fig)


def plot_conversion_penalty(all_df: pd.DataFrame, output_dir: Path) -> None:
    longest = int(all_df["prefill_length"].max())
    int8_df = int8_dataframe(all_df)
    int8_df = int8_df[int8_df["prefill_length"] == longest]
    penalty = (
        int8_df.pivot_table(
            index=["platform", "device_key", "case_label", "variant"],
            columns="conversion",
            values="total_latency_ms",
        )
        .reset_index()
        .copy()
    )
    penalty = penalty[penalty["variant"].isin(["flashattention", "customsa"])]
    penalty["ratio"] = penalty["with_conversion"] / penalty["no_conversion"]

    platforms = available_platforms(all_df)
    fig, axes = plt.subplots(
        1,
        len(platforms),
        figsize=(4.8 * len(platforms), 4.4),
        sharey=True,
        constrained_layout=True,
    )
    variant_labels = {
        "flashattention": "FlashAttention",
        "customsa": "CustomSA",
    }
    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=VARIANT_COLORS["flashattention"],
            label=variant_labels["flashattention"],
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=VARIANT_COLORS["customsa"],
            label=variant_labels["customsa"],
        ),
    ]
    bar_w = 0.34
    for ax, platform in zip(np.atleast_1d(axes), platforms):
        platform_df = penalty[penalty["platform"] == platform].copy()
        order = ordered_devices(platform_df, platform)
        label_map = case_label_map(platform_df)
        labels = [label_map[key] for key in order if key in label_map]
        x = np.arange(len(labels))
        for offset, variant in zip(
            [-bar_w / 2, bar_w / 2], ["flashattention", "customsa"]
        ):
            variant_df = (
                platform_df[platform_df["variant"] == variant]
                .set_index("case_label")
                .reindex(labels)
            )
            bars = ax.bar(
                x + offset,
                variant_df["ratio"],
                width=bar_w,
                color=VARIANT_COLORS[variant],
                label=variant_labels[variant],
            )
            for bar, value in zip(bars, variant_df["ratio"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if platform == "gpu":
            ax.set_title("GPU INT8 Conversion Penalty")
        elif platform == "aws":
            ax.set_title("AWS INT8 Conversion Penalty")
        else:
            ax.set_title("TPU INT8 Conversion Penalty")
        ax.grid(True, axis="y", linestyle="--", alpha=0.22)
        ax.set_ylim(0.95, penalty["ratio"].max() * 1.12)
    np.atleast_1d(axes)[0].set_ylabel("With-Conversion / No-Conversion Latency (x)")
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03)
    )
    save_figure(fig, output_dir, "figure_paper_int8_conversion_penalty")
    plt.close(fig)


def plot_requested_customsa_conversion_ablation(
    all_df: pd.DataFrame, output_dir: Path
) -> None:
    int8 = int8_dataframe(all_df)
    custom = int8[int8["variant"] == "customsa"]
    ratio = (
        custom.pivot_table(
            index=["device_key", "prefill_length"],
            columns="conversion",
            values="total_latency_ms",
        )
        .reset_index()
        .copy()
    )
    ratio["overhead_ratio"] = ratio["with_conversion"] / ratio["no_conversion"]

    fig, axes = plt.subplots(
        1,
        len(ordered_devices(all_df)),
        figsize=(2.65 * len(ordered_devices(all_df)), 2.3),
        # sharey=True,
        constrained_layout=True,
    )
    for ax, dev in zip(np.atleast_1d(axes), ordered_devices(all_df)):
        subset = ratio[ratio["device_key"] == dev].sort_values("prefill_length")
        ax.plot(
            subset["prefill_length"],
            subset["overhead_ratio"],
            color="#0a9396",
            linewidth=2.2,
            marker="o",
            markersize=4.6,
        )
        ymax = np.max(subset["overhead_ratio"]) * 1.2
        ymin = np.min(subset["overhead_ratio"]) - 0.1
        yrange = ymax - ymin
        for x, y in zip(subset["prefill_length"], subset["overhead_ratio"]):
            if y < 1.0:
                continue
            ax.text(
                x,
                y + yrange * 0.1,
                f"{y:.2f}x",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
            )
        
        ax.set_ylim(ymin, ymax)
        ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(SEQUENCE_LENGTHS)
        ax.set_xlim(1300, 50000)
        ax.xaxis.set_major_formatter(FuncFormatter(format_seq_length))
        ax.grid(True, axis="y", linestyle="--", alpha=0.22)
        ax.set_title(display_name(dev))
        ax.set_xlabel("Prefill Length", fontweight="bold", fontsize=14)
        # ax.set_box_aspect(1.0)

    np.atleast_1d(axes)[0].set_ylabel("Speedup", fontweight="bold", fontsize=14)
    # fig.suptitle(
    #     "CustomSA Type Conversion Overhead Across Devices", y=1.11, fontsize=13
    # )
    save_figure(fig, output_dir, "figure_paper_requested_customsa_conversion_ablation")
    plt.close(fig)


def write_figure_notes(
    condition_suite_dirs: dict[str, Path],
    output_dir: Path,
    requested_only: bool,
    all_df: pd.DataFrame,
    selected_devices: list[str] | None,
) -> None:
    note_path = output_dir / "figure_notes.md"
    unique_sources = []
    for suite_dir in condition_suite_dirs.values():
        suite_str = str(suite_dir)
        if suite_str not in unique_sources:
            unique_sources.append(suite_str)
    lines = [
        "# Paper Figure Notes",
        "",
        f"- GPU source suite: `{condition_suite_dirs['gpu_conv']}`",
        f"- AWS source suite: `{condition_suite_dirs['aws_conv']}`"
        if "aws_conv" in condition_suite_dirs
        else "- AWS source suite: `(none)`",
        f"- TPU source suite: `{condition_suite_dirs['tpu_conv']}`",
        f"- Unique source suites: {', '.join(f'`{suite}`' for suite in unique_sources)}",
        f"- Requested-only generation: `{int(requested_only)}`",
        f"- Selected devices: `{', '.join(selected_devices)}`"
        if selected_devices
        else "- Selected devices: `(all available)`",
    ]
    if not requested_only:
        lines.extend(
            [
                "- `figure_paper_gpu_latency_scaling`: per-GPU scaling curves over prefill length; FP16 on the top row, INT8 on the bottom row.",
                "- `figure_paper_tpu_latency_scaling`: per-TPU scaling curves over prefill length; FP16 on the top row, INT8 on the bottom row.",
                "- `figure_paper_longest_context_latency`: absolute end-to-end latency at sequence length 32768, split by platform and dtype.",
                "- `figure_paper_int8_conversion_penalty`: ratio between INT8 with-conversion and no-conversion runs for FlashAttention and CustomSA at sequence length 32768.",
            ]
        )
    lines.extend(
        [
            f"- `figure_paper_requested_e2e_latency_lineary`: requested all-device 2x{len(ordered_devices(all_df))} grid using baseline, FlashAttention with conversion, and CustomSA without conversion, with linear y-axis.",
            f"- `figure_paper_requested_e2e_latency_logy`: same requested all-device 2x{len(ordered_devices(all_df))} grid with logarithmic y-axis.",
            f"- `figure_paper_requested_e2e_throughput`: requested all-device 2x{len(ordered_devices(all_df))} grid of useful attention GEMM throughput in TFLOPS using only `QK + AV` FLOPs divided by end-to-end latency, with linear y-axis, annotated above the SCNA line with SCNA speedup over FlashAttention at the same prefill length.",
            f"- `figure_paper_requested_e2e_throughput_with_speedup`: requested all-device 4x{len(ordered_devices(all_df))} composite figure; for each dtype row, throughput curves are shown above and SCNA-vs-FlashAttention speedup bars are shown directly below with a `0.7/0.3` height split.",
            f"- `figure_paper_requested_speedup_vs_flashattention`: requested 1x{len(ordered_devices(all_df))} grouped-bar figure showing CustomSA speedup over FlashAttention by prefill length for FP16 and INT8.",
            f"- `figure_paper_requested_customsa_conversion_ablation`: requested 1x{len(ordered_devices(all_df))} figure showing CustomSA type-conversion overhead ratio versus no-conversion across prefill lengths.",
            "- All latencies come from runs with `--ignore-hbm-bottleneck`, so the figures emphasize end-to-end compute and on-chip scheduling behavior rather than HBM throttling.",
            "- Throughput is computed as `(QK FLOPs + AV FLOPs) / total_latency_s`, intentionally excluding softmax and exp work so the figure focuses on useful GEMM throughput.",
            "- Each SCNA throughput point is annotated with `SCNA / FlashAttention` throughput speedup at the same prefill length.",
            "- In the composite throughput-with-speedup figure, the top subpanel keeps only the curves; the speedup labels are moved into the lower bar-chart subpanel.",
        ]
    )
    gpu_df = all_df[all_df["platform"] == "gpu"]
    if not gpu_df.empty and "ignore_onchip_io_bottleneck" in gpu_df.columns:
        if bool(gpu_df["ignore_onchip_io_bottleneck"].fillna(False).all()):
            lines.append(
                "- The GPU source suite also uses `--ignore-onchip-io-bottleneck`, so the GPU figures reflect compute-limited behavior after removing the on-chip/global-buffer bottleneck."
            )
        elif bool(gpu_df["ignore_onchip_io_bottleneck"].fillna(False).any()):
            lines.append(
                "- The GPU source data mixes runs with and without `--ignore-onchip-io-bottleneck`; interpret GPU comparisons accordingly."
            )
        else:
            lines.append(
                "- The GPU source suite keeps on-chip I/O bottlenecks enabled, so GPU results may still be limited by the modeled on-chip/global-buffer path."
            )
    note_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    condition_suite_dirs = resolve_condition_suite_dirs(args)
    selected_devices = parse_device_filter(args)
    default_output_root = (
        Path(args.suite_dir).resolve()
        if args.suite_dir
        else condition_suite_dirs["gpu_conv"]
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else default_output_root / "paper_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    all_df = filter_devices(
        load_suite_dataframe(condition_suite_dirs), selected_devices
    )
    if not args.requested_only:
        for platform in available_platforms(all_df):
            plot_platform_scaling(all_df, platform, output_dir)
        plot_longest_context_summary(all_df, output_dir)
        plot_conversion_penalty(all_df, output_dir)
    plot_requested_end_to_end(all_df, output_dir, log_y=False)
    plot_requested_end_to_end(all_df, output_dir, log_y=True)
    plot_requested_throughput(all_df, output_dir)
    plot_requested_throughput_with_speedup(all_df, output_dir)
    plot_requested_speedup_vs_flash(all_df, output_dir)
    plot_requested_customsa_conversion_ablation(all_df, output_dir)
    write_figure_notes(
        condition_suite_dirs,
        output_dir,
        requested_only=args.requested_only,
        all_df=all_df,
        selected_devices=selected_devices,
    )


if __name__ == "__main__":
    main()
