import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


DEVICE_ARGS = [
    ("b300", "B300"),
    ("b200", "B200"),
    ("awsv4", "AWSv4"),
    ("tpuv6e", "TPUv6e"),
]


def format_context(length: int) -> str:
    if length % 1024 == 0:
        return f"{length // 1024}K"
    return str(length)


def read_speedups(path: str, device_key: str, device_label: str) -> List[Dict[str, object]]:
    rows = []
    with open(os.path.join(path, "full_model_speedup.csv"), newline="") as f:
        source_rows = list(csv.DictReader(f))
    if not source_rows:
        return rows
    if "customsa_vs_flashattention_attention_core_x" in source_rows[0]:
        for row in source_rows:
            rows.append(
                {
                    "device_key": device_key,
                    "device_label": device_label,
                    "context_length": int(row["context_length"]),
                    "attention_only_speedup_x": float(
                        row["customsa_vs_flashattention_attention_core_x"]
                    ),
                    "end_to_end_speedup_x": float(
                        row["customsa_vs_flashattention_x"]
                    ),
                    "flashattention_model_ms": float(row["flashattention_model_ms"]),
                    "customsa_model_ms": float(row["customsa_model_ms"]),
                }
            )
        return rows

    grouped: Dict[int, Dict[str, Dict[str, str]]] = {}
    for row in source_rows:
        grouped.setdefault(int(row["context_length"]), {})[row["variant"]] = row
    for context_length, variants in sorted(grouped.items()):
        flash = variants["flashattention"]
        customsa = variants["customsa"]
        flashattention_model_ms = float(flash["variant_model_ms"])
        customsa_model_ms = float(customsa["variant_model_ms"])
        rows.append(
            {
                "device_key": device_key,
                "device_label": device_label,
                "context_length": context_length,
                "attention_only_speedup_x": float(flash["variant_attention_core_ms"])
                / float(customsa["variant_attention_core_ms"]),
                "end_to_end_speedup_x": flashattention_model_ms / customsa_model_ms,
                "flashattention_model_ms": flashattention_model_ms,
                "customsa_model_ms": customsa_model_ms,
            }
        )
    return rows


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: List[Dict[str, object]],
    output_dir: str,
    devices: List[tuple],
    title: str,
    label_mode: str,
) -> None:
    colors = {
        "b300": "#9467bd",
        "b200": "#1f77b4",
        "awsv4": "#d95f02",
        "tpuv6e": "#2ca02c",
    }
    label_offsets = {
        "b300": (0, 9),
        "b200": (0, 9),
        "awsv4": (0, -16),
        "tpuv6e": (0, -16),
    }
    metrics = [
        ("attention_only_speedup_x", "Attention Only"),
        ("end_to_end_speedup_x", "End to End"),
    ]

    all_lengths = sorted({int(row["context_length"]) for row in rows})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharex=True)
    for ax, (metric, panel_title) in zip(axes, metrics):
        for device_key, device_label in devices:
            device_rows = sorted(
                [row for row in rows if row["device_key"] == device_key],
                key=lambda row: row["context_length"],
            )
            xs = [row["context_length"] for row in device_rows]
            ys = [row[metric] for row in device_rows]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=colors[device_key],
                label=device_label,
            )
            labeled_points = list(zip(xs, ys))
            if label_mode == "endpoints" and labeled_points:
                labeled_points = [labeled_points[-1]]
            elif label_mode == "none":
                labeled_points = []
            for x, y in labeled_points:
                x_offset, y_offset = label_offsets[device_key]
                ax.annotate(
                    f"{y:.2f}x",
                    (x, y),
                    textcoords="offset points",
                    xytext=(x_offset, y_offset),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=colors[device_key],
                )
        ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_lengths)
        ax.set_xticklabels([format_context(length) for length in all_lengths])
        ax.set_title(panel_title)
        ax.set_xlabel("Context Length")
        ax.grid(True, axis="y", alpha=0.25)
        y_values = [row[metric] for row in rows]
        y_span = max(y_values) - min(y_values)
        ax.set_ylim(
            max(0.85, min(y_values) - 0.22 * y_span - 0.04),
            max(y_values) + 0.18 * y_span + 0.03,
        )
    axes[0].set_ylabel("CustomSA / FlashAttention Speedup")
    axes[1].legend(loc="upper left", frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "customsa_flash_speedups.png"), dpi=220)
    fig.savefig(os.path.join(output_dir, "customsa_flash_speedups.pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b200-dir", required=True)
    parser.add_argument("--b300-dir")
    parser.add_argument("--awsv4-dir", required=True)
    parser.add_argument("--tpuv6e-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--title",
        default="CustomSA over FlashAttention Speedup on Llama 3 8B Prefill",
    )
    parser.add_argument(
        "--label-mode",
        choices=["all", "endpoints", "none"],
        default="all",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_dirs = {
        "b200": args.b200_dir,
        "awsv4": args.awsv4_dir,
        "tpuv6e": args.tpuv6e_dir,
    }
    devices = [item for item in DEVICE_ARGS if item[0] in input_dirs]
    if args.b300_dir:
        input_dirs["b300"] = args.b300_dir
        devices = [item for item in DEVICE_ARGS if item[0] in input_dirs]
    rows: List[Dict[str, object]] = []
    for device_key, device_label in devices:
        rows.extend(read_speedups(input_dirs[device_key], device_key, device_label))
    rows.sort(key=lambda row: (row["device_key"], row["context_length"]))
    write_csv(os.path.join(args.output_dir, "customsa_flash_speedups.csv"), rows)
    plot(rows, args.output_dir, devices, args.title, args.label_mode)


if __name__ == "__main__":
    main()
