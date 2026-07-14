from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a smoothed training history plot from a history.csv file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("history_csv", type=Path, help="Path to history.csv.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <history dir>/loss_curve_smooth.png.",
    )
    parser.add_argument(
        "--smooth-span",
        type=int,
        default=None,
        help="EMA span used for smoothing. Defaults to a value inferred from the history length.",
    )
    parser.add_argument(
        "--raw-max-points",
        type=int,
        default=4000,
        help="Maximum number of raw points to draw per series before downsampling.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.15,
        help="Fraction of the run shown in the right-side zoom panels.",
    )
    parser.add_argument(
        "--tail-min-points",
        type=int,
        default=500,
        help="Minimum number of epochs shown in the zoom panels.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output resolution.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.history_csv.is_file():
        raise FileNotFoundError(f"History CSV not found: {args.history_csv}")
    if args.smooth_span is not None and args.smooth_span <= 1:
        raise ValueError("--smooth-span must be greater than 1.")
    if args.raw_max_points <= 0:
        raise ValueError("--raw-max-points must be positive.")
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1].")
    if args.tail_min_points <= 0:
        raise ValueError("--tail-min-points must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")


def load_history(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"History CSV is empty: {path}")

    columns = reader.fieldnames or []
    history: dict[str, np.ndarray] = {}
    for name in columns:
        history[name] = np.array([float(row[name]) for row in rows], dtype=np.float64)
    return history


def infer_smooth_span(num_points: int) -> int:
    return max(25, min(401, num_points // 60))


def ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        smoothed[idx] = alpha * values[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def smooth_metric(values: np.ndarray, span: int) -> np.ndarray:
    if np.all(values > 0.0):
        return np.power(10.0, ema(np.log10(values), span))
    return ema(values, span)


def downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = int(math.ceil(len(x) / max_points))
    return x[::step], y[::step]


def format_epoch(value: float, _: float) -> str:
    return f"{int(value):,}"


def plot_metric(
    axis: plt.Axes,
    epochs: np.ndarray,
    raw_series: list[tuple[str, np.ndarray, str]],
    smooth_series_map: list[tuple[str, np.ndarray, str]],
    *,
    raw_max_points: int,
    ylabel: str,
    tail_start_epoch: float | None = None,
    best_epoch: int | None = None,
) -> None:
    for label, values, color in raw_series:
        raw_x, raw_y = downsample(epochs, values, raw_max_points)
        axis.plot(raw_x, raw_y, color=color, alpha=0.18, linewidth=0.8, label=f"{label} raw")

    for label, values, color in smooth_series_map:
        axis.plot(epochs, values, color=color, linewidth=2.2, label=f"{label} smooth")

    if tail_start_epoch is not None:
        axis.axvline(tail_start_epoch, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)
    if best_epoch is not None:
        axis.axvline(best_epoch, color="#111827", linestyle=":", linewidth=1.0, alpha=0.8)

    axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    axis.legend(loc="best", fontsize=9)


def main() -> int:
    args = parse_args()
    validate_args(args)

    history = load_history(args.history_csv)
    epochs = history["epoch"]
    num_points = len(epochs)
    smooth_span = args.smooth_span or infer_smooth_span(num_points)

    output_path = args.output or args.history_csv.with_name("loss_curve_smooth.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tail_points = max(args.tail_min_points, int(num_points * args.tail_fraction))
    tail_points = min(num_points, tail_points)
    tail_slice = slice(num_points - tail_points, num_points)
    tail_start_epoch = float(epochs[tail_slice][0])

    smoothed = {
        key: smooth_metric(values, smooth_span)
        for key, values in history.items()
        if key != "epoch"
    }

    best_epoch_index = int(np.argmin(history["mse"])) if "mse" in history else None
    best_epoch = int(epochs[best_epoch_index]) if best_epoch_index is not None else None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 8.5),
        sharey="row",
        constrained_layout=True,
        width_ratios=(3.8, 1.7),
    )

    loss_raw = [
        ("Avg loss", history["avg_loss"], "#2563eb"),
        ("Max loss", history["max_loss"], "#dc2626"),
    ]
    loss_smooth = [
        ("Avg loss", smoothed["avg_loss"], "#1d4ed8"),
        ("Max loss", smoothed["max_loss"], "#b91c1c"),
    ]
    error_raw = [
        ("MSE", history["mse"], "#059669"),
        ("RMSE", history["rmse"], "#7c3aed"),
    ]
    error_smooth = [
        ("MSE", smoothed["mse"], "#047857"),
        ("RMSE", smoothed["rmse"], "#6d28d9"),
    ]

    plot_metric(
        axes[0, 0],
        epochs,
        loss_raw,
        loss_smooth,
        raw_max_points=args.raw_max_points,
        ylabel="Loss",
        tail_start_epoch=tail_start_epoch,
        best_epoch=best_epoch,
    )
    plot_metric(
        axes[1, 0],
        epochs,
        error_raw,
        error_smooth,
        raw_max_points=args.raw_max_points,
        ylabel="Error",
        tail_start_epoch=tail_start_epoch,
        best_epoch=best_epoch,
    )
    plot_metric(
        axes[0, 1],
        epochs[tail_slice],
        [
            ("Avg loss", history["avg_loss"][tail_slice], "#2563eb"),
            ("Max loss", history["max_loss"][tail_slice], "#dc2626"),
        ],
        [
            ("Avg loss", smoothed["avg_loss"][tail_slice], "#1d4ed8"),
            ("Max loss", smoothed["max_loss"][tail_slice], "#b91c1c"),
        ],
        raw_max_points=args.raw_max_points,
        ylabel="Loss",
    )
    plot_metric(
        axes[1, 1],
        epochs[tail_slice],
        [
            ("MSE", history["mse"][tail_slice], "#059669"),
            ("RMSE", history["rmse"][tail_slice], "#7c3aed"),
        ],
        [
            ("MSE", smoothed["mse"][tail_slice], "#047857"),
            ("RMSE", smoothed["rmse"][tail_slice], "#6d28d9"),
        ],
        raw_max_points=args.raw_max_points,
        ylabel="Error",
        best_epoch=best_epoch if best_epoch is not None and best_epoch >= tail_start_epoch else None,
    )

    axes[0, 0].set_title("Full History")
    axes[0, 1].set_title(f"Tail View ({tail_points:,} epochs)")

    for axis in axes[1]:
        axis.set_xlabel("Epoch")
    for axis in axes.flat:
        axis.xaxis.set_major_formatter(FuncFormatter(format_epoch))

    fig.suptitle(
        f"{args.history_csv.parent.name}: smoothed history (EMA span={smooth_span})",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
