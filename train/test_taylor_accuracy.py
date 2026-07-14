from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TensorFn = Callable[[torch.Tensor], torch.Tensor]
MACLAURIN_CENTER = 0.0
SCRIPT_DIR = Path(__file__).resolve().parent


# These defaults are intentionally local to x=0 so the Maclaurin comparison is meaningful.
ANALYSIS_RANGES: dict[str, tuple[float, float]] = {
    "exp": (-1.0, 1.0),
    "exp2": (-1.0, 1.0),
    "sigmoid": (-2.0, 2.0),
    "softplus": (-2.0, 2.0),
    "tanh": (-1.0, 1.0),
    "erf": (-1.0, 1.0),
    "sin": (-math.pi / 2, math.pi / 2),
    "cos": (-math.pi / 2, math.pi / 2),
    "gelu": (-1.5, 1.5),
}

SQRT_PI = math.sqrt(math.pi)
SQRT_2PI = math.sqrt(2.0 * math.pi)
LN2 = math.log(2.0)


# Coefficients are for sum_k c_k * x^k, truncated to 6th order.
MACLAURIN_COEFFICIENTS: dict[str, list[float]] = {
    "exp": [
        1.0,
        1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
    ],
    "exp2": [
        1.0,
        LN2,
        LN2**2 / 2.0,
        LN2**3 / 6.0,
        LN2**4 / 24.0,
        LN2**5 / 120.0,
        LN2**6 / 720.0,
    ],
    "sigmoid": [
        1.0 / 2.0,
        1.0 / 4.0,
        0.0,
        -1.0 / 48.0,
        0.0,
        1.0 / 480.0,
        0.0,
    ],
    "softplus": [
        LN2,
        1.0 / 2.0,
        1.0 / 8.0,
        0.0,
        -1.0 / 192.0,
        0.0,
        1.0 / 2880.0,
    ],
    "tanh": [
        0.0,
        1.0,
        0.0,
        -1.0 / 3.0,
        0.0,
        2.0 / 15.0,
        0.0,
    ],
    "erf": [
        0.0,
        2.0 / SQRT_PI,
        0.0,
        -2.0 / (3.0 * SQRT_PI),
        0.0,
        1.0 / (5.0 * SQRT_PI),
        0.0,
    ],
    "sin": [
        0.0,
        1.0,
        0.0,
        -1.0 / 6.0,
        0.0,
        1.0 / 120.0,
        0.0,
    ],
    "cos": [
        1.0,
        0.0,
        -1.0 / 2.0,
        0.0,
        1.0 / 24.0,
        0.0,
        -1.0 / 720.0,
    ],
    "gelu": [
        0.0,
        1.0 / 2.0,
        1.0 / SQRT_2PI,
        0.0,
        -1.0 / (6.0 * SQRT_2PI),
        0.0,
        1.0 / (40.0 * SQRT_2PI),
    ],
}


@dataclass(frozen=True)
class TaylorFunctionSpec:
    name: str
    target_fn: TensorFn
    l_range: float
    r_range: float


@dataclass(frozen=True)
class TaylorResult:
    function: str
    order: int
    l_range: float
    r_range: float
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    x_at_max_abs_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Maclaurin Taylor-series accuracy on local intervals near zero.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        choices=[
            "exp",
            "exp2",
            "sigmoid",
            "softplus",
            "tanh",
            "erf",
            "sin",
            "cos",
            "gelu",
        ],
        default=[
            "exp",
            "exp2",
            "sigmoid",
            "softplus",
            "tanh",
            "erf",
            "sin",
            "cos",
            "gelu",
        ],
        help="Functions to evaluate.",
    )
    parser.add_argument(
        "--min-order",
        type=int,
        default=3,
        help="Minimum Taylor order to test.",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=6,
        help="Maximum Taylor order to test.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8193,
        help="Number of evaluation points per interval.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SCRIPT_DIR / "taylor_accuracy_results.csv",
        help="CSV path for the accuracy table.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=SCRIPT_DIR / "taylor_series_comparison.png",
        help="Image path for the multi-panel function plot.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=200,
        help="Saved figure DPI.",
    )
    args = parser.parse_args()

    if args.min_order < 0:
        parser.error("--min-order must be non-negative.")
    if args.max_order < args.min_order:
        parser.error("--max-order must be >= --min-order.")
    if args.num_points < 2:
        parser.error("--num-points must be at least 2.")

    return args


def build_function_specs() -> dict[str, TaylorFunctionSpec]:
    functions: dict[str, TensorFn] = {
        "exp": torch.exp,
        "exp2": torch.exp2,
        "sigmoid": torch.sigmoid,
        "softplus": F.softplus,
        "tanh": torch.tanh,
        "erf": torch.erf,
        "sin": torch.sin,
        "cos": torch.cos,
        "gelu": lambda x: F.gelu(x, approximate="none"),
    }

    specs: dict[str, TaylorFunctionSpec] = {}
    for name, target_fn in functions.items():
        l_range, r_range = ANALYSIS_RANGES[name]
        specs[name] = TaylorFunctionSpec(
            name=name,
            target_fn=target_fn,
            l_range=l_range,
            r_range=r_range,
        )

    return specs


def taylor_coefficients(name: str, order: int) -> list[float]:
    return MACLAURIN_COEFFICIENTS[name][: order + 1]


def evaluate_polynomial(coefficients: list[float], x: torch.Tensor) -> torch.Tensor:
    approximation = torch.zeros_like(x, dtype=torch.float64)
    for coefficient in reversed(coefficients):
        approximation = approximation * x + coefficient
    return approximation


def evaluate_accuracy(
    spec: TaylorFunctionSpec,
    order: int,
    num_points: int,
) -> TaylorResult:
    x = torch.linspace(spec.l_range, spec.r_range, steps=num_points, dtype=torch.float64)
    target = spec.target_fn(x)
    coefficients = taylor_coefficients(spec.name, order)
    approximation = evaluate_polynomial(coefficients, x)

    abs_error = (approximation - target).abs()
    max_abs_error, max_index = torch.max(abs_error, dim=0)
    mean_abs_error = torch.mean(abs_error)
    rmse = torch.sqrt(torch.mean((approximation - target) ** 2))

    return TaylorResult(
        function=spec.name,
        order=order,
        l_range=spec.l_range,
        r_range=spec.r_range,
        max_abs_error=float(max_abs_error.item()),
        mean_abs_error=float(mean_abs_error.item()),
        rmse=float(rmse.item()),
        x_at_max_abs_error=float(x[max_index].item()),
    )


def should_use_symlog(y_values: list[torch.Tensor]) -> bool:
    finite_tensors = [values[torch.isfinite(values)] for values in y_values]
    finite_tensors = [values for values in finite_tensors if values.numel() > 0]
    if not finite_tensors:
        return False

    combined = torch.cat(finite_tensors)
    min_value = float(torch.min(combined).item())
    max_value = float(torch.max(combined).item())
    max_abs = float(torch.max(torch.abs(combined)).item())

    if min_value < 0.0 < max_value:
        return True
    if max_abs == 0.0:
        return False

    nonzero = torch.abs(combined)
    nonzero = nonzero[nonzero > 0.0]
    if nonzero.numel() == 0:
        return False

    min_nonzero = float(torch.min(nonzero).item())
    return max_abs / min_nonzero > 1e3


def plot_functions(
    function_names: list[str],
    function_specs: dict[str, TaylorFunctionSpec],
    min_order: int,
    max_order: int,
    num_points: int,
    output_path: Path,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ncols = 3
    nrows = math.ceil(len(function_names) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.5 * ncols, 5.5 * nrows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for axis, function_name in zip(axes_list, function_names):
        spec = function_specs[function_name]
        x = torch.linspace(spec.l_range, spec.r_range, steps=num_points, dtype=torch.float64)
        target = spec.target_fn(x)
        series_values: list[torch.Tensor] = []

        axis.plot(
            x.numpy(),
            target.numpy(),
            color="black",
            linewidth=2.5,
            label=f"{function_name}(x)",
        )
        for order in range(min_order, max_order + 1):
            approximation = evaluate_polynomial(taylor_coefficients(function_name, order), x)
            series_values.append(approximation)
            axis.plot(
                x.numpy(),
                approximation.numpy(),
                linewidth=1.8,
                label=f"Taylor order {order}",
            )

        if should_use_symlog([target, *series_values]):
            axis.set_yscale("symlog", linthresh=1e-3)

        axis.set_title(f"{function_name}: [{spec.l_range:.3f}, {spec.r_range:.3f}]")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=9)

    for axis in axes_list[len(function_names) :]:
        axis.axis("off")

    fig.suptitle("Function vs. Maclaurin Taylor Series (orders 3-6)", fontsize=20)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_results_csv(path: Path, results: list[TaylorResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "function",
                "order",
                "l_range",
                "r_range",
                "max_abs_error",
                "mean_abs_error",
                "rmse",
                "x_at_max_abs_error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "function": result.function,
                    "order": result.order,
                    "l_range": result.l_range,
                    "r_range": result.r_range,
                    "max_abs_error": result.max_abs_error,
                    "mean_abs_error": result.mean_abs_error,
                    "rmse": result.rmse,
                    "x_at_max_abs_error": result.x_at_max_abs_error,
                }
            )


def print_summary(results: list[TaylorResult]) -> None:
    print()
    print("Best order by RMSE")
    for function_name in sorted({result.function for result in results}):
        candidates = [result for result in results if result.function == function_name]
        best = min(candidates, key=lambda result: result.rmse)
        print(
            f"  {function_name:8s} order={best.order} rmse={best.rmse:.6e}"
            f" max_abs={best.max_abs_error:.6e}"
        )


def print_results(results: list[TaylorResult]) -> None:
    print(f"Taylor approximation regions (Maclaurin center = {MACLAURIN_CENTER:g})")
    for function_name in sorted({result.function for result in results}):
        first_result = next(result for result in results if result.function == function_name)
        print(f"  {function_name:8s} [{first_result.l_range:8.3f}, {first_result.r_range:8.3f}]")

    print()
    header = (
        f"{'function':8s} {'order':>5s} {'max_abs':>14s} {'mean_abs':>14s}"
        f" {'rmse':>14s} {'x@max_abs':>14s}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.function:8s} {result.order:5d} "
            f"{result.max_abs_error:14.6e} {result.mean_abs_error:14.6e} "
            f"{result.rmse:14.6e} {result.x_at_max_abs_error:14.6f}"
        )


def main() -> None:
    args = parse_args()
    function_specs = build_function_specs()

    results: list[TaylorResult] = []
    for function_name in args.functions:
        spec = function_specs[function_name]
        for order in range(args.min_order, args.max_order + 1):
            results.append(
                evaluate_accuracy(
                    spec=spec,
                    order=order,
                    num_points=args.num_points,
                )
            )

    print_results(results)
    print_summary(results)

    write_results_csv(args.output_csv, results)
    plot_functions(
        function_names=args.functions,
        function_specs=function_specs,
        min_order=args.min_order,
        max_order=args.max_order,
        num_points=args.num_points,
        output_path=args.output_plot,
        dpi=args.plot_dpi,
    )

    print()
    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved plot to {args.output_plot}")


if __name__ == "__main__":
    main()
