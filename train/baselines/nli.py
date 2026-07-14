from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit, prange, set_num_threads


TAU_FP16_NORMAL = 2.0 ** -14

KIND_EXP = 0
KIND_EXP2 = 1
KIND_SIGMOID = 2
KIND_ERF_SHIFT = 3
KIND_NEG_RECIPROCAL = 4
KIND_SIN_SHIFT = 5
KIND_TANH_SHIFT = 6
KIND_SOFTSIGN_SHIFT = 7
KIND_ARCTAN_SHIFT = 8
KIND_SILU = 9
KIND_GELU = 10
KIND_MISH = 11
KIND_HARDSWISH = 12
KIND_RECIPROCAL = 13
KIND_RSQRT = 14


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    kind: int
    default_l_range: float
    default_r_range: float
    description: str


FUNCTION_SPECS: dict[str, FunctionSpec] = {
    "exp": FunctionSpec("exp", KIND_EXP, -256.0, 0.0, "exp(x)"),
    "exp2": FunctionSpec("exp2", KIND_EXP2, -256.0, 0.0, "2^x"),
    "sigmoid": FunctionSpec("sigmoid", KIND_SIGMOID, -6.234, 0.0, "sigmoid(x)"),
    "erf": FunctionSpec("erf", KIND_ERF_SHIFT, -2.469, 0.0, "erf(x) + 1"),
    "rsqrt": FunctionSpec("rsqrt", KIND_NEG_RECIPROCAL, -1024.0, -0.1, "-(1 / x) remapping"),
    "sin": FunctionSpec("sin", KIND_SIN_SHIFT, -math.pi / 2, 0.0, "sin(x) + 1"),
    "tanh": FunctionSpec("tanh", KIND_TANH_SHIFT, -3.465, 0.0, "tanh(x) + 1"),
    "softsign": FunctionSpec("softsign", KIND_SOFTSIGN_SHIFT, -128.0, 0.0, "softsign(x) + 1"),
    "arctan": FunctionSpec("arctan", KIND_ARCTAN_SHIFT, -128.0, 0.0, "atan(x) + pi/2"),
    "silu": FunctionSpec("silu", KIND_SILU, -150.0, 150.0, "SiLU(x)"),
    "gelu": FunctionSpec("gelu", KIND_GELU, -16.0, 16.0, "GELU(x)"),
    "mish": FunctionSpec("mish", KIND_MISH, -16.0, 16.0, "Mish(x)"),
    "hardswish": FunctionSpec("hardswish", KIND_HARDSWISH, -16.0, 16.0, "HardSwish(x)"),
    "reciprocal": FunctionSpec("reciprocal", KIND_RECIPROCAL, 2.0 ** -16, 65504.0, "1 / x"),
    "paper_rsqrt": FunctionSpec(
        "paper_rsqrt",
        KIND_RSQRT,
        5.9604645e-08,
        65504.0,
        "1 / sqrt(x)",
    ),
}


@dataclass
class Metrics:
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    max_rel_error: float
    mean_rel_error: float
    objective_sum: float
    objective_mean: float
    x_at_max_abs_error: float


@dataclass
class SearchResult:
    coarse_grid_size: int
    coarse_objective: float
    full_objective_before_refine: float
    full_objective_after_refine: float
    macro_full_indices: list[int]
    macro_x: list[float]
    lut_x: list[float]
    lut_y: list[float]
    interval_bins: list[int]
    interval_scales: list[float]
    interval_bases: list[int]
    metrics: Metrics
    runtime_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the NLI macro-cutpoint search and 259-point LUT construction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--func",
        choices=sorted(FUNCTION_SPECS),
        default="exp",
        help="Target scalar function.",
    )
    parser.add_argument("--l-range", type=float, default=None, help="Override left domain boundary.")
    parser.add_argument("--r-range", type=float, default=None, help="Override right domain boundary.")
    parser.add_argument(
        "--macro-cutpoints",
        type=int,
        default=11,
        help="Number of macro cutpoints. The paper uses 11.",
    )
    parser.add_argument(
        "--micro-bins",
        type=int,
        default=32,
        help="Uniform bins inside each non-edge macro interval. The paper uses 32.",
    )
    parser.add_argument(
        "--search-budget",
        type=int,
        default=257,
        help="Coarse FP16-grid budget used for the exact DP stage. Higher values are slower but can help harder functions. Use a value >= the full grid size for exact full-grid DP.",
    )
    parser.add_argument(
        "--refine-radii",
        type=str,
        default="256,64,16,4",
        help="Comma-separated local-search radii on the full FP16 grid.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU_FP16_NORMAL,
        help="Relative-error denominator floor.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to train/baselines/runs/<timestamp>_<func>_nli.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render a function/error plot into the output directory.",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip full-grid local refinement after coarse DP.",
    )
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=1,
        help="Thread count used by numba's parallel kernels. Keep this low on constrained systems.",
    )
    return parser.parse_args()


def get_function_spec(name: str) -> FunctionSpec:
    try:
        return FUNCTION_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported function: {name}") from exc


def apply_function_defaults(args: argparse.Namespace) -> None:
    spec = get_function_spec(args.func)
    if args.l_range is None:
        args.l_range = spec.default_l_range
    if args.r_range is None:
        args.r_range = spec.default_r_range


def validate_args(args: argparse.Namespace) -> None:
    if args.l_range >= args.r_range:
        raise ValueError("--l-range must be smaller than --r-range.")
    if args.macro_cutpoints < 2:
        raise ValueError("--macro-cutpoints must be at least 2.")
    if args.micro_bins < 1:
        raise ValueError("--micro-bins must be positive.")
    if args.search_budget < args.macro_cutpoints:
        raise ValueError("--search-budget must be at least --macro-cutpoints.")
    if args.tau <= 0.0:
        raise ValueError("--tau must be positive.")
    if args.numba_threads < 1:
        raise ValueError("--numba-threads must be at least 1.")


def parse_radii(text: str) -> list[int]:
    radii: list[int] = []
    for chunk in text.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        radius = int(stripped)
        if radius < 0:
            raise ValueError(f"Refine radii must be non-negative, got {radius}.")
        radii.append(radius)
    return radii


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (Path("train") / "baselines" / "runs" / f"{stamp}_{args.func}_nli").resolve()


def enumerate_fp16_domain(l_range: float, r_range: float) -> np.ndarray:
    raw = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    finite = raw[np.isfinite(raw)].astype(np.float32)
    unique = np.unique(finite)
    mask = (unique >= l_range) & (unique <= r_range)
    domain = unique[mask].astype(np.float64)
    if len(domain) < 2:
        raise ValueError(
            f"FP16 domain in [{l_range}, {r_range}] resolves to fewer than two points; cannot build interpolation."
        )
    return domain


def evaluate_target(kind: int, x: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(x.astype(np.float64))

    if kind == KIND_EXP:
        values = torch.exp(tensor)
    elif kind == KIND_EXP2:
        values = torch.exp2(tensor)
    elif kind == KIND_SIGMOID:
        values = torch.sigmoid(tensor)
    elif kind == KIND_ERF_SHIFT:
        values = torch.erf(tensor) + 1.0
    elif kind == KIND_NEG_RECIPROCAL:
        values = -torch.reciprocal(tensor)
    elif kind == KIND_SIN_SHIFT:
        values = torch.sin(tensor) + 1.0
    elif kind == KIND_TANH_SHIFT:
        values = torch.tanh(tensor) + 1.0
    elif kind == KIND_SOFTSIGN_SHIFT:
        values = F.softsign(tensor) + 1.0
    elif kind == KIND_ARCTAN_SHIFT:
        values = torch.atan(tensor) + tensor.new_tensor(math.pi / 2.0)
    elif kind == KIND_SILU:
        values = F.silu(tensor)
    elif kind == KIND_GELU:
        values = 0.5 * tensor * (1.0 + torch.erf(tensor / math.sqrt(2.0)))
    elif kind == KIND_MISH:
        values = tensor * torch.tanh(F.softplus(tensor))
    elif kind == KIND_HARDSWISH:
        values = tensor * torch.clamp(tensor + 3.0, 0.0, 6.0) / 6.0
    elif kind == KIND_RECIPROCAL:
        values = torch.reciprocal(tensor)
    elif kind == KIND_RSQRT:
        values = torch.rsqrt(tensor)
    else:
        raise ValueError(f"Unhandled function kind: {kind}")

    result = values.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("Target evaluation produced non-finite values. Check the function domain.")
    return result


@njit(cache=True)
def eval_function_scalar(kind: int, x: float) -> float:
    if kind == KIND_EXP:
        return math.exp(x)
    if kind == KIND_EXP2:
        return 2.0 ** x
    if kind == KIND_SIGMOID:
        if x >= 0.0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
    if kind == KIND_ERF_SHIFT:
        return math.erf(x) + 1.0
    if kind == KIND_NEG_RECIPROCAL:
        return -(1.0 / x)
    if kind == KIND_SIN_SHIFT:
        return math.sin(x) + 1.0
    if kind == KIND_TANH_SHIFT:
        return math.tanh(x) + 1.0
    if kind == KIND_SOFTSIGN_SHIFT:
        return x / (1.0 + abs(x)) + 1.0
    if kind == KIND_ARCTAN_SHIFT:
        return math.atan(x) + math.pi / 2.0
    if kind == KIND_SILU:
        if x >= 0.0:
            z = math.exp(-x)
            return x / (1.0 + z)
        z = math.exp(x)
        return x * z / (1.0 + z)
    if kind == KIND_GELU:
        return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
    if kind == KIND_MISH:
        if x > 20.0:
            softplus = x
        elif x < -20.0:
            softplus = math.exp(x)
        else:
            softplus = math.log1p(math.exp(x))
        return x * math.tanh(softplus)
    if kind == KIND_HARDSWISH:
        gate = x + 3.0
        if gate < 0.0:
            gate = 0.0
        elif gate > 6.0:
            gate = 6.0
        return x * gate / 6.0
    if kind == KIND_RECIPROCAL:
        return 1.0 / x
    if kind == KIND_RSQRT:
        return 1.0 / math.sqrt(x)
    raise ValueError(kind)


@njit(cache=True)
def segment_cost_uniform(
    x: np.ndarray,
    y: np.ndarray,
    denom: np.ndarray,
    i: int,
    k: int,
    bins: int,
    func_kind: int,
) -> float:
    left = x[i]
    right = x[k]
    total = 0.0

    if bins <= 1:
        inv_width = 1.0 / (right - left)
        delta_y = y[k] - y[i]
        for j in range(i, k + 1):
            t = (x[j] - left) * inv_width
            pred = y[i] + t * delta_y
            total += abs(y[j] - pred) / denom[j]
        return total / (k - i + 1)

    step = (right - left) / bins
    inv_step = 1.0 / step
    cut_y = np.empty(bins + 1, dtype=np.float64)
    for b in range(bins + 1):
        cut_y[b] = eval_function_scalar(func_kind, left + step * b)

    for j in range(i, k + 1):
        pos = (x[j] - left) * inv_step
        addr = int(pos)
        if addr < 0:
            addr = 0
            frac = 0.0
        elif addr >= bins:
            addr = bins - 1
            frac = 1.0
        else:
            frac = pos - addr
        pred = cut_y[addr] + frac * (cut_y[addr + 1] - cut_y[addr])
        total += abs(y[j] - pred) / denom[j]
    return total / (k - i + 1)


@njit(cache=True, parallel=True)
def build_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    denom: np.ndarray,
    bins: int,
    func_kind: int,
) -> np.ndarray:
    size = len(x)
    costs = np.full((size, size), np.inf, dtype=np.float64)
    for i in prange(size - 1):
        for k in range(i + 1, size):
            costs[i, k] = segment_cost_uniform(x, y, denom, i, k, bins, func_kind)
    return costs


@njit(cache=True)
def dp_search_fixed_endpoints(
    edge_cost: np.ndarray,
    middle_cost: np.ndarray,
    macro_cutpoints: int,
) -> tuple[np.ndarray, float]:
    size = edge_cost.shape[0]
    interior = macro_cutpoints - 2

    if macro_cutpoints > size:
        raise ValueError("macro_cutpoints exceeds candidate-grid size")

    if interior == 0:
        out = np.empty(2, dtype=np.int64)
        out[0] = 0
        out[1] = size - 1
        return out, edge_cost[0, size - 1]

    dp = np.full((interior + 1, size), np.inf, dtype=np.float64)
    prev = np.full((interior + 1, size), -1, dtype=np.int64)

    max_first = (size - 2) - (interior - 1)
    for k in range(1, max_first + 1):
        dp[1, k] = edge_cost[0, k]
        prev[1, k] = 0

    for level in range(2, interior + 1):
        max_k = (size - 2) - (interior - level)
        for k in range(level, max_k + 1):
            best = math.inf
            arg = -1
            for i in range(level - 1, k):
                val = dp[level - 1, i] + middle_cost[i, k]
                if val < best:
                    best = val
                    arg = i
            dp[level, k] = best
            prev[level, k] = arg

    best_total = math.inf
    best_last = -1
    for i in range(interior, size - 1):
        val = dp[interior, i] + edge_cost[i, size - 1]
        if val < best_total:
            best_total = val
            best_last = i

    path = np.empty(macro_cutpoints, dtype=np.int64)
    path[0] = 0
    path[macro_cutpoints - 1] = size - 1
    current = best_last
    for level in range(interior, 0, -1):
        path[level] = current
        current = prev[level, current]
    return path, best_total


def choose_coarse_indices(full_size: int, budget: int) -> np.ndarray:
    if budget >= full_size:
        return np.arange(full_size, dtype=np.int64)
    raw = np.linspace(0, full_size - 1, num=budget)
    indices = np.unique(np.rint(raw).astype(np.int64))
    if indices[0] != 0:
        indices = np.concatenate([np.array([0], dtype=np.int64), indices])
    if indices[-1] != full_size - 1:
        indices = np.concatenate([indices, np.array([full_size - 1], dtype=np.int64)])
    return indices


def macro_objective(
    x: np.ndarray,
    y: np.ndarray,
    denom: np.ndarray,
    macro_indices: np.ndarray,
    micro_bins: int,
    func_kind: int,
) -> float:
    total = 0.0
    num_intervals = len(macro_indices) - 1
    for interval_idx in range(num_intervals):
        bins = 1 if interval_idx == 0 or interval_idx == num_intervals - 1 else micro_bins
        total += segment_cost_uniform(
            x,
            y,
            denom,
            int(macro_indices[interval_idx]),
            int(macro_indices[interval_idx + 1]),
            bins,
            func_kind,
        )
    return float(total)


def refine_macro_indices(
    x: np.ndarray,
    y: np.ndarray,
    denom: np.ndarray,
    macro_indices: np.ndarray,
    micro_bins: int,
    func_kind: int,
    radii: Iterable[int],
) -> np.ndarray:
    refined = macro_indices.copy()
    num_intervals = len(refined) - 1

    for radius in radii:
        if radius == 0:
            continue
        any_change = True
        while any_change:
            any_change = False
            for cutpoint_idx in range(1, len(refined) - 1):
                prev_idx = int(refined[cutpoint_idx - 1])
                curr_idx = int(refined[cutpoint_idx])
                next_idx = int(refined[cutpoint_idx + 1])

                left_bins = 1 if cutpoint_idx - 1 == 0 else micro_bins
                right_bins = 1 if cutpoint_idx == num_intervals - 1 else micro_bins

                lo = max(prev_idx + 1, curr_idx - radius)
                hi = min(next_idx - 1, curr_idx + radius)
                best_idx = curr_idx
                best_cost = (
                    segment_cost_uniform(x, y, denom, prev_idx, curr_idx, left_bins, func_kind)
                    + segment_cost_uniform(x, y, denom, curr_idx, next_idx, right_bins, func_kind)
                )

                for candidate in range(lo, hi + 1):
                    cost = (
                        segment_cost_uniform(x, y, denom, prev_idx, candidate, left_bins, func_kind)
                        + segment_cost_uniform(x, y, denom, candidate, next_idx, right_bins, func_kind)
                    )
                    if cost + 1e-15 < best_cost:
                        best_cost = cost
                        best_idx = candidate

                if best_idx != curr_idx:
                    refined[cutpoint_idx] = best_idx
                    any_change = True
    return refined


def build_uniform_lut(
    macro_x: np.ndarray,
    macro_cutpoints: int,
    micro_bins: int,
    func_kind: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_intervals = macro_cutpoints - 1
    interval_bins = np.array(
        [1 if idx == 0 or idx == num_intervals - 1 else micro_bins for idx in range(num_intervals)],
        dtype=np.int64,
    )

    bases = np.empty(num_intervals, dtype=np.int64)
    bases[0] = 0
    for idx in range(1, num_intervals):
        bases[idx] = bases[idx - 1] + interval_bins[idx - 1]

    lut_x: list[float] = [float(macro_x[0]), float(macro_x[1])]
    for interval_idx in range(1, num_intervals - 1):
        left = float(macro_x[interval_idx])
        right = float(macro_x[interval_idx + 1])
        step = (right - left) / micro_bins
        for step_idx in range(1, micro_bins + 1):
            lut_x.append(left + step * step_idx)
    lut_x.append(float(macro_x[-1]))

    lut_x_array = np.array(lut_x, dtype=np.float64)
    lut_y_array = evaluate_target(func_kind, lut_x_array)
    scales = interval_bins.astype(np.float64) / (macro_x[1:] - macro_x[:-1])
    return lut_x_array, lut_y_array, interval_bins, scales, bases


def evaluate_lut(
    x: np.ndarray,
    macro_x: np.ndarray,
    lut_y: np.ndarray,
    interval_bins: np.ndarray,
    interval_scales: np.ndarray,
    interval_bases: np.ndarray,
) -> np.ndarray:
    num_intervals = len(macro_x) - 1
    clipped = np.clip(x, macro_x[0], macro_x[-1])
    interval_idx = np.searchsorted(macro_x, clipped, side="right") - 1
    interval_idx = np.clip(interval_idx, 0, num_intervals - 1)

    left = macro_x[interval_idx]
    bins = interval_bins[interval_idx]
    scales = interval_scales[interval_idx]
    bases = interval_bases[interval_idx]

    position = (clipped - left) * scales
    address = np.floor(position).astype(np.int64)
    address = np.clip(address, 0, bins - 1)
    decimal = position - address

    global_index = bases + address
    left_values = lut_y[global_index]
    right_values = lut_y[global_index + 1]
    return left_values + decimal * (right_values - left_values)


def compute_metrics(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    denom: np.ndarray,
    macro_indices: np.ndarray,
    micro_bins: int,
    func_kind: int,
) -> Metrics:
    abs_error = np.abs(y_true - y_pred)
    rel_error = abs_error / denom
    objective_sum = macro_objective(x, y_true, denom, macro_indices, micro_bins, func_kind)
    objective_mean = objective_sum / (len(macro_indices) - 1)
    max_idx = int(np.argmax(abs_error))

    return Metrics(
        max_abs_error=float(abs_error[max_idx]),
        mean_abs_error=float(abs_error.mean()),
        rmse=float(np.sqrt(np.mean(np.square(abs_error)))),
        max_rel_error=float(rel_error.max()),
        mean_rel_error=float(rel_error.mean()),
        objective_sum=float(objective_sum),
        objective_mean=float(objective_mean),
        x_at_max_abs_error=float(x[max_idx]),
    )


def render_plot(
    output_path: Path,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    macro_x: np.ndarray,
    lut_x: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abs_error = np.abs(y_true - y_pred)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)
    axes[0].plot(x, y_true, color="#2563eb", linewidth=1.8, label="Reference")
    axes[0].plot(x, y_pred, color="#ea580c", linewidth=1.4, label="NLI")
    axes[0].scatter(lut_x, evaluate_target(get_function_spec(title).kind, lut_x), s=8, color="#dc2626", alpha=0.45)
    axes[0].scatter(macro_x, evaluate_target(get_function_spec(title).kind, macro_x), s=24, color="#111827", label="Macro cutpoints")
    axes[0].set_ylabel("Output")
    axes[0].legend(loc="best")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    axes[1].plot(x, abs_error, color="#16a34a", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Input")
    axes[1].set_ylabel("|error|")
    axes[1].grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)
    axes[1].grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)

    fig.suptitle(f"NLI approximation for {title}")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    spec: FunctionSpec,
    full_grid_size: int,
    result: SearchResult,
) -> None:
    payload = {
        "args": {
            "func": args.func,
            "l_range": args.l_range,
            "r_range": args.r_range,
            "macro_cutpoints": args.macro_cutpoints,
            "micro_bins": args.micro_bins,
            "search_budget": args.search_budget,
            "refine_radii": parse_radii(args.refine_radii),
            "tau": args.tau,
            "plot": bool(args.plot),
            "no_refine": bool(args.no_refine),
            "numba_threads": args.numba_threads,
        },
        "function": asdict(spec),
        "full_grid_size": full_grid_size,
        "result": {
            **asdict(result),
            "metrics": asdict(result.metrics),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    metrics = result.metrics
    report_lines = [
        f"# NLI baseline result: {args.func}",
        "",
        "## Setup",
        "",
        f"- Function: `{spec.description}`",
        f"- Domain: `[{args.l_range}, {args.r_range}]`",
        f"- Full FP16 grid size: `{full_grid_size}`",
        f"- Coarse DP grid size: `{result.coarse_grid_size}`",
        f"- Macro cutpoints: `{args.macro_cutpoints}`",
        f"- Micro bins per middle interval: `{args.micro_bins}`",
        f"- Runtime: `{result.runtime_seconds:.2f}` seconds",
        "",
        "## Objective",
        "",
        f"- Coarse-grid DP objective: `{result.coarse_objective:.8e}`",
        f"- Full-grid objective before refinement: `{result.full_objective_before_refine:.8e}`",
        f"- Full-grid objective after refinement: `{result.full_objective_after_refine:.8e}`",
        "",
        "## Error metrics",
        "",
        f"- Max abs error: `{metrics.max_abs_error:.8e}` at `x = {metrics.x_at_max_abs_error:.8f}`",
        f"- Mean abs error: `{metrics.mean_abs_error:.8e}`",
        f"- RMSE: `{metrics.rmse:.8e}`",
        f"- Max rel error: `{metrics.max_rel_error:.8e}`",
        f"- Mean rel error: `{metrics.mean_rel_error:.8e}`",
        "",
        "## Macro cutpoints",
        "",
        ", ".join(f"`{value:.10g}`" for value in result.macro_x),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    artifact_payload = {
        "backend": "nli",
        "target": spec.name,
        "description": spec.description,
        "l_range": args.l_range,
        "r_range": args.r_range,
        "macro_cutpoints": args.macro_cutpoints,
        "micro_bins": args.micro_bins,
        "lut_entries": len(result.lut_y),
        "macro_x": result.macro_x,
        "lut_x": result.lut_x,
        "lut_y": result.lut_y,
        "interval_bins": result.interval_bins,
        "interval_scales": result.interval_scales,
        "interval_bases": result.interval_bases,
        "metrics": asdict(result.metrics),
        "source_summary": "summary.json",
    }
    with (output_dir / "artifact.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact_payload, handle, indent=2)


def run_search(args: argparse.Namespace) -> SearchResult:
    spec = get_function_spec(args.func)
    start_time = time.perf_counter()

    x_full = enumerate_fp16_domain(args.l_range, args.r_range)
    y_full = evaluate_target(spec.kind, x_full)
    denom_full = np.maximum(np.abs(y_full), args.tau)

    coarse_full_indices = choose_coarse_indices(len(x_full), args.search_budget)
    x_coarse = x_full[coarse_full_indices]
    y_coarse = y_full[coarse_full_indices]
    denom_coarse = denom_full[coarse_full_indices]

    edge_cost = build_cost_matrix(x_coarse, y_coarse, denom_coarse, 1, spec.kind)
    middle_cost = build_cost_matrix(x_coarse, y_coarse, denom_coarse, args.micro_bins, spec.kind)
    coarse_path, coarse_objective = dp_search_fixed_endpoints(edge_cost, middle_cost, args.macro_cutpoints)

    macro_full_indices = coarse_full_indices[coarse_path]
    before_refine = macro_objective(x_full, y_full, denom_full, macro_full_indices, args.micro_bins, spec.kind)

    if args.no_refine:
        refined_indices = macro_full_indices.copy()
    else:
        refined_indices = refine_macro_indices(
            x_full,
            y_full,
            denom_full,
            macro_full_indices,
            args.micro_bins,
            spec.kind,
            parse_radii(args.refine_radii),
        )

    after_refine = macro_objective(x_full, y_full, denom_full, refined_indices, args.micro_bins, spec.kind)
    macro_x = x_full[refined_indices]
    lut_x, lut_y, interval_bins, interval_scales, interval_bases = build_uniform_lut(
        macro_x,
        args.macro_cutpoints,
        args.micro_bins,
        spec.kind,
    )
    y_pred = evaluate_lut(x_full, macro_x, lut_y, interval_bins, interval_scales, interval_bases)
    metrics = compute_metrics(x_full, y_full, y_pred, denom_full, refined_indices, args.micro_bins, spec.kind)

    runtime_seconds = time.perf_counter() - start_time
    return SearchResult(
        coarse_grid_size=int(len(x_coarse)),
        coarse_objective=float(coarse_objective),
        full_objective_before_refine=float(before_refine),
        full_objective_after_refine=float(after_refine),
        macro_full_indices=[int(value) for value in refined_indices.tolist()],
        macro_x=[float(value) for value in macro_x.tolist()],
        lut_x=[float(value) for value in lut_x.tolist()],
        lut_y=[float(value) for value in lut_y.tolist()],
        interval_bins=[int(value) for value in interval_bins.tolist()],
        interval_scales=[float(value) for value in interval_scales.tolist()],
        interval_bases=[int(value) for value in interval_bases.tolist()],
        metrics=metrics,
        runtime_seconds=float(runtime_seconds),
    )


def main() -> int:
    args = parse_args()
    apply_function_defaults(args)
    validate_args(args)
    set_num_threads(args.numba_threads)

    spec = get_function_spec(args.func)
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_search(args)
    full_grid = enumerate_fp16_domain(args.l_range, args.r_range)
    y_full = evaluate_target(spec.kind, full_grid)
    macro_x = np.array(result.macro_x, dtype=np.float64)
    lut_y = np.array(result.lut_y, dtype=np.float64)
    interval_bins = np.array(result.interval_bins, dtype=np.int64)
    interval_scales = np.array(result.interval_scales, dtype=np.float64)
    interval_bases = np.array(result.interval_bases, dtype=np.int64)
    y_pred = evaluate_lut(full_grid, macro_x, lut_y, interval_bins, interval_scales, interval_bases)

    write_summary(output_dir, args, spec, len(full_grid), result)

    if args.plot:
        render_plot(
            output_dir / "fit.png",
            full_grid,
            y_full,
            y_pred,
            macro_x,
            np.array(result.lut_x, dtype=np.float64),
            args.func,
        )

    print(f"output_dir={output_dir}")
    print(f"function={args.func}")
    print(f"full_grid_size={len(full_grid)}")
    print(f"coarse_grid_size={result.coarse_grid_size}")
    print(f"max_abs_error={result.metrics.max_abs_error:.8e}")
    print(f"mean_abs_error={result.metrics.mean_abs_error:.8e}")
    print(f"rmse={result.metrics.rmse:.8e}")
    print(f"objective_after_refine={result.full_objective_after_refine:.8e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
