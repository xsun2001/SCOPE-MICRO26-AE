from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Keep NumPy/SciPy imports single-threaded by default so repeated baseline runs do
# not hit process/thread limits on shared machines.
for env_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

import matplotlib
import numpy as np
from scipy import special

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACT_FUNCS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "swish": lambda x: x / (1.0 + np.exp(-x)),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": lambda x: np.tanh(x),
    "gelu": lambda x: 0.5 * x * (1.0 + special.erf(x / np.sqrt(2.0))),
    "hswish": lambda x: x * np.clip(x + 3.0, 0.0, 6.0) / 6.0,
    "exp": np.exp,
    "reci": np.reciprocal,
    "sqrt_reci": lambda x: np.reciprocal(np.sqrt(x)),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
}

ACT_ALIASES = {
    "div": "reci",
    "rsqrt": "sqrt_reci",
    "hardswish": "hswish",
}


@dataclass(frozen=True)
class PaperPreset:
    act_func: str
    entries: int
    x_range: tuple[float, float]
    sp_range: tuple[float, float]
    neg_inf: float
    pos_inf: float
    offset: int
    coeff_bit_width: int = 8
    decimal_bit_range: tuple[int, int] = (0, 6)
    total_iters: int = 500
    pop_size: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2


PAPER_PRESETS: dict[tuple[str, int], PaperPreset] = {
    ("gelu", 8): PaperPreset("gelu", 8, (-4.0, 4.0), (-4.0, 4.0), -10000.0, 10000.0, 2),
    ("gelu", 16): PaperPreset("gelu", 16, (-4.0, 4.0), (-4.0, 4.0), -10000.0, 10000.0, 0),
    ("hswish", 8): PaperPreset("hswish", 8, (-4.0, 4.0), (-4.0, 4.0), -10000.0, 10000.0, 0),
    ("hswish", 16): PaperPreset("hswish", 16, (-4.0, 4.0), (-4.0, 4.0), -10000.0, 10000.0, 2),
    ("exp", 8): PaperPreset("exp", 8, (-8.0, 0.0), (-8.0, 0.0), -16.0, 0.0, 0),
    ("exp", 16): PaperPreset("exp", 16, (-8.0, 0.0), (-6.5, 0.0), -16.0, 0.0, 0),
    ("reci", 8): PaperPreset("reci", 8, (0.5, 4.0), (0.5, 4.0), 0.5, 4.0, 0),
    ("reci", 16): PaperPreset("reci", 16, (0.5, 4.0), (0.5, 4.0), 0.5, 4.0, 0),
    ("sqrt_reci", 8): PaperPreset("sqrt_reci", 8, (0.25, 4.0), (0.25, 4.0), 0.25, 4.0, 0),
    ("sqrt_reci", 16): PaperPreset("sqrt_reci", 16, (0.25, 4.0), (0.25, 4.0), 0.25, 4.0, 0),
}


@dataclass(frozen=True)
class SearchConfig:
    act_func: str
    x_range: tuple[float, float]
    sp_range: tuple[float, float]
    num_splits: int
    total_iters: int
    decimal_bit_range: tuple[int, int]
    coeff_bit_width: int
    pop_size: int
    crossover_prob: float
    mutation_prob: float
    mutation_sigma: float
    mutation_mode: str
    offset: int
    neg_inf: float
    pos_inf: float
    eval_step: float
    seed: int
    output_dir: str | None
    run_name: str | None
    compare_reference: str | None


@dataclass
class SearchResult:
    best_individual: np.ndarray
    best_split_points: np.ndarray
    slopes: np.ndarray
    intercepts: np.ndarray
    best_mse: float
    history: list[dict[str, float]]


def canonical_act_func(name: str) -> str:
    return ACT_ALIASES.get(name.lower(), name.lower())


def target_function(act_func: str) -> Callable[[np.ndarray], np.ndarray]:
    try:
        return ACT_FUNCS[act_func]
    except KeyError as exc:
        supported = ", ".join(sorted(ACT_FUNCS))
        raise ValueError(f"Unsupported act_func {act_func!r}. Supported: {supported}") from exc


def round_to_nearest_bits(values: np.ndarray | float, decimal_bits: int) -> np.ndarray | float:
    scale = 2**decimal_bits
    return np.round(np.asarray(values) * scale) / scale


def calculate_coeff_bias(a1: np.ndarray, a2: np.ndarray, act_func: str, coeff_bit_width: int) -> tuple[np.ndarray, np.ndarray]:
    func = target_function(act_func)
    y1 = func(a1)
    y2 = func(a2)
    delta = a2 - a1
    coeff = np.divide(y2 - y1, delta, out=np.zeros_like(delta), where=delta != 0.0)
    bias = -a1 * coeff + y1
    resize = 2 ** (coeff_bit_width - 2)
    coeff = np.round(coeff * resize) / resize
    bias = np.round(bias * resize) / resize
    return coeff, bias


def build_segment_parameters(split_points: np.ndarray, act_func: str, coeff_bit_width: int) -> tuple[np.ndarray, np.ndarray]:
    left = split_points[:, :-1]
    right = split_points[:, 1:]
    return calculate_coeff_bias(left, right, act_func, coeff_bit_width)


def evaluate_population(
    population: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    act_func: str,
    neg_inf: float,
    pos_inf: float,
    coeff_bit_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sorted_population = np.sort(population, axis=1)
    split_points = np.concatenate(
        [
            np.full((len(population), 1), neg_inf, dtype=np.float64),
            sorted_population,
            np.full((len(population), 1), pos_inf, dtype=np.float64),
        ],
        axis=1,
    )
    slopes, intercepts = build_segment_parameters(split_points, act_func, coeff_bit_width)
    indices = (x_values[None, :, None] >= split_points[:, None, 1:]).sum(axis=2)
    approx = np.take_along_axis(slopes, indices, axis=1) * x_values[None, :] + np.take_along_axis(
        intercepts, indices, axis=1
    )
    mse = np.mean((y_values[None, :] - approx) ** 2, axis=1)
    return mse, sorted_population, split_points, slopes, intercepts


def mutate_individual(
    individual: np.ndarray,
    *,
    decimal_bit_range: tuple[int, int],
    sp_range: tuple[float, float],
    rng: np.random.RandomState,
    offset: int,
    mutation_sigma: float,
    mutation_mode: str,
) -> np.ndarray:
    mutated = individual.copy()
    min_bit, max_bit = decimal_bit_range

    if mutation_mode == "gaussian":
        for idx in range(len(mutated)):
            if rng.random() >= 0.9:
                mutated[idx] += rng.normal(0.0, mutation_sigma)
                mutated[idx] = np.clip(mutated[idx], sp_range[0], sp_range[1])
        return mutated

    for idx in range(len(mutated)):
        for decimal_bits in range(min_bit + offset, max_bit + 1):
            p = rng.random()
            if 0.05 * decimal_bits <= p < 0.05 * (decimal_bits + 1):
                mutated[idx] = float(round_to_nearest_bits(mutated[idx], decimal_bits))
                mutated[idx] = np.clip(mutated[idx], sp_range[0], sp_range[1])
            elif p >= 0.9:
                mutated[idx] += rng.normal(0.0, mutation_sigma)
        mutated[idx] = np.clip(mutated[idx], sp_range[0], sp_range[1])

    return mutated


def two_point_crossover(left: np.ndarray, right: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if len(left) < 3:
        return left.copy(), right.copy()

    points = sorted(rng.sample(range(1, len(left)), 2))
    start, end = points[0], points[1]
    out_left = left.copy()
    out_right = right.copy()
    out_left[start:end], out_right[start:end] = right[start:end].copy(), left[start:end].copy()
    return out_left, out_right


def tournament_select(population: np.ndarray, fitness: np.ndarray, *, tournsize: int, rng: random.Random) -> np.ndarray:
    selected = np.empty_like(population)
    pop_size = len(population)
    for idx in range(pop_size):
        contestants = [rng.randrange(pop_size) for _ in range(tournsize)]
        winner = min(contestants, key=lambda item: fitness[item])
        selected[idx] = population[winner]
    return selected


def run_search(config: SearchConfig) -> SearchResult:
    np_rng = np.random.RandomState(config.seed)
    py_rng = random.Random(config.seed)
    func = target_function(config.act_func)
    x_values = np.arange(config.x_range[0], config.x_range[1], config.eval_step, dtype=np.float64)
    y_values = func(x_values)

    population = np_rng.uniform(
        low=config.sp_range[0],
        high=config.sp_range[1],
        size=(config.pop_size, config.num_splits),
    ).astype(np.float64, copy=False)

    history: list[dict[str, float]] = []
    final_sorted_population: np.ndarray | None = None
    final_split_points: np.ndarray | None = None
    final_slopes: np.ndarray | None = None
    final_intercepts: np.ndarray | None = None
    final_fitness: np.ndarray | None = None

    for generation in range(config.total_iters + 1):
        fitness, sorted_population, split_points, slopes, intercepts = evaluate_population(
            population,
            x_values,
            y_values,
            act_func=config.act_func,
            neg_inf=config.neg_inf,
            pos_inf=config.pos_inf,
            coeff_bit_width=config.coeff_bit_width,
        )
        best_index = int(np.argmin(fitness))
        history.append(
            {
                "generation": float(generation),
                "best_mse": float(fitness[best_index]),
                "mean_mse": float(np.mean(fitness)),
                "worst_mse": float(np.max(fitness)),
            }
        )
        final_sorted_population = sorted_population
        final_split_points = split_points
        final_slopes = slopes
        final_intercepts = intercepts
        final_fitness = fitness

        if generation == config.total_iters:
            break

        offspring = tournament_select(population, fitness, tournsize=3, rng=py_rng)

        for pair_start in range(0, len(offspring) - 1, 2):
            if py_rng.random() < config.crossover_prob:
                left, right = two_point_crossover(offspring[pair_start], offspring[pair_start + 1], py_rng)
                offspring[pair_start] = left
                offspring[pair_start + 1] = right

        for idx in range(len(offspring)):
            if py_rng.random() < config.mutation_prob:
                offspring[idx] = mutate_individual(
                    offspring[idx],
                    decimal_bit_range=config.decimal_bit_range,
                    sp_range=config.sp_range,
                    rng=np_rng,
                    offset=config.offset,
                    mutation_sigma=config.mutation_sigma,
                    mutation_mode=config.mutation_mode,
                )

        population = offspring

    assert final_sorted_population is not None
    assert final_split_points is not None
    assert final_slopes is not None
    assert final_intercepts is not None
    assert final_fitness is not None

    best_index = int(np.argmin(final_fitness))
    return SearchResult(
        best_individual=final_sorted_population[best_index].copy(),
        best_split_points=final_split_points[best_index].copy(),
        slopes=final_slopes[best_index].copy(),
        intercepts=final_intercepts[best_index].copy(),
        best_mse=float(final_fitness[best_index]),
        history=history,
    )


def build_decimal_bit_tables(
    result: SearchResult,
    *,
    act_func: str,
    decimal_bit_range: tuple[int, int],
) -> dict[str, dict[str, list[float]]]:
    min_bit, max_bit = decimal_bit_range
    tables: dict[str, dict[str, list[float]]] = {}
    for bit in range(min_bit, max_bit + 1):
        rounded_splits = round_to_nearest_bits(result.best_split_points, bit)
        tables[str(bit)] = {
            "breakpoints": rounded_splits[1:-1].astype(float).tolist(),
            "slopes": result.slopes.astype(float).tolist(),
            "intercepts": result.intercepts.astype(float).tolist(),
        }
    return {act_func: tables}


def piecewise_linear_approximation(
    x_values: np.ndarray,
    split_points: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    indices = np.digitize(x_values, split_points) - 1
    indices = np.clip(indices, 0, len(slopes) - 1)
    return slopes[indices] * x_values + intercepts[indices]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    mse = float(np.mean(diff**2))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
    }


def evaluate_decimal_bit_tables(
    tables: dict[str, dict[str, list[float]]],
    *,
    act_func: str,
    x_range: tuple[float, float],
    eval_step: float,
    neg_inf: float,
    pos_inf: float,
) -> dict[str, dict[str, dict[str, float] | float]]:
    func = target_function(act_func)
    x_values = np.arange(x_range[0], x_range[1], eval_step, dtype=np.float64)
    y_true = func(x_values)

    results: dict[str, dict[str, dict[str, float] | float]] = {}
    for bit, params in tables.items():
        breakpoints = np.asarray(params["breakpoints"], dtype=np.float64)
        slopes = np.asarray(params["slopes"], dtype=np.float64)
        intercepts = np.asarray(params["intercepts"], dtype=np.float64)
        split_points = np.concatenate(([neg_inf], breakpoints, [pos_inf]))
        y_pred = piecewise_linear_approximation(x_values, split_points, slopes, intercepts)
        direct_metrics = compute_metrics(y_true, y_pred)

        scale = 2.0 ** (-int(bit))
        q_min = int(np.ceil(x_range[0] / scale))
        q_max = int(np.floor((x_range[1] - 1e-12) / scale))
        q_values = np.arange(q_min, q_max + 1, dtype=np.float64)
        x_quantized = q_values * scale
        scaled_breakpoints = breakpoints / scale
        scaled_intercepts = intercepts / scale
        indices = np.digitize(q_values, scaled_breakpoints)
        indices = np.clip(indices, 0, len(slopes) - 1)
        y_quantized = (slopes[indices] * q_values + scaled_intercepts[indices]) * scale
        quantized_metrics = compute_metrics(func(x_quantized), y_quantized)

        results[bit] = {
            "scale": scale,
            "direct_metrics": direct_metrics,
            "quantized_metrics": quantized_metrics,
        }
    return results


def build_output_dir(args: argparse.Namespace, act_func: str, entries: int) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        parts = [timestamp, "gqalut", act_func, f"e{entries}", args.mutation_mode]
        if args.run_name:
            parts.append(args.run_name)
        output_dir = Path(__file__).resolve().parents[1] / "runs" / "-".join(parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_search_history(path: Path, history: list[dict[str, float]]) -> None:
    generations = np.array([int(row["generation"]) for row in history], dtype=np.int64)
    best = np.array([row["best_mse"] for row in history], dtype=np.float64)
    mean = np.array([row["mean_mse"] for row in history], dtype=np.float64)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.plot(generations, best, label="best MSE", color="#0f766e", linewidth=2.0)
    axis.plot(generations, mean, label="mean MSE", color="#1d4ed8", linewidth=1.7, alpha=0.85)
    axis.set_yscale("log")
    axis.set_xlabel("Generation")
    axis.set_ylabel("MSE")
    axis.set_title("GQA-LUT Search History")
    axis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    axis.legend(loc="best")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_best_fit(
    path: Path,
    *,
    act_func: str,
    x_range: tuple[float, float],
    eval_step: float,
    split_points: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
) -> None:
    func = target_function(act_func)
    x_values = np.arange(x_range[0], x_range[1], eval_step, dtype=np.float64)
    y_true = func(x_values)
    y_pred = piecewise_linear_approximation(x_values, split_points, slopes, intercepts)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.plot(x_values, y_true, label=f"{act_func} reference", color="#111827", linewidth=2.2)
    axis.plot(x_values, y_pred, label="best PWL", color="#dc2626", linewidth=2.0, alpha=0.9)
    for breakpoint in split_points[1:-1]:
        axis.axvline(float(breakpoint), color="#94a3b8", linestyle=":", linewidth=0.8, alpha=0.6)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(f"GQA-LUT best fit for {act_func}")
    axis.legend(loc="best")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def compare_with_reference(
    reference_path: Path,
    generated_tables: dict[str, dict[str, list[float]]],
    *,
    act_func: str,
) -> dict[str, Any]:
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_tables = reference_payload[act_func]
    comparison: dict[str, Any] = {}
    for bit, generated in generated_tables.items():
        reference = reference_tables[bit]
        comparison[bit] = {
            "breakpoint_max_abs_diff": float(
                np.max(
                    np.abs(
                        np.asarray(generated["breakpoints"], dtype=np.float64)
                        - np.asarray(reference["breakpoints"], dtype=np.float64)
                    )
                )
            ),
            "slope_max_abs_diff": float(
                np.max(
                    np.abs(
                        np.asarray(generated["slopes"], dtype=np.float64)
                        - np.asarray(reference["slopes"], dtype=np.float64)
                    )
                )
            ),
            "intercept_max_abs_diff": float(
                np.max(
                    np.abs(
                        np.asarray(generated["intercepts"], dtype=np.float64)
                        - np.asarray(reference["intercepts"], dtype=np.float64)
                    )
                )
            ),
        }
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the published GQA-LUT scalar PWL search without requiring DEAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--act-func",
        default="gelu",
        help="Target function. Supports paper names such as gelu, hswish, exp, div, rsqrt.",
    )
    parser.add_argument("--entries", type=int, choices=[8, 16], default=8, help="LUT entry count.")
    parser.add_argument(
        "--paper-defaults",
        action="store_true",
        help="Use the public GQA-LUT repo presets for the selected function and --entries.",
    )
    parser.add_argument("--x-range", nargs=2, type=float, default=None, metavar=("L", "R"))
    parser.add_argument("--sp-range", nargs=2, type=float, default=None, metavar=("L", "R"))
    parser.add_argument(
        "--num-splits",
        type=int,
        default=None,
        help="Number of interior split points. Defaults to --entries - 1.",
    )
    parser.add_argument("--neg-inf", type=float, default=None, help="Left outer segment boundary.")
    parser.add_argument("--pos-inf", type=float, default=None, help="Right outer segment boundary.")
    parser.add_argument(
        "--decimal-bit-range",
        nargs=2,
        type=int,
        default=(0, 6),
        metavar=("MIN", "MAX"),
        help="Breakpoint decimal-bit range saved in the output tables.",
    )
    parser.add_argument(
        "--coeff-bit-width",
        type=int,
        default=8,
        help="Coefficient/intercept storage width. The released code quantizes these with width-2 fractional bits.",
    )
    parser.add_argument("--total-iters", type=int, default=500, help="Number of GA generations.")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size.")
    parser.add_argument("--crossover-prob", type=float, default=0.7, help="Two-point crossover probability.")
    parser.add_argument("--mutation-prob", type=float, default=0.2, help="Mutation probability per individual.")
    parser.add_argument("--mutation-sigma", type=float, default=0.2, help="Gaussian mutation stddev.")
    parser.add_argument(
        "--mutation-mode",
        choices=["rm", "gaussian"],
        default="rm",
        help="rm reproduces the released rounding-mutation logic; gaussian disables it.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Starting-bit offset used by rounding mutation. Paper presets fill this automatically.",
    )
    parser.add_argument("--eval-step", type=float, default=0.01, help="Search/evaluation grid step.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--run-name", default=None, help="Optional suffix for the run directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit output directory.")
    parser.add_argument(
        "--compare-reference",
        type=Path,
        default=None,
        help="Optional reference JSON in the original paper format for numeric comparison.",
    )
    args = parser.parse_args()
    return args


def resolve_config(args: argparse.Namespace) -> tuple[SearchConfig, int]:
    act_func = canonical_act_func(args.act_func)
    entries = int(args.entries)
    num_splits = entries - 1 if args.num_splits is None else int(args.num_splits)

    if num_splits <= 0:
        raise ValueError("--num-splits must be positive.")
    if args.eval_step <= 0:
        raise ValueError("--eval-step must be positive.")
    if args.pop_size <= 0:
        raise ValueError("--pop-size must be positive.")
    if args.total_iters < 0:
        raise ValueError("--total-iters must be non-negative.")
    if not 0.0 <= args.crossover_prob <= 1.0:
        raise ValueError("--crossover-prob must be in [0, 1].")
    if not 0.0 <= args.mutation_prob <= 1.0:
        raise ValueError("--mutation-prob must be in [0, 1].")

    preset: PaperPreset | None = None
    if args.paper_defaults:
        try:
            preset = PAPER_PRESETS[(act_func, entries)]
        except KeyError as exc:
            raise ValueError(
                f"No paper preset for act_func={act_func!r} and entries={entries}."
            ) from exc

    if preset is not None:
        x_range = tuple(args.x_range) if args.x_range is not None else preset.x_range
        sp_range = tuple(args.sp_range) if args.sp_range is not None else preset.sp_range
        neg_inf = float(args.neg_inf) if args.neg_inf is not None else preset.neg_inf
        pos_inf = float(args.pos_inf) if args.pos_inf is not None else preset.pos_inf
        offset = int(args.offset) if args.offset is not None else preset.offset
        coeff_bit_width = int(args.coeff_bit_width) if args.coeff_bit_width is not None else preset.coeff_bit_width
        total_iters = int(args.total_iters) if args.total_iters is not None else preset.total_iters
        pop_size = int(args.pop_size) if args.pop_size is not None else preset.pop_size
        crossover_prob = float(args.crossover_prob) if args.crossover_prob is not None else preset.crossover_prob
        mutation_prob = float(args.mutation_prob) if args.mutation_prob is not None else preset.mutation_prob
        decimal_bit_range = tuple(int(v) for v in args.decimal_bit_range)
    else:
        if args.x_range is None or args.sp_range is None:
            raise ValueError("--x-range and --sp-range are required unless --paper-defaults is used.")
        x_range = tuple(args.x_range)
        sp_range = tuple(args.sp_range)
        neg_inf = float(args.neg_inf) if args.neg_inf is not None else x_range[0]
        pos_inf = float(args.pos_inf) if args.pos_inf is not None else x_range[1]
        offset = int(args.offset) if args.offset is not None else 0
        coeff_bit_width = int(args.coeff_bit_width)
        total_iters = int(args.total_iters)
        pop_size = int(args.pop_size)
        crossover_prob = float(args.crossover_prob)
        mutation_prob = float(args.mutation_prob)
        decimal_bit_range = tuple(int(v) for v in args.decimal_bit_range)

    if x_range[0] >= x_range[1]:
        raise ValueError("--x-range must have L < R.")
    if sp_range[0] > sp_range[1]:
        raise ValueError("--sp-range must have L <= R.")
    if decimal_bit_range[0] < 0 or decimal_bit_range[0] > decimal_bit_range[1]:
        raise ValueError("--decimal-bit-range must satisfy 0 <= min <= max.")

    config = SearchConfig(
        act_func=act_func,
        x_range=(float(x_range[0]), float(x_range[1])),
        sp_range=(float(sp_range[0]), float(sp_range[1])),
        num_splits=num_splits,
        total_iters=total_iters,
        decimal_bit_range=decimal_bit_range,
        coeff_bit_width=coeff_bit_width,
        pop_size=pop_size,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        mutation_sigma=float(args.mutation_sigma),
        mutation_mode=args.mutation_mode,
        offset=offset,
        neg_inf=neg_inf,
        pos_inf=pos_inf,
        eval_step=float(args.eval_step),
        seed=int(args.seed),
        output_dir=str(args.output_dir) if args.output_dir is not None else None,
        run_name=args.run_name,
        compare_reference=str(args.compare_reference) if args.compare_reference is not None else None,
    )
    return config, entries


def main() -> None:
    args = parse_args()
    config, entries = resolve_config(args)
    output_dir = build_output_dir(args, config.act_func, entries)

    result = run_search(config)
    tables_payload = build_decimal_bit_tables(
        result,
        act_func=config.act_func,
        decimal_bit_range=config.decimal_bit_range,
    )
    act_tables = tables_payload[config.act_func]
    metrics = evaluate_decimal_bit_tables(
        act_tables,
        act_func=config.act_func,
        x_range=config.x_range,
        eval_step=config.eval_step,
        neg_inf=config.neg_inf,
        pos_inf=config.pos_inf,
    )

    table_filename = f"{config.act_func}_pwl_{config.num_splits}.json"
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / table_filename, tables_payload)
    save_json(
        output_dir / "best_fp32.json",
        {
            "act_func": config.act_func,
            "best_individual": result.best_individual.astype(float).tolist(),
            "best_split_points": result.best_split_points.astype(float).tolist(),
            "slopes": result.slopes.astype(float).tolist(),
            "intercepts": result.intercepts.astype(float).tolist(),
            "best_mse": result.best_mse,
        },
    )
    save_history_csv(output_dir / "history.csv", result.history)
    plot_search_history(output_dir / "history.png", result.history)
    plot_best_fit(
        output_dir / "approximation.png",
        act_func=config.act_func,
        x_range=config.x_range,
        eval_step=config.eval_step,
        split_points=result.best_split_points,
        slopes=result.slopes,
        intercepts=result.intercepts,
    )

    summary: dict[str, Any] = {
        "act_func": config.act_func,
        "entries": entries,
        "num_splits": config.num_splits,
        "best_mse": result.best_mse,
        "best_individual": result.best_individual.astype(float).tolist(),
        "best_split_points": result.best_split_points.astype(float).tolist(),
        "slopes": result.slopes.astype(float).tolist(),
        "intercepts": result.intercepts.astype(float).tolist(),
        "decimal_bit_metrics": metrics,
        "output_dir": str(output_dir),
        "table_file": table_filename,
    }

    if args.compare_reference is not None:
        summary["reference_comparison"] = compare_with_reference(
            args.compare_reference,
            act_tables,
            act_func=config.act_func,
        )

    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
