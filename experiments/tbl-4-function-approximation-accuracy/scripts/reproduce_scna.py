from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import erf


def target(name: str, x: np.ndarray) -> np.ndarray:
    if name == "exp":
        return np.exp(x)
    if name == "exp2":
        return np.exp2(x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "softsign_shift":
        return x / (1.0 + np.abs(x)) + 1.0
    if name == "softplus":
        return np.logaddexp(0.0, x)
    if name == "tanh_shift":
        return np.tanh(x) + 1.0
    if name == "arctan_shift":
        return np.arctan(x) + np.pi / 2.0
    if name == "erf_shift":
        return erf(x) + 1.0
    if name == "sin_shift":
        return np.sin(x) + 1.0
    if name == "reflected_reciprocal":
        return 1.0 / (-x)
    if name == "gelu_erf_branch":
        # SCNA approximates the nonlinear erf branch; x/2 is an exact outer
        # multiply in the canonical GeLU decomposition.
        return erf(x / np.sqrt(2.0)) + 1.0
    raise ValueError(f"unknown target {name!r}")


def curvature_spline(row: dict[str, object], grid_points: int, units: int) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the archived curvature-density recipe to SCNA ReLU coefficients."""
    lo, hi = float(row["l_range"]), float(row["r_range"])
    grid = np.linspace(lo, hi, grid_points, dtype=np.float64)
    values = target(str(row["target"]), grid)
    dx = grid[1] - grid[0]
    second = np.gradient(np.gradient(values, dx), dx)
    density = np.maximum(np.abs(second), 1.0e-30) ** 0.4
    cumulative = np.empty_like(grid)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum((density[:-1] + density[1:]) * (0.5 * dx))
    knot_mass = np.linspace(0.0, cumulative[-1], units + 1, dtype=np.float64)
    knots = np.interp(knot_mass, cumulative, grid)
    knot_values = target(str(row["target"]), knots)
    slopes = np.diff(knot_values) / np.diff(knots)
    increments = np.empty(units, dtype=np.float64)
    increments[0] = slopes[0]
    increments[1:] = np.diff(slopes)
    thresholds = np.empty(units, dtype=np.float64)
    thresholds[0] = knots[0] - knot_values[0] / slopes[0]
    thresholds[1:] = knots[1:-1]
    wk = increments * float(row["scale"])
    bk = -wk * thresholds

    # The archived recipe includes a tiny affine calibration. The first unit is
    # active over the complete evaluation interval, so it carries this term.
    affine_slope = float(row["affine_slope"])
    affine_offset = float(row["affine_offset"])
    wk[0] += affine_slope
    bk[0] += affine_offset - affine_slope * lo
    return wk, bk


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate all 11 archived SCNA checkpoints independently of paper values.")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    specification = json.loads(args.checkpoints.read_text())
    np.random.seed(int(specification["seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "scna_metrics.csv"
    raw_path = args.output_dir / "raw_predictions.csv"
    metric_rows: list[dict[str, object]] = []

    with raw_path.open("w", newline="") as raw_file:
        raw_writer = csv.DictWriter(
            raw_file,
            fieldnames=("function", "point", "x", "target", "prediction", "error"),
        )
        raw_writer.writeheader()
        for row in specification["rows"].values():
            lo, hi = float(row["l_range"]), float(row["r_range"])
            if row["kind"] == "fused_checkpoint":
                wk = np.asarray(row["wk"], dtype=np.float64)
                bk = np.asarray(row["bk"], dtype=np.float64)
            else:
                wk, bk = curvature_spline(
                    row,
                    int(specification["curvature_grid_points"]),
                    int(specification["num_units"]),
                )
            x = np.linspace(lo, hi, int(specification["eval_points"]), dtype=np.float64)
            expected = target(str(row["target"]), x)
            prediction = np.maximum(x[:, None] * wk[None, :] + bk[None, :], 0.0).sum(axis=1)
            error = prediction - expected
            for index in range(x.size):
                raw_writer.writerow(
                    {
                        "function": row["display_name"],
                        "point": index,
                        "x": format(x[index], ".17g"),
                        "target": format(expected[index], ".17g"),
                        "prediction": format(prediction[index], ".17g"),
                        "error": format(error[index], ".17g"),
                    }
                )
            metric_rows.append(
                {
                    "function": row["display_name"],
                    "mse": format(float(np.mean(error**2)), ".17g"),
                    "mae": format(float(np.mean(np.abs(error))), ".17g"),
                    "l_range": format(lo, ".17g"),
                    "r_range": format(hi, ".17g"),
                    "eval_points": x.size,
                    "num_units": wk.size,
                    "seed": specification["seed"],
                }
            )
    with metrics_path.open("w", newline="") as metric_file:
        writer = csv.DictWriter(metric_file, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    print(json.dumps({"status": "complete", "rows": len(metric_rows), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
