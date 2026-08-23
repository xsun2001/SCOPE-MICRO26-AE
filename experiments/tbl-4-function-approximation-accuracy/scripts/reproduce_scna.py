from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import erf


DEFAULT_PARAMETERS = Path(__file__).resolve().parents[1] / "data" / "scna_parameters.json"


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
    if name == "reflected_rsqrt":
        return 1.0 / np.sqrt(-x)
    if name == "gelu_erf_branch":
        # SCNA approximates the nonlinear erf branch; x/2 is an exact outer
        # multiply in the canonical GeLU decomposition.
        return erf(x / np.sqrt(2.0)) + 1.0
    raise ValueError(f"unknown target {name!r}")


def embedded_parameters(
    row: dict[str, object], units: int
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Read fused SCNA weights and biases directly from the data manifest."""
    wk = np.asarray(row["weights"], dtype=np.float64)
    bk = np.asarray(row["biases"], dtype=np.float64)
    if wk.shape != (units,) or bk.shape != (units,):
        raise ValueError(
            f"{row['display_name']}: expected {units} weights and biases, "
            f"got {wk.shape} and {bk.shape}"
        )
    if not np.all(np.isfinite(wk)) or not np.all(np.isfinite(bk)):
        raise ValueError(f"{row['display_name']}: parameters must be finite")
    return wk, bk, int(row["training_seed"]), str(row["source_checkpoint"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate embedded trained SCNA parameters independently of paper values."
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=DEFAULT_PARAMETERS,
        help="combined SCNA parameter manifest (defaults to data/scna_parameters.json)",
    )
    parser.add_argument(
        "--variant",
        choices=("scna16", "scna32"),
        help="parameter variant; defaults to the manifest's SCNA-16 variant",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    specification = json.loads(args.parameters.read_text())
    variant_name = args.variant or str(specification["default_variant"])
    variant = specification["variants"][variant_name]
    units = int(variant["num_units"])
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
        for row in variant["rows"].values():
            lo, hi = float(row["l_range"]), float(row["r_range"])
            if row["kind"] != "embedded_trained_parameters":
                raise ValueError(f"unknown parameter kind {row['kind']!r}")
            wk, bk, training_seed, parameter_source = embedded_parameters(row, units)
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
                    "variant": variant_name,
                    "parameter_kind": row["kind"],
                    "parameter_source": parameter_source,
                    "training_seed": training_seed,
                }
            )
    with metrics_path.open("w", newline="") as metric_file:
        writer = csv.DictWriter(metric_file, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    print(
        json.dumps(
            {
                "status": "complete",
                "variant": variant_name,
                "num_units": units,
                "rows": len(metric_rows),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
