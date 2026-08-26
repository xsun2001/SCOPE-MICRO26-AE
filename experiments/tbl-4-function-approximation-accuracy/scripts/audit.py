from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def finite(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated Table 4 evidence against a separate paper reference.")
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--variant", choices=("scna16", "scna32"), default="scna16")
    parser.add_argument("--expected-nnlut-geomean", type=float, required=True)
    parser.add_argument("--expected-tlut-geomean", type=float, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics_path = args.generated / "scna_metrics.csv"
    raw_path = args.generated / "raw_predictions.csv"
    generated = {row["function"]: row for row in csv.DictReader(metrics_path.open())}
    expected = {row["function"]: row for row in csv.DictReader(args.expected.open())}
    raw: dict[str, list[tuple[float, float, float]]] = {}
    for row in csv.DictReader(raw_path.open()):
        raw.setdefault(row["function"], []).append(
            (float(row["x"]), float(row["target"]), float(row["prediction"]))
        )
    specification = json.loads(args.parameters.read_text())
    variant = specification["variants"][args.variant]
    failures: list[str] = []
    comparisons: list[dict[str, object]] = []
    parameter_counts: dict[str, int] = {}

    if set(generated) != set(expected):
        failures.append(f"function set mismatch: generated={sorted(generated)}, expected={sorted(expected)}")
    configured = {row["display_name"]: row for row in variant["rows"].values()}
    rsqrt_config = configured.get("Rsqrt")
    if rsqrt_config is None:
        failures.append("Rsqrt: no checkpoint configuration")
    elif (
        rsqrt_config.get("target") != "reflected_rsqrt"
        or float(rsqrt_config["l_range"]) != -1024.0
        or float(rsqrt_config["r_range"]) != -0.1
        or float(rsqrt_config.get("lambda_bound", -1.0)) != 0.10
    ):
        failures.append(
            "Rsqrt: expected reflected_rsqrt on [-1024, -0.1] with lambda_bound=0.10"
        )
    for name in sorted(set(generated) & set(expected)):
        config = configured.get(name)
        if config is None:
            failures.append(f"{name}: no checkpoint configuration")
            continue
        metadata = generated[name]
        if int(metadata["eval_points"]) != int(specification["eval_points"]):
            failures.append(f"{name}: evaluation-point count mismatch")
        if int(metadata["num_units"]) != int(variant["num_units"]):
            failures.append(f"{name}: SCNA unit count mismatch")
        if metadata["variant"] != args.variant:
            failures.append(f"{name}: SCNA variant mismatch")
        parameter_kind = str(metadata["parameter_kind"])
        parameter_counts[parameter_kind] = parameter_counts.get(parameter_kind, 0) + 1
        if parameter_kind != config["kind"]:
            failures.append(f"{name}: parameter provenance kind mismatch")
        if parameter_kind == "embedded_trained_parameters":
            if int(metadata["training_seed"]) != int(config["training_seed"]):
                failures.append(f"{name}: training seed mismatch")
            if metadata["parameter_source"] != config["source_checkpoint"]:
                failures.append(f"{name}: parameter source mismatch")
        if float(metadata["l_range"]) != float(config["l_range"]) or float(metadata["r_range"]) != float(config["r_range"]):
            failures.append(f"{name}: evaluation range mismatch")
        points = raw.get(name, [])
        if len(points) != int(generated[name]["eval_points"]):
            failures.append(f"{name}: raw point count mismatch")
            continue
        if name == "Rsqrt" and any(
            x >= 0.0
            or not math.isclose(reference, 1.0 / math.sqrt(-x), rel_tol=1e-12, abs_tol=1e-15)
            for x, reference, _ in points
        ):
            failures.append("Rsqrt: raw targets do not equal 1 / sqrt(-x)")
        errors = [prediction - reference for _, reference, prediction in points]
        recomputed_mse = sum(error * error for error in errors) / len(errors)
        recomputed_mae = sum(abs(error) for error in errors) / len(errors)
        reported_mse = float(generated[name]["mse"])
        reported_mae = float(generated[name]["mae"])
        if not math.isclose(recomputed_mse, reported_mse, rel_tol=1e-12, abs_tol=1e-18):
            failures.append(f"{name}: MSE cannot be reconstructed from raw points")
        if not math.isclose(recomputed_mae, reported_mae, rel_tol=1e-12, abs_tol=1e-18):
            failures.append(f"{name}: MAE cannot be reconstructed from raw points")
        for metric, actual in (("mse", reported_mse), ("mae", reported_mae)):
            reference = float(expected[name][f"{metric}_ours"])
            relative_error = abs(actual - reference) / reference
            passed = relative_error <= args.relative_tolerance
            comparisons.append(
                {
                    "function": name,
                    "metric": metric,
                    "actual": actual,
                    "reference": reference,
                    "relative_error": relative_error,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(f"{name} {metric}: relative error {relative_error:.6g} > {args.relative_tolerance}")

    ratios: dict[str, float | int] = {}
    baselines = ("taylor", "frac_t", "interp", "frac_i", "linearlut", "nnlut", "tlut")
    for metric in ("mse", "mae"):
        for baseline in baselines:
            valid: list[float] = []
            for name, row in expected.items():
                base = finite(row[f"{metric}_{baseline}"])
                if base is not None and name in generated:
                    valid.append(base / float(generated[name][metric]))
            ratios[f"geomean_{metric}_improvement_vs_{baseline}"] = math.exp(
                sum(math.log(value) for value in valid) / len(valid)
            )
            ratios[f"geomean_{metric}_comparisons_vs_{baseline}"] = len(valid)

    # The revised paper reports NN-LUT over all 11 functions and T-LUT over
    # the nine rows for which the baseline publishes a numeric result.
    paper_geomeans = {
        "nnlut": args.expected_nnlut_geomean,
        "tlut": args.expected_tlut_geomean,
    }
    for baseline, reference in paper_geomeans.items():
        actual = float(ratios[f"geomean_mse_improvement_vs_{baseline}"])
        if abs(actual - reference) / reference > args.relative_tolerance:
            failures.append(f"{baseline} geomean: {actual:.6g} is not within the portability envelope of {reference}")

    payload = {
        "status": "fail" if failures else "pass",
        "variant": args.variant,
        "num_units": int(variant["num_units"]),
        "published_rows": len(generated),
        "reproduced_method": "SCNA",
        "comparisons": len(comparisons),
        "relative_tolerance": args.relative_tolerance,
        "parameter_counts": parameter_counts,
        "parameter_provenance": "all rows use fused trained weights and biases embedded in the data manifest",
        "reference_role": "validation only; the reproduction command does not read this file",
        **ratios,
        "details": comparisons,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
