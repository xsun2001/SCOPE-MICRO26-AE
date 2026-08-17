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
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics_path = args.generated / "scna_metrics.csv"
    raw_path = args.generated / "raw_predictions.csv"
    generated = {row["function"]: row for row in csv.DictReader(metrics_path.open())}
    expected = {row["function"]: row for row in csv.DictReader(args.expected.open())}
    raw: dict[str, list[tuple[float, float]]] = {}
    for row in csv.DictReader(raw_path.open()):
        raw.setdefault(row["function"], []).append((float(row["target"]), float(row["prediction"])))
    specification = json.loads(args.checkpoints.read_text())
    failures: list[str] = []
    comparisons: list[dict[str, object]] = []

    if set(generated) != set(expected):
        failures.append(f"function set mismatch: generated={sorted(generated)}, expected={sorted(expected)}")
    configured = {row["display_name"]: row for row in specification["rows"].values()}
    for name in sorted(set(generated) & set(expected)):
        config = configured.get(name)
        if config is None:
            failures.append(f"{name}: no checkpoint configuration")
            continue
        metadata = generated[name]
        if int(metadata["eval_points"]) != int(specification["eval_points"]):
            failures.append(f"{name}: evaluation-point count mismatch")
        if int(metadata["num_units"]) != int(specification["num_units"]):
            failures.append(f"{name}: SCNA unit count mismatch")
        if int(metadata["seed"]) != int(specification["seed"]):
            failures.append(f"{name}: seed mismatch")
        if float(metadata["l_range"]) != float(config["l_range"]) or float(metadata["r_range"]) != float(config["r_range"]):
            failures.append(f"{name}: evaluation range mismatch")
        points = raw.get(name, [])
        if len(points) != int(generated[name]["eval_points"]):
            failures.append(f"{name}: raw point count mismatch")
            continue
        errors = [prediction - reference for reference, prediction in points]
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
    for baseline in ("nnlut", "tlut"):
        valid: list[float] = []
        for name, row in expected.items():
            # The paper's 431x NN-LUT geomean covers the nine general
            # nonlinearities; Exp/Exp2 are omitted from that comparison.
            if baseline == "nnlut" and name in {"Exp", "Exp2"}:
                continue
            base = finite(row[f"mse_{baseline}"])
            if base is not None and name in generated:
                valid.append(base / float(generated[name]["mse"]))
        ratios[f"geomean_mse_improvement_vs_{baseline}"] = math.exp(
            sum(math.log(value) for value in valid) / len(valid)
        )
        ratios[f"geomean_mse_comparisons_vs_{baseline}"] = len(valid)

    paper_geomeans = {"nnlut": 431.0, "tlut": 14.9}
    for baseline, reference in paper_geomeans.items():
        actual = float(ratios[f"geomean_mse_improvement_vs_{baseline}"])
        if abs(actual - reference) / reference > args.relative_tolerance:
            failures.append(f"{baseline} geomean: {actual:.6g} is not within the portability envelope of {reference}")

    payload = {
        "status": "fail" if failures else "pass",
        "published_rows": len(generated),
        "reproduced_method": "SCNA",
        "comparisons": len(comparisons),
        "relative_tolerance": args.relative_tolerance,
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
