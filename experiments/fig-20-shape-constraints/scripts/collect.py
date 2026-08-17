from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def reference(path: Path) -> dict[tuple[str, str], float]:
    rows = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if int(row["num_units"]) == 16:
                rows[(row["func"], "exp")] = float(row["exp_best_mse"])
                rows[(row["func"], "none")] = float(row["none_best_mse"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.20)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0002)
    args = parser.parse_args()
    expected = reference(args.expected)
    rows = []
    missing = []
    actual = {}
    for (func, reparam), target in expected.items():
        path = args.runs_dir / f"{func}_16_{reparam}" / "summary.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        value = float(json.loads(path.read_text())["best_mse"])
        absolute_error = abs(value - target)
        error = absolute_error / target
        actual[(func, reparam)] = value
        rows.append({"function": func, "reparam": reparam, "expected_best_mse": target, "actual_best_mse": value, "absolute_error": absolute_error, "relative_error": error, "pass": error <= args.relative_tolerance or absolute_error <= args.absolute_tolerance})
    gains = [{"function": func, "gain_none_over_scna": actual[(func, "none")] / actual[(func, "exp")]} for func in sorted({key[0] for key in actual}) if (func, "exp") in actual and (func, "none") in actual]
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    with (args.analysis_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["function", "reparam", "expected_best_mse", "actual_best_mse", "relative_error", "pass"])
        writer.writeheader(); writer.writerows(rows)
    with (args.analysis_dir / "gains.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["function", "gain_none_over_scna"])
        writer.writeheader(); writer.writerows(gains)
    passed = sum(bool(row["pass"]) for row in rows)
    trend_passed = len(gains) == 9 and all(row["gain_none_over_scna"] > 1.0 for row in gains)
    payload = {"status": "pass" if not missing and passed == len(expected) and trend_passed else "fail", "comparisons": len(expected), "passed": passed, "missing": missing, "relative_tolerance": args.relative_tolerance, "absolute_tolerance": args.absolute_tolerance, "validation_rule": "relative envelope or absolute MSE floor, plus SCNA improvement for all nine functions", "trend_passed": trend_passed, "gain_min": min((row["gain_none_over_scna"] for row in gains), default=None), "gain_max": max((row["gain_none_over_scna"] for row in gains), default=None)}
    (args.analysis_dir / "validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
