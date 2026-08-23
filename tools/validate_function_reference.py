from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_number(value: str) -> float | None:
    text = value.strip()
    if not text or text.upper() == "N/A":
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def load_manifest(path: Path) -> list[tuple[str, int, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [(row["function"], int(row["width"]), row["reparam"]) for row in rows]


def load_expected(path: Path) -> tuple[dict[tuple[str, int, str], float], list[str]]:
    values: dict[tuple[str, int, str], float] = {}
    failures: list[str] = []
    with path.open() as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            func = row["func"]
            width = int(row["num_units"])
            exp = parse_number(row["exp_best_mse"])
            none = parse_number(row["none_best_mse"])
            if exp is None or exp <= 0.0:
                failures.append(f"line {line_number}: invalid exp_best_mse")
            else:
                values[(func, width, "exp")] = exp
            if none is None or none <= 0.0:
                failures.append(f"line {line_number}: invalid none_best_mse")
            else:
                values[(func, width, "none")] = none
            ratio = parse_number(row["ratio_exp_to_none"])
            if exp is not None and none is not None:
                calculated = exp / none
                if ratio is None or not math.isclose(ratio, calculated, rel_tol=1e-9):
                    failures.append(f"line {line_number}: inconsistent exp/none ratio")
                expected_better = "exp" if exp < none else "none"
                if row["better"] != expected_better:
                    failures.append(f"line {line_number}: inconsistent better field")
    return values, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a function-approximation reference matrix without runtime traces."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("width", "shape"), required=True)
    parser.add_argument("--figure", type=Path, action="append", default=[])
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    expected, failures = load_expected(args.expected)
    if len(set(manifest)) != len(manifest):
        failures.append("manifest contains duplicate configurations")

    rows: list[dict[str, object]] = []
    for func, width, reparam in manifest:
        value = expected.get((func, width, reparam))
        passed = value is not None and value > 0.0
        if not passed:
            failures.append(f"missing expected value: {func}/{width}/{reparam}")
        rows.append(
            {
                "function": func,
                "width": width,
                "reparam": reparam,
                "expected_best_mse": value,
                "pass": passed,
            }
        )

    functions = sorted({func for func, _, _ in manifest})
    gains: list[dict[str, object]] = []
    if args.mode == "width":
        for func in functions:
            narrow = expected.get((func, 4, "exp"))
            wide = expected.get((func, 32, "exp"))
            if narrow is None or wide is None:
                failures.append(f"missing width endpoints for {func}")
                continue
            gain = narrow / wide
            gains.append({"function": func, "gain_32_over_4": gain})
            if gain <= 1.0:
                failures.append(f"width-32 does not improve over width-4 for {func}")
    else:
        for func in functions:
            constrained = expected.get((func, 16, "exp"))
            unconstrained = expected.get((func, 16, "none"))
            if constrained is None or unconstrained is None:
                failures.append(f"missing shape-constraint pair for {func}")
                continue
            gain = unconstrained / constrained
            gains.append({"function": func, "gain_none_over_scna": gain})
            if gain <= 1.0:
                failures.append(f"shape constraints do not improve {func}")

    for figure in args.figure:
        if not figure.is_file() or figure.stat().st_size == 0:
            failures.append(f"missing canonical figure: {figure}")

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    with (args.analysis_dir / "reference_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gain_name = "gain_32_over_4" if args.mode == "width" else "gain_none_over_scna"
    with (args.analysis_dir / "gains.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["function", gain_name])
        writer.writeheader()
        writer.writerows(gains)

    payload = {
        "status": "fail" if failures else "pass",
        "comparisons": len(manifest),
        "passed": sum(bool(row["pass"]) for row in rows),
        "validation_rule": "manifest/reference integrity and reported trend",
        "runtime_traces_required": False,
        "failures": failures,
    }
    (args.analysis_dir / "validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
