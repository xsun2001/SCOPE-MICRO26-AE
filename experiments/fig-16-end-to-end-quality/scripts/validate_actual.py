from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TASK_METRICS = ("arc_easy", "hellaswag", "piqa", "winogrande")
PUBLISHED_METRICS = {"ppl", "group1_mean"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute Figure 16 validation from the bundled long-form metric records."
    )
    parser.add_argument("--actual", type=Path, required=True, help="Packaged analysis directory")
    parser.add_argument("--ppl-tolerance", type=float, default=0.05)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.005)
    args = parser.parse_args()

    experiment_root = Path(__file__).resolve().parents[1]
    expected_path = experiment_root / "expected-results" / "paper_figure16.csv"
    actual_path = args.actual / "results_long.csv"
    failures: list[str] = []

    with expected_path.open() as handle:
        expected_rows = list(csv.DictReader(handle))
    if not expected_rows:
        failures.append("paper_figure16.csv is empty")
        variants: list[str] = []
    else:
        variants = [field for field in expected_rows[0] if field not in {"model", "metric"}]

    expected: dict[tuple[str, str, str], float] = {}
    expected_row_keys: set[tuple[str, str]] = set()
    for row in expected_rows:
        row_key = (row["model"], row["metric"])
        if row_key in expected_row_keys:
            failures.append(f"duplicate expected row: {row_key}")
        expected_row_keys.add(row_key)
        for variant in variants:
            key = (*row_key, variant)
            value = float(row[variant])
            if not math.isfinite(value):
                failures.append(f"non-finite expected value: {key}")
            expected[key] = value

    with actual_path.open() as handle:
        actual_rows = list(csv.DictReader(handle))
    actual: dict[tuple[str, str, str], float] = {}
    for row in actual_rows:
        key = (row["model"], row["metric"], row["variant"])
        if key in actual:
            failures.append(f"duplicate actual record: {key}")
            continue
        value = float(row["value"])
        if not math.isfinite(value):
            failures.append(f"non-finite actual value: {key}")
        expected_kind = "wikitext" if row["metric"] == "ppl" else "lm_eval"
        if row["kind"] != expected_kind:
            failures.append(f"wrong source kind for {key}: {row['kind']} != {expected_kind}")
        actual[key] = value

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        failures.append(f"missing {len(missing)} records; first={missing[0]}")
    if extra:
        failures.append(f"unexpected {len(extra)} records; first={extra[0]}")

    for model, metric in expected_row_keys:
        if metric != "group1_mean":
            continue
        for variant in variants:
            mean_key = (model, metric, variant)
            task_keys = [(model, task, variant) for task in TASK_METRICS]
            if mean_key not in actual or any(key not in actual for key in task_keys):
                continue
            recomputed = sum(actual[key] for key in task_keys) / len(task_keys)
            if not math.isclose(actual[mean_key], recomputed, rel_tol=0.0, abs_tol=1e-12):
                failures.append(
                    f"inconsistent group1_mean for {(model, variant)}: "
                    f"stored={actual[mean_key]} recomputed={recomputed}"
                )

    comparisons = 0
    passed = 0
    for key, target in expected.items():
        if key[1] not in PUBLISHED_METRICS or key not in actual:
            continue
        comparisons += 1
        tolerance = args.ppl_tolerance if key[1] == "ppl" else args.accuracy_tolerance
        if abs(actual[key] - target) <= tolerance:
            passed += 1

    if comparisons != 80:
        failures.append(f"expected 80 published comparisons, recomputed {comparisons}")
    if passed != comparisons:
        failures.append(f"only {passed}/{comparisons} published values are within tolerance")

    payload = {
        "status": "fail" if failures else "pass",
        "comparisons": comparisons,
        "passed": passed,
        "source_records": len(actual_rows),
        "expected_records": len(expected),
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
