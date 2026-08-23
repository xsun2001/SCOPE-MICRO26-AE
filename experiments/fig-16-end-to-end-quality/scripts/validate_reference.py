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
        description="Audit the compact Figure 16 reference table and its derived means."
    )
    parser.add_argument("--expected", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    with args.expected.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    variants = [] if not rows else [
        field for field in rows[0] if field not in {"model", "metric"}
    ]
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["model"], row["metric"])
        if key in indexed:
            failures.append(f"duplicate row: {key}")
        indexed[key] = row
        for variant in variants:
            value = float(row[variant])
            if not math.isfinite(value):
                failures.append(f"non-finite value: {(*key, variant)}")

    models = sorted({model for model, _ in indexed})
    expected_metrics = {"ppl", "group1_mean", *TASK_METRICS}
    for model in models:
        missing = sorted(expected_metrics - {metric for candidate, metric in indexed if candidate == model})
        if missing:
            failures.append(f"{model}: missing metrics {missing}")
            continue
        for variant in variants:
            stored = float(indexed[(model, "group1_mean")][variant])
            recomputed = sum(
                float(indexed[(model, task)][variant]) for task in TASK_METRICS
            ) / len(TASK_METRICS)
            if not math.isclose(stored, recomputed, rel_tol=0.0, abs_tol=1e-12):
                failures.append(
                    f"inconsistent group1_mean for {(model, variant)}: "
                    f"stored={stored} recomputed={recomputed}"
                )

    source_records = len(rows) * len(variants)
    comparisons = sum(
        len(variants) for _, metric in indexed if metric in PUBLISHED_METRICS
    )
    if len(models) != 5 or len(rows) != 30 or len(variants) != 8:
        failures.append(
            f"expected 5 models, 30 rows, and 8 variants; got "
            f"{len(models)}, {len(rows)}, and {len(variants)}"
        )
    if comparisons != 80:
        failures.append(f"expected 80 published values, found {comparisons}")

    payload = {
        "status": "fail" if failures else "pass",
        "comparisons": comparisons,
        "passed": comparisons if not failures else 0,
        "source_records": source_records,
        "precision_note": "legacy fp16_* columns execute torch.bfloat16 and are reported as BF16",
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
