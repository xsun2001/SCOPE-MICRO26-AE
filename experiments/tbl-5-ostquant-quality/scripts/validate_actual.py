from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from table5_metrics import (
    FOUR_TASKS,
    MODEL_PREFIXES,
    PROTOCOL_DESCRIPTION,
    four_task_average,
    load_fp16_baselines,
)


MODE_MAP = {
    "OSTQuant": "exact_eager_maskfix_acc",
    "SCNA-8": "scna_d8_maskfix_acc",
    "SCNA-16": "scna_d16_maskfix_acc",
    "SCNA-32": "scna_d32_maskfix_acc",
}
QUANT_MAP = {"W6A6": "w6a6kv6", "W4A4": "w4a4kv4"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the unified four-task Table 5 from packaged metric JSON files."
    )
    parser.add_argument("--actual", type=Path, required=True, help="Packaged analysis directory")
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--fp16-source", type=Path, required=True)
    parser.add_argument("--ppl-tolerance", type=float, default=0.04)
    parser.add_argument("--accuracy-tolerance-percent", type=float, default=1.0)
    args = parser.parse_args()

    run_root = args.actual.parent
    failures: list[str] = []
    fp16 = load_fp16_baselines(args.fp16_source)

    with args.expected.open() as handle:
        expected_rows = list(csv.DictReader(handle))
    expected_dirs: set[str] = set()
    comparisons = 0
    passed = 0
    metric_files = 0

    for expected in expected_rows:
        method = expected["method"]
        quant_label = expected["quantization"]
        if quant_label not in QUANT_MAP or (method != "BF16 Baseline" and method not in MODE_MAP):
            failures.append(f"unknown expected method/quantization: {expected}")
            continue
        quant = QUANT_MAP[quant_label]
        for prefix, model in MODEL_PREFIXES.items():
            target_ppl = float(expected[f"{prefix}_ppl"])
            target_acc = float(expected[f"{prefix}_accuracy_percent"])
            if method == "BF16 Baseline":
                ppl, acc_percent = fp16[model]
            else:
                mode = MODE_MAP[method]
                result_dir = run_root / f"eval_{model}_{quant}_{mode}"
                expected_dirs.add(result_dir.name)
                metrics_path = result_dir / "metrics.json"
                status_path = result_dir / "status.json"
                if not metrics_path.is_file() or not status_path.is_file():
                    failures.append(f"missing metrics/status for {result_dir.name}")
                    continue
                status = json.loads(status_path.read_text())
                if status.get("returncode") != 0:
                    failures.append(f"nonzero return code for {result_dir.name}: {status.get('returncode')}")
                metrics = json.loads(metrics_path.read_text())
                final = metrics.get("lm_eval", {}).get("final", {})
                try:
                    ppl = float(metrics["ppl"])
                    acc_percent = 100.0 * four_task_average(final)
                except (KeyError, TypeError, ValueError) as error:
                    failures.append(f"invalid metrics for {result_dir.name}: {error}")
                    continue
                if tuple(final.get("tasks", ())) != FOUR_TASKS:
                    failures.append(f"official task list mismatch for {result_dir.name}: {final.get('tasks')}")
                if not math.isclose(float(final.get("acc_avg", math.nan)), acc_percent / 100.0, rel_tol=0.0, abs_tol=1e-12):
                    failures.append(f"stored four-task average mismatch for {result_dir.name}")
                protocol_tasks = final.get("accuracy_protocol", {}).get("tasks", [])
                if tuple(protocol_tasks) != FOUR_TASKS:
                    failures.append(f"accuracy protocol metadata mismatch for {result_dir.name}")

                bits = int(quant[1])
                quantization = metrics.get("quantization", {})
                if any(quantization.get(name) != bits for name in ("w_bits", "a_bits", "k_bits", "v_bits")):
                    failures.append(f"quantization metadata mismatch for {result_dir.name}")
                scna = metrics.get("scna", {})
                expected_dim = None if mode.startswith("exact_") else int(mode.split("_d", 1)[1].split("_", 1)[0])
                if bool(scna.get("enabled")) != (expected_dim is not None) or scna.get("dim") != expected_dim:
                    failures.append(f"SCNA metadata mismatch for {result_dir.name}")
                metric_files += 1

            comparisons += 1
            if not math.isfinite(ppl) or not math.isfinite(acc_percent):
                failures.append(f"non-finite values for {(model, quant_label, method)}")
            elif (
                abs(ppl - target_ppl) <= args.ppl_tolerance
                and abs(acc_percent - target_acc) <= args.accuracy_tolerance_percent
            ):
                passed += 1

    actual_dirs = {path.name for path in run_root.glob("eval_*_maskfix_acc") if path.is_dir()}
    if expected_dirs - actual_dirs:
        failures.append(f"missing evaluation directories: {sorted(expected_dirs - actual_dirs)}")
    if actual_dirs - expected_dirs:
        failures.append(f"unexpected evaluation directories: {sorted(actual_dirs - expected_dirs)}")
    if len(expected_rows) != 10:
        failures.append(f"expected 10 Table 5 rows, found {len(expected_rows)}")
    if comparisons != 20 or passed != comparisons:
        failures.append(f"only {passed}/{comparisons} four-task values pass; expected 20/20")
    if metric_files != 16:
        failures.append(f"validated {metric_files} quantized metric files; expected 16")

    payload = {
        "status": "fail" if failures else "pass",
        "comparisons": comparisons,
        "passed": passed,
        "metric_files": metric_files,
        "accuracy_tasks": list(FOUR_TASKS),
        "aggregation": PROTOCOL_DESCRIPTION,
        "ppl_tolerance": args.ppl_tolerance,
        "accuracy_tolerance_percent": args.accuracy_tolerance_percent,
        "precision": "BF16 baseline; fp16_exact is a retained legacy source-column identifier",
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
