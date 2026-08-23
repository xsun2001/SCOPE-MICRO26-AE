from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from table5_metrics import MODEL_PREFIXES, PROTOCOL_DESCRIPTION, load_bf16_baselines


MODE_MAP = {
    "OSTQuant": "exact_eager_maskfix_acc",
    "SCNA-8": "scna_d8_maskfix_acc",
    "SCNA-16": "scna_d16_maskfix_acc",
    "SCNA-32": "scna_d32_maskfix_acc",
}
QUANT_MAP = {"W6A6": "w6a6kv6", "W4A4": "w4a4kv4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the unified four-task Table 5.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result-dir", type=Path)
    source.add_argument("--summary", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Analysis output directory; defaults to RESULT_DIR/analysis",
    )
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument(
        "--bf16-source",
        type=Path,
        required=True,
        help="Figure 16 BF16 baseline CSV (legacy source columns are named fp16_exact)",
    )
    parser.add_argument("--ppl-tolerance", type=float, default=0.04)
    parser.add_argument("--accuracy-tolerance-percent", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = (
        args.summary
        if args.summary is not None
        else args.result_dir / "analysis" / "maskfix_summary.csv"
    )
    analysis_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.result_dir / "analysis"
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open() as handle:
        actual = {
            (row["model_key"], row["quant_key"], row["mode"]): row
            for row in csv.DictReader(handle)
        }
    bf16 = load_bf16_baselines(args.bf16_source)

    comparisons: list[dict[str, object]] = []
    with args.expected.open() as handle:
        for expected in csv.DictReader(handle):
            quant = QUANT_MAP[expected["quantization"]]
            for prefix, model in MODEL_PREFIXES.items():
                target_ppl = float(expected[f"{prefix}_ppl"])
                target_acc = float(expected[f"{prefix}_accuracy_percent"])
                if expected["method"] == "BF16 Baseline":
                    reproduced_ppl, reproduced_acc = bf16[model]
                else:
                    row = actual.get((model, quant, MODE_MAP[expected["method"]]))
                    reproduced_ppl = None if row is None else float(row["ppl"])
                    reproduced_acc = None if row is None else 100.0 * float(row["acc_avg"])
                passed = (
                    reproduced_ppl is not None
                    and reproduced_acc is not None
                    and abs(reproduced_ppl - target_ppl) <= args.ppl_tolerance
                    and abs(reproduced_acc - target_acc) <= args.accuracy_tolerance_percent
                )
                comparisons.append(
                    {
                        "model": model,
                        "quantization": expected["quantization"],
                        "method": expected["method"],
                        "target_ppl": target_ppl,
                        "reproduced_ppl": "" if reproduced_ppl is None else reproduced_ppl,
                        "target_accuracy_percent": target_acc,
                        "reproduced_accuracy_percent": "" if reproduced_acc is None else reproduced_acc,
                        "pass": passed,
                    }
                )

    comparison_path = analysis_dir / "table5_comparison.csv"
    with comparison_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(comparisons[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(comparisons)
    passed = sum(bool(row["pass"]) for row in comparisons)
    summary = {
        "status": "pass" if passed == len(comparisons) and len(comparisons) == 20 else "fail",
        "comparisons": len(comparisons),
        "passed": passed,
        "accuracy_tasks": ["arc_easy", "hellaswag", "piqa", "winogrande"],
        "aggregation": PROTOCOL_DESCRIPTION,
        "ppl_tolerance": args.ppl_tolerance,
        "accuracy_tolerance_percent": args.accuracy_tolerance_percent,
    }
    (analysis_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (analysis_dir / "validation.md").write_text(
        "# Table 5 four-task reproduction\n\n"
        f"Status: **{summary['status']}**\n\n"
        f"Passed {passed}/{len(comparisons)} values using ARC-Easy, HellaSwag, PIQA, and WinoGrande.\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
