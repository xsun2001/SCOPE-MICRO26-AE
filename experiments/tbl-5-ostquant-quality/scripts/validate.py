from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODE_MAP = {
    "OSTQuant": "exact_eager_maskfix_acc",
    "SCNA-8": "scna_d8_maskfix_acc",
    "SCNA-16": "scna_d16_maskfix_acc",
    "SCNA-32": "scna_d32_maskfix_acc",
}
QUANT_MAP = {"W6A6": "w6a6kv6", "W4A4": "w4a4kv4"}
MODEL_MAP = {"llama2": "llama2_7b", "llama3": "llama3_8b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--ppl-tolerance", type=float, default=0.03)
    parser.add_argument("--accuracy-tolerance-percent", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    actual_path = args.result_dir / "analysis" / "maskfix_summary.csv"
    expected_path = args.expected
    with actual_path.open() as handle:
        actual = {
            (row["model_key"], row["quant_key"], row["mode"]): row for row in csv.DictReader(handle)
        }

    comparisons = []
    with expected_path.open() as handle:
        for expected in csv.DictReader(handle):
            mode = MODE_MAP[expected["method"]]
            quant = QUANT_MAP[expected["quantization"]]
            for prefix, model in MODEL_MAP.items():
                row = actual.get((model, quant, mode))
                paper_ppl = float(expected[f"{prefix}_ppl"])
                paper_acc = float(expected[f"{prefix}_accuracy_percent"])
                reproduced_ppl = None if row is None else float(row["ppl"])
                reproduced_acc = None if row is None else 100.0 * float(row["acc_avg"])
                passed = (
                    reproduced_ppl is not None
                    and reproduced_acc is not None
                    and abs(reproduced_ppl - paper_ppl) <= args.ppl_tolerance
                    and abs(reproduced_acc - paper_acc) <= args.accuracy_tolerance_percent
                )
                comparisons.append(
                    {
                        "model": model,
                        "quantization": expected["quantization"],
                        "method": expected["method"],
                        "paper_ppl": paper_ppl,
                        "reproduced_ppl": "" if reproduced_ppl is None else reproduced_ppl,
                        "paper_accuracy_percent": paper_acc,
                        "reproduced_accuracy_percent": "" if reproduced_acc is None else reproduced_acc,
                        "pass": passed,
                    }
                )

    analysis_dir = args.result_dir / "analysis"
    with (analysis_dir / "paper_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    passed = sum(bool(row["pass"]) for row in comparisons)
    summary = {
        "status": "pass" if passed == len(comparisons) else "fail",
        "comparisons": len(comparisons),
        "passed": passed,
        "ppl_tolerance": args.ppl_tolerance,
        "accuracy_tolerance_percent": args.accuracy_tolerance_percent,
    }
    (analysis_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (analysis_dir / "validation.md").write_text(
        "# Table 5 reproduction\n\n"
        f"Status: **{summary['status']}**\n\n"
        f"Passed {passed}/{len(comparisons)} paper-value comparisons.\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
