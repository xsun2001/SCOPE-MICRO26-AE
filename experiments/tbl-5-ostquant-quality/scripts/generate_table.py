from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_ORDER = ("OSTQuant", "SCNA-8", "SCNA-16", "SCNA-32", "BF16 Baseline")
COLUMNS = (
    ("W6A6", "llama2_7b", "W6A6 Llama-2-7B"),
    ("W6A6", "llama3_8b", "W6A6 Llama-3-8B"),
    ("W4A4", "llama2_7b", "W4A4 Llama-2-7B"),
    ("W4A4", "llama3_8b", "W4A4 Llama-3-8B"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.comparison.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["method"] == "FP16 Baseline":
            row["method"] = "BF16 Baseline"
    indexed = {
        (row["method"], row["quantization"], row["model"]): row for row in rows
    }
    headers = ["Method", *[label for _, _, label in COLUMNS]]
    lines = [
        "# Table 5 — OSTQuant and SCNA model quality",
        "",
        "Each result is `WikiText-2 PPL / four-task accuracy (%)`, where accuracy is the unweighted mean of ARC-Easy, HellaSwag, PIQA, and WinoGrande.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *("---:" for _ in COLUMNS)]) + " |",
    ]
    for method in METHOD_ORDER:
        cells = []
        for quant, model, _ in COLUMNS:
            row = indexed[(method, quant, model)]
            cells.append(
                f"{float(row['reproduced_ppl']):.2f} / {float(row['reproduced_accuracy_percent']):.2f}"
            )
        label = method if method in {"OSTQuant", "BF16 Baseline"} else f"w/ {method}"
        lines.append("| " + " | ".join([label, *cells]) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
