from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.comparison.open()))
    lines = [
        "# Table 5 — OSTQuant and SCNA model quality",
        "",
        "| Model | Quantization | Method | Paper PPL | Actual PPL | Paper Acc. (%) | Actual Acc. (%) | Pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[key] for key in ("model", "quantization", "method", "paper_ppl", "reproduced_ppl", "paper_accuracy_percent", "reproduced_accuracy_percent", "pass")) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
