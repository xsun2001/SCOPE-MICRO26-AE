from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.expected.open()))
    methods = ["taylor", "frac_t", "interp", "frac_i", "linearlut", "nnlut", "tlut", "ours"]
    fields = [f"mse_{method}" for method in methods] + [f"mae_{method}" for method in methods]
    labels = [method.upper() + " MSE" for method in methods] + [method.upper() + " MAE" for method in methods]
    lines = [
        "# Table 4 — nonlinear approximation accuracy",
        "",
        "| Function | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + row["function"] + " | " + " | ".join(row[field] for field in fields) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
