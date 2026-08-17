from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.expected.open()))
    generated = {row["function"]: row for row in csv.DictReader(args.generated.open())}
    methods = ["taylor", "frac_t", "interp", "frac_i", "linearlut", "nnlut", "tlut", "ours"]
    fields = [f"mse_{method}" for method in methods] + [f"mae_{method}" for method in methods]
    method_labels = {
        "taylor": "Taylor",
        "frac_t": "Frac-T",
        "interp": "Interp",
        "frac_i": "Frac-I",
        "linearlut": "LinearLUT",
        "nnlut": "NN-LUT",
        "tlut": "T-LUT",
        "ours": "SCNA",
    }
    labels = [method_labels[method] + " MSE" for method in methods] + [
        method_labels[method] + " MAE" for method in methods
    ]
    lines = [
        "# Table 4 — nonlinear approximation accuracy",
        "",
        "| Function | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        actual = generated[row["function"]]
        row["mse_ours"] = f'{float(actual["mse"]):.3e}'
        row["mae_ours"] = f'{float(actual["mae"]):.3e}'
        lines.append("| " + row["function"] + " | " + " | ".join(row[field] for field in fields) + " |")
    lines.extend(
        [
            "",
            "SCNA values above come from the generated raw predictions. Other method columns are literature reference values and are never inputs to the SCNA reproduction command.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
