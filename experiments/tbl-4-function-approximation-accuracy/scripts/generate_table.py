from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHODS = ["taylor", "frac_t", "interp", "frac_i", "linearlut", "nnlut", "tlut", "ours"]
FIELDS = [f"mse_{method}" for method in METHODS] + [f"mae_{method}" for method in METHODS]
METHOD_LABELS = {
    "taylor": "Taylor",
    "frac_t": "Frac-T",
    "interp": "Interp",
    "frac_i": "Frac-I",
    "linearlut": "LinearLUT",
    "nnlut": "NN-LUT",
    "tlut": "T-LUT",
    "ours": "SCNA",
}
LABELS = [METHOD_LABELS[method] + " MSE" for method in METHODS] + [
    METHOD_LABELS[method] + " MAE" for method in METHODS
]


def table_lines(title: str, rows: list[dict[str, str]], generated_path: Path) -> list[str]:
    generated = {row["function"]: row for row in csv.DictReader(generated_path.open())}
    lines = [
        f"# {title}",
        "",
        "| Function | " + " | ".join(LABELS) + " |",
        "| --- | " + " | ".join("---" for _ in FIELDS) + " |",
    ]
    for reference in rows:
        row = dict(reference)
        actual = generated[row["function"]]
        row["mse_ours"] = f'{float(actual["mse"]):.3e}'
        row["mae_ours"] = f'{float(actual["mae"]):.3e}'
        lines.append("| " + row["function"] + " | " + " | ".join(row[field] for field in FIELDS) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference-generated", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.expected.open()))
    lines = table_lines("Table 4 — nonlinear approximation accuracy (SCNA-16)", rows, args.generated)
    if args.reference_generated is not None:
        lines.extend(
            [""]
            + table_lines(
                "SCNA-32 reference",
                rows,
                args.reference_generated,
            )
        )
    lines.extend(
        [
            "",
            "SCNA-16 and SCNA-32 values above come from generated raw predictions. Other method columns are literature reference values and are never inputs to the SCNA reproduction command.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
