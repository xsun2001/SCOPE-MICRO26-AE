#!/usr/bin/env python3
"""Calculate and draw Figure 19 from the Figure 18 per-PE fit."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIELDS = ["acc_type", "method", "area_um2", "power_mw"]
METHOD_ORDER = ["SCNA-8", "SCNA-16", "OneSA", "FuseMax", "FSA", "NN-LUT", "T-LUT", "PICACHU"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def find(rows: list[dict[str, str]], data_type: str, acc_type: str, design: str) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["data_type"] == data_type and row["acc_type"] == acc_type and row["design"] == design
    )


def derive(figure18: Path, literature: Path) -> list[dict[str, object]]:
    per_pe = read_csv(figure18)
    rows: list[dict[str, object]] = []
    for data_type, acc_type in (("FP16", "FP16"), ("INT16", "INT32")):
        values = {
            design: find(per_pe, data_type, acc_type, design)
            for design in ("Baseline", "SCOPE", "OneSA", "FuseMax")
        }
        try:
            values["FSA"] = find(per_pe, data_type, acc_type, "FSA")
        except StopIteration:
            pass

        def increment(design: str, metric: str, multiplier: int) -> float:
            # Figure 19 deliberately uses the paper-rounded Figure 18 values.
            return (float(values[design][metric]) - float(values["Baseline"][metric])) * multiplier

        local = [
            ("SCNA-8", "SCOPE", 16),
            ("SCNA-16", "SCOPE", 32),
            ("OneSA", "OneSA", 32),
            ("FuseMax", "FuseMax", 32),
        ]
        if "FSA" in values:
            local.append(("FSA", "FSA", 32))
        for method, design, multiplier in local:
            rows.append(
                {
                    "acc_type": acc_type,
                    "method": method,
                    "area_um2": f"{increment(design, 'area_per_pe_um2', multiplier):.3f}",
                    "power_mw": f"{increment(design, 'power_per_pe_mw', multiplier):.3f}",
                }
            )

    rows.extend(read_csv(literature))
    rows.sort(key=lambda row: (("FP16", "INT32").index(str(row["acc_type"])), METHOD_ORDER.index(str(row["method"]))))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure18", type=Path, required=True)
    parser.add_argument("--literature", type=Path, default=HERE / "literature.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected", type=Path, default=HERE / "expected-results" / "figure19.csv")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    result_csv = output_dir / "figure19.csv"
    write_csv(result_csv, derive(args.figure18, args.literature))
    actual = read_csv(result_csv)
    expected = read_csv(args.expected)
    if actual != expected:
        raise ValueError(f"{result_csv} does not match {args.expected}")
    print(f"PASS figure19.csv: {len(actual)} rows match the paper bundle")
    subprocess.run(
        [sys.executable, str(HERE / "plot.py"), "--input", str(result_csv), "--output-dir", str(output_dir)],
        check=True,
    )
    print("Figure 19 reproduced as incremental overhead over a 32x32 baseline SA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
