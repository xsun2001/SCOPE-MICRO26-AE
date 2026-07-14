#!/usr/bin/env python3
"""Extract native synthesis reports and reproduce the Figure 18 fit."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_REPORTS = HERE.parents[1] / "hardware" / "synthesis" / "reports"
AREA_RE = re.compile(r"^Total cell area:\s+([0-9.eE+-]+)\s*$", re.MULTILINE)
POWER_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*\s+\S+\s+\S+\s+\S+\s+([0-9.eE+-]+)\s+100\.0\s*$",
    re.MULTILINE,
)
TIMING_RE = re.compile(r"slack \((MET|VIOLATED)\)\s+(-?[0-9.eE+-]+)")

DESIGN_LABELS = {
    "baseline": "Baseline",
    "scope": "SCOPE",
    "onesa": "OneSA",
    "fusemax": "FuseMax",
    "fsa": "FSA",
}
TYPE_LABELS = {
    "fp16-fp16": ("FP16", "FP16"),
    "fp8-fp16": ("FP8", "FP16"),
    "int16-int32": ("INT16", "INT32"),
    "int8-int32": ("INT8", "INT32"),
}
DESIGN_ORDER = {name: index for index, name in enumerate(DESIGN_LABELS)}
TYPE_ORDER = {name: index for index, name in enumerate(TYPE_LABELS)}

# This is the corrected FSA FP8 report that contains the two hierarchy rows
# used by the paper. All other samples use whole-array totals divided by N^2.
FSA_FP8_AREA_INSTANCE = "mesh_1_2"
FSA_FP8_POWER_INSTANCE = "mesh_3_3"

REPORT_FIELDS = [
    "design",
    "data_type",
    "acc_type",
    "array_size",
    "area_scope",
    "power_scope",
    "report_area_um2",
    "report_power_mw",
    "area_per_pe_um2",
    "power_per_pe_mw",
]
FIGURE_FIELDS = [
    "data_type",
    "acc_type",
    "design",
    "sample_count",
    "array_sizes",
    "area_per_pe_um2",
    "power_per_pe_mw",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def match_float(pattern: re.Pattern[str], text: str, path: Path) -> float:
    match = pattern.search(text)
    if not match:
        raise ValueError(f"cannot extract a value from {path}")
    return float(match.group(1))


def hierarchy_area(text: str, instance: str, path: Path) -> float:
    pattern = re.compile(rf"^{re.escape(instance)}\s+([0-9.eE+-]+)\s+", re.MULTILINE)
    return match_float(pattern, text, path)


def hierarchy_power(text: str, instance: str, path: Path) -> float:
    pattern = re.compile(
        rf"^\s*{re.escape(instance)}\s+\([^)]*\)\s+\S+\s+\S+\s+\S+\s+([0-9.eE+-]+)\s+",
        re.MULTILINE,
    )
    return match_float(pattern, text, path)


def extract_reports(reports_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for area_path in reports_root.glob("*/*/n??/area.rpt"):
        report_dir = area_path.parent
        design_key = report_dir.parents[1].name
        type_key = report_dir.parent.name
        if design_key not in DESIGN_LABELS or type_key not in TYPE_LABELS:
            raise ValueError(f"unexpected report path: {report_dir}")
        array_size = int(report_dir.name.removeprefix("n"))
        power_path = report_dir / "power.rpt"
        timing_path = report_dir / "timing.rpt"
        area_text = area_path.read_text(encoding="utf-8", errors="replace")
        power_text = power_path.read_text(encoding="utf-8", errors="replace")
        timing_text = timing_path.read_text(encoding="utf-8", errors="replace")

        timing_matches = TIMING_RE.findall(timing_text)
        if not timing_matches or timing_matches[-1][0] != "MET":
            raise ValueError(f"timing is not MET in {timing_path}")

        if design_key == "fsa" and type_key == "fp8-fp16":
            report_area = hierarchy_area(area_text, FSA_FP8_AREA_INSTANCE, area_path)
            report_power = hierarchy_power(power_text, FSA_FP8_POWER_INSTANCE, power_path)
            area_scope = FSA_FP8_AREA_INSTANCE
            power_scope = FSA_FP8_POWER_INSTANCE
            area_per_pe = report_area
            power_per_pe = report_power
        else:
            report_area = match_float(AREA_RE, area_text, area_path)
            report_power = match_float(POWER_RE, power_text, power_path)
            area_scope = "whole_array"
            power_scope = "whole_array"
            area_per_pe = report_area / array_size**2
            power_per_pe = report_power / array_size**2

        data_type, acc_type = TYPE_LABELS[type_key]
        rows.append(
            {
                "design": DESIGN_LABELS[design_key],
                "data_type": data_type,
                "acc_type": acc_type,
                "array_size": array_size,
                "area_scope": area_scope,
                "power_scope": power_scope,
                "report_area_um2": f"{report_area:.6f}",
                "report_power_mw": f"{report_power:.6f}",
                "area_per_pe_um2": f"{area_per_pe:.9f}",
                "power_per_pe_mw": f"{power_per_pe:.12f}",
            }
        )

    rows.sort(
        key=lambda row: (
            TYPE_ORDER[next(key for key, value in TYPE_LABELS.items() if value == (row["data_type"], row["acc_type"]))],
            DESIGN_ORDER[next(key for key, value in DESIGN_LABELS.items() if value == row["design"])],
            int(row["array_size"]),
        )
    )
    if len(rows) != 112:
        raise ValueError(f"expected 112 filtered report sets, found {len(rows)}")
    return rows


def fit_per_pe(report_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in report_rows:
        grouped[(str(row["data_type"]), str(row["acc_type"]), str(row["design"]))].append(row)

    figure_rows: list[dict[str, object]] = []
    for (data_type, acc_type, design), samples in grouped.items():
        # Constant least-squares fit on the normalized samples:
        #   argmin_c sum_N (total_N/N^2 - c)^2 = arithmetic mean(total_N/N^2)
        area = sum(float(sample["area_per_pe_um2"]) for sample in samples) / len(samples)
        power = sum(float(sample["power_per_pe_mw"]) for sample in samples) / len(samples)
        sizes = sorted(int(sample["array_size"]) for sample in samples)
        figure_rows.append(
            {
                "data_type": data_type,
                "acc_type": acc_type,
                "design": design,
                "sample_count": len(samples),
                "array_sizes": ";".join(map(str, sizes)),
                "area_per_pe_um2": f"{area:.3f}",
                "power_per_pe_mw": f"{power:.3f}",
            }
        )

    type_labels = list(TYPE_LABELS.values())
    design_labels = list(DESIGN_LABELS.values())
    figure_rows.sort(
        key=lambda row: (
            type_labels.index((row["data_type"], row["acc_type"])),
            design_labels.index(row["design"]),
        )
    )
    return figure_rows


def verify(actual: Path, expected: Path, fields: list[str]) -> None:
    actual_rows = read_csv(actual)
    expected_rows = read_csv(expected)
    if actual_rows != expected_rows:
        raise ValueError(f"{actual.name} does not match {expected}")
    print(f"PASS {actual.name}: {len(actual_rows)} rows match the paper bundle")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dir", type=Path, default=HERE / "expected-results")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    report_rows = extract_reports(args.reports.resolve())
    figure_rows = fit_per_pe(report_rows)
    report_csv = output_dir / "report_values.csv"
    figure_csv = output_dir / "figure18.csv"
    write_csv(report_csv, REPORT_FIELDS, report_rows)
    write_csv(figure_csv, FIGURE_FIELDS, figure_rows)
    verify(report_csv, args.expected_dir / "report_values.csv", REPORT_FIELDS)
    verify(figure_csv, args.expected_dir / "figure18.csv", FIGURE_FIELDS)

    subprocess.run(
        [sys.executable, str(HERE / "plot.py"), "--input", str(figure_csv), "--output-dir", str(output_dir)],
        check=True,
    )
    print("Figure 18 reproduced from 112 native report sets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
