#!/usr/bin/env python3
"""Compare a CPU AE rerun with the rounded values reported in SCOPE-revision.pdf."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LENGTHS = [2048, 4096, 8192, 16384, 32768]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latency(path: Path, case: str, dtype: str, length: int, variant: str) -> float:
    rows = read_csv(path)
    row = next(
        item
        for item in rows
        if item["case_name"] == f"{case}_{dtype}"
        and int(item["prefill_length"]) == length
        and item["variant"] == variant
    )
    return float(row["total_latency_ms"])


def max_csv_delta(
    actual_path: Path,
    expected_path: Path,
    key_fields: list[str],
    numeric_fields: list[str],
) -> float:
    actual_rows = read_csv(actual_path)
    expected_rows = read_csv(expected_path)
    expected_index = {
        tuple(row[field] for field in key_fields): row for row in expected_rows
    }
    deltas = []
    for row in actual_rows:
        key = tuple(row[field] for field in key_fields)
        expected = expected_index[key]
        for field in numeric_fields:
            deltas.append(abs(float(row[field]) - float(expected[field])))
    if not deltas:
        raise ValueError(f"No rows compared between {actual_path} and {expected_path}")
    return max(deltas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_id = args.run_id
    experiments = ROOT / "experiments"
    figure13_root = experiments / "fig-13-prefill-attention" / "actual-results" / run_id
    figure14_root = experiments / "fig-14-full-prefill" / "actual-results" / run_id
    figure15_root = experiments / "fig-15-b300-sensitivity" / "actual-results" / run_id
    table3_root = experiments / "tbl-3-integer-softmax" / "actual-results" / run_id
    figure18_root = experiments / "fig-18-pe-area-power" / "actual-results" / run_id
    figure19_root = experiments / "fig-19-hardware-comparison" / "actual-results" / run_id
    records: list[dict[str, object]] = []

    def check(
        experiment: str,
        metric: str,
        actual: float,
        expected: float,
        tolerance: float,
    ) -> None:
        delta = abs(actual - expected)
        records.append(
            {
                "experiment": experiment,
                "metric": metric,
                "actual": actual,
                "expected": expected,
                "tolerance": tolerance,
                "status": "PASS" if delta <= tolerance else "FAIL",
                "absolute_delta": delta,
            }
        )

    def presence(experiment: str, metric: str, present: bool, detail: str = "") -> None:
        records.append(
            {
                "experiment": experiment,
                "metric": metric,
                "actual": detail or str(present),
                "expected": "present",
                "tolerance": "n/a",
                "status": "PASS" if present else "FAIL",
                "absolute_delta": "n/a",
            }
        )

    # Figures 13 and 21: FlashAttention includes conversion while SCOPE does not.
    platform = {
        "b200": ("gpu", "b200"),
        "awsv4": ("aws", "awsv4"),
        "tpuv6e": ("tpu", "tpuv6e"),
    }
    fp16_32k = {"b200": 1.34, "awsv4": 1.34, "tpuv6e": 1.70}
    int8_32k = {"b200": 3.05, "awsv4": 2.51, "tpuv6e": 2.81}
    ablation = {
        ("b200", 2048): 1.06,
        ("b200", 32768): 1.11,
        ("awsv4", 4096): 1.12,
        ("awsv4", 8192): 1.73,
        ("awsv4", 16384): 1.91,
        ("awsv4", 32768): 1.97,
        ("tpuv6e", 2048): 1.10,
        ("tpuv6e", 32768): 1.46,
    }
    if figure13_root.exists():
        for device, (prefix, case) in platform.items():
            conv = figure13_root / device / f"{prefix}_conv" / "attention_latency.csv"
            no_conv = figure13_root / device / f"{prefix}_no_conv" / "attention_latency.csv"
            if not conv.exists() or not no_conv.exists():
                presence("Figure 13/21", f"{device} result CSVs", False)
                continue
            for dtype, expected in (("fp16", fp16_32k[device]), ("int8", int8_32k[device])):
                speedup = latency(conv, case, dtype, 32768, "flashattention") / latency(
                    no_conv, case, dtype, 32768, "customsa"
                )
                check("Figure 13", f"{device} {dtype} 32K speedup", speedup, expected, 0.01)
            for (target, length), expected in ablation.items():
                if target != device:
                    continue
                penalty = latency(conv, case, "int8", length, "customsa") / latency(
                    no_conv, case, "int8", length, "customsa"
                )
                check("Figure 21", f"{device} {length} conversion-fusion speedup", penalty, expected, 0.01)
            expected_root = (
                experiments / "fig-13-prefill-attention" / "expected-results" / device
            )
            for condition in (f"{prefix}_conv", f"{prefix}_no_conv"):
                actual_csv = figure13_root / device / condition / "attention_latency.csv"
                expected_csv = expected_root / condition / "attention_latency.csv"
                archive_delta = max_csv_delta(
                    actual_csv,
                    expected_csv,
                    ["case_name", "data_type", "prefill_length", "variant"],
                    ["total_latency_ms"],
                )
                check(
                    "Figure 13/21 archive audit",
                    f"{device} {condition} all reproduced latency rows",
                    archive_delta,
                    0.0,
                    1e-12,
                )

    # Figures 14 and 15: compare the paper-ready CSV produced by the rerun.
    paper_main = figure14_root / "paper_figures" / "paper_main_e2e_speedups.csv"
    if paper_main.exists():
        rows = read_csv(paper_main)

        def e2e(dtype: str, device: str, length: int) -> float:
            row = next(
                item
                for item in rows
                if item["dtype"] == dtype
                and item["device_key"] == device
                and int(item["context_length"]) == length
            )
            return float(row["end_to_end_speedup_x"])

        expected_main = {
            ("FP16", "awsv4", 32768): 1.207,
            ("FP16", "b200", 32768): 1.183,
            ("FP16", "tpuv6e", 32768): 1.473,
            ("FP16", "awsv4", 524288): 1.329,
            ("FP16", "b200", 524288): 1.341,
            ("FP16", "tpuv6e", 524288): 1.681,
            ("INT8", "awsv4", 524288): 1.28,
            ("INT8", "b200", 524288): 2.69,
            ("INT8", "tpuv6e", 524288): 1.91,
        }
        for key, expected in expected_main.items():
            check("Figure 14", f"{' '.join(map(str, key))} E2E speedup", e2e(*key), expected, 0.005)

    paper_b300 = figure15_root / "paper_figures" / "paper_b300_speedups.csv"
    if paper_b300.exists():
        rows = read_csv(paper_b300)
        for dtype, expected_attn, expected_e2e in (
            ("FP16", 1.09, 1.08),
            ("INT8", 1.94, 1.90),
        ):
            row = next(
                item
                for item in rows
                if item["dtype"] == dtype and int(item["context_length"]) == 524288
            )
            check("Figure 15", f"B300 {dtype} 512K attention speedup", float(row["attention_speedup_x"]), expected_attn, 0.01)
            check("Figure 15", f"B300 {dtype} 512K E2E speedup", float(row["end_to_end_speedup_x"]), expected_e2e, 0.01)

    expected_paper14 = experiments / "fig-14-full-prefill" / "expected-results" / "paper_figures"
    expected_paper15 = experiments / "fig-15-b300-sensitivity" / "expected-results" / "paper_figures"
    if paper_main.exists():
        delta = max_csv_delta(
            paper_main,
            expected_paper14 / "paper_main_e2e_speedups.csv",
            ["dtype", "device_key", "context_length"],
            ["end_to_end_speedup_x", "flashattention_model_ms", "customsa_model_ms"],
        )
        check("Figure 14 archive audit", "all 54 paper CSV rows", delta, 0.0, 1e-12)
    if paper_b300.exists():
        delta = max_csv_delta(
            paper_b300,
            expected_paper15 / "paper_b300_speedups.csv",
            ["dtype", "context_length"],
            [
                "attention_speedup_x",
                "end_to_end_speedup_x",
                "flashattention_model_ms",
                "customsa_model_ms",
            ],
        )
        check("Figure 15 archive audit", "all 18 paper CSV rows", delta, 0.0, 1e-12)

    # Table 3: useful attention throughput, calculated from the modeled latency.
    table3 = table3_root / "h100_int8" / "attention_latency.csv"
    if table3.exists():
        rows = read_csv(table3)
        expected_tflops = {
            "customsa": [1130.86, 1526.51, 1672.82, 1713.89],
            "illm": [641.65, 746.36, 772.04, 782.08],
            "intattention": [888.73, 1093.30, 1170.41, 1104.60],
        }
        for variant, expected_values in expected_tflops.items():
            for length, expected in zip([2048, 4096, 8192, 16384], expected_values, strict=True):
                row = next(
                    item
                    for item in rows
                    if item["variant"] == variant and int(item["prefill_length"]) == length
                )
                useful_flops = 4.0 * 32 * length * length * 128
                actual = useful_flops / (float(row["total_latency_ms"]) / 1000.0) / 1e12
                check("Table 3", f"{variant} {length} throughput TFLOP/s", actual, expected, 0.01)

    # Figure 18 is fitted directly from native reports; Figure 19 consumes that fit.
    figure18_csv = figure18_root / "figure18.csv"
    figure19_csv = figure19_root / "figure19.csv"
    fig18_rows = read_csv(figure18_csv)
    fig18_index = {
        (row["data_type"], row["acc_type"], row["design"]): row
        for row in fig18_rows
    }
    scope_area_ratios = []
    scope_power_ratios = []
    for data_type, acc_type in (("FP16", "FP16"), ("FP8", "FP16"), ("INT16", "INT32"), ("INT8", "INT32")):
        baseline = fig18_index[(data_type, acc_type, "Baseline")]
        scope = fig18_index[(data_type, acc_type, "SCOPE")]
        scope_area_ratios.append(float(scope["area_per_pe_um2"]) / float(baseline["area_per_pe_um2"]))
        scope_power_ratios.append(float(scope["power_per_pe_mw"]) / float(baseline["power_per_pe_mw"]))
    check("Figure 18", "SCOPE minimum per-PE area overhead", min(scope_area_ratios), 1.09, 0.01)
    check("Figure 18", "SCOPE maximum per-PE area overhead", max(scope_area_ratios), 1.44, 0.01)
    check("Figure 18", "SCOPE minimum per-PE power overhead", min(scope_power_ratios), 1.18, 0.01)
    check("Figure 18", "SCOPE maximum per-PE power overhead", max(scope_power_ratios), 1.34, 0.01)

    fig19_rows = read_csv(figure19_csv)
    fig19_index = {(row["acc_type"], row["method"]): row for row in fig19_rows}
    area_reductions = []
    power_reductions = []
    for acc_type in ("FP16", "INT32"):
        scna = fig19_index[(acc_type, "SCNA-8")]
        for row in fig19_rows:
            if row["acc_type"] != acc_type or row["method"] in ("SCNA-8", "SCNA-16"):
                continue
            area_reductions.append(float(row["area_um2"]) / float(scna["area_um2"]))
            power_reductions.append(float(row["power_mw"]) / float(scna["power_mw"]))
    area_geomean = math.exp(sum(math.log(value) for value in area_reductions) / len(area_reductions))
    power_geomean = math.exp(sum(math.log(value) for value in power_reductions) / len(power_reductions))
    check("Figure 19", "SCNA-8 area reduction geomean", area_geomean, 12.8, 0.1)
    check("Figure 19", "SCNA-8 power reduction geomean", power_geomean, 9.5, 0.1)

    hardware_figures = [
        figure18_root / "figure18.png",
        figure19_root / "figure19.png",
    ]
    presence(
        "Figures 18/19",
        "rendered hardware figures",
        all(path.exists() for path in hardware_figures),
    )

    generated = ROOT / "hardware" / "rtl" / "generated" / "meshes" / "pinnacle"
    if generated.exists():
        verilog = list(generated.rglob("*.sv")) + list(generated.rglob("*.v"))
        presence("RTL", "generated Pinnacle N=8 Verilog", len(verilog) >= 4, f"{len(verilog)} Verilog files")

    synth_reports = ROOT / "hardware" / "synthesis" / "reports"
    report_values = read_csv(figure18_root / "report_values.csv")
    area_reports = list(synth_reports.rglob("area.rpt"))
    power_reports = list(synth_reports.rglob("power.rpt"))
    timing_reports = list(synth_reports.rglob("timing.rpt"))
    presence("Synthesis evidence", "paper-matching report sets extracted", len(report_values) == 112, f"{len(report_values)} rows")
    presence("Synthesis evidence", "archived Design Compiler area reports", len(area_reports) == 112, f"{len(area_reports)} files")
    presence("Synthesis evidence", "archived Design Compiler power reports", len(power_reports) == 112, f"{len(power_reports)} files")
    presence("Synthesis evidence", "archived Design Compiler timing reports", len(timing_reports) == 112, f"{len(timing_reports)} files")
    sample_text = area_reports[0].read_text(encoding="utf-8", errors="replace") if area_reports else ""
    presence("Synthesis evidence", "Synopsys DC version recorded in reports", "V-2023.12" in sample_text)
    scope_int8 = [
        row
        for row in report_values
        if row["design"] == "SCOPE" and row["data_type"] == "INT8"
    ]
    presence(
        "Synthesis evidence",
        "SCOPE INT8 synthesized array sizes",
        len(scope_int8) == 7,
        f"{len(scope_int8)} sizes",
    )
    corrected_fsa = next(
        row
        for row in report_values
        if row["design"] == "FSA" and row["data_type"] == "FP8"
    )
    presence(
        "Synthesis evidence",
        "corrected FSA FP8 paper hierarchy source",
        corrected_fsa["area_scope"] == "mesh_1_2"
        and corrected_fsa["power_scope"] == "mesh_3_3"
        and float(corrected_fsa["area_per_pe_um2"]) == 1838.214
        and float(corrected_fsa["power_per_pe_mw"]) == 0.588,
        f"area={corrected_fsa['area_scope']}, power={corrected_fsa['power_scope']}",
    )
    sample_timing = timing_reports[0].read_text(encoding="utf-8", errors="replace") if timing_reports else ""
    presence(
        "Synthesis evidence",
        "TSMC 28 nm library recorded in timing report",
        "tcbn28" in sample_timing,
    )
    rtl_manifest = read_csv(ROOT / "hardware" / "rtl" / "RTL_MANIFEST.csv")
    exact_rtl = sum(row["byte_matches_retained_upload_zip"] == "true" for row in rtl_manifest)
    overwritten_zip_rtl = sum(row["byte_matches_retained_upload_zip"] == "false" for row in rtl_manifest)
    presence("RTL provenance", "selected paper RTL directories indexed", len(rtl_manifest) == 112, f"{len(rtl_manifest)} directories")
    presence("RTL provenance", "retained upload ZIP byte matches", exact_rtl == 108, f"{exact_rtl} exact matches")
    presence("RTL provenance", "overwritten ZIP caveats explicitly indexed", overwritten_zip_rtl == 4, f"{overwritten_zip_rtl} caveats")

    csv_delta18 = max_csv_delta(
        figure18_csv,
        experiments / "fig-18-pe-area-power" / "expected-results" / "figure18.csv",
        ["data_type", "acc_type", "design"],
        ["sample_count", "area_per_pe_um2", "power_per_pe_mw"],
    )
    csv_delta19 = max_csv_delta(
        figure19_csv,
        experiments / "fig-19-hardware-comparison" / "expected-results" / "figure19.csv",
        ["acc_type", "method"],
        ["area_um2", "power_mw"],
    )
    check("Figures 18/19", "report-to-fit CSV", csv_delta18, 0.0, 1e-12)
    check("Figures 18/19", "32x32 derived CSV", csv_delta19, 0.0, 1e-12)

    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "validation" / "results" / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "validation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out_dir / "validation_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    passed = sum(record["status"] == "PASS" for record in records)
    failed = sum(record["status"] == "FAIL" for record in records)
    lines = [
        "# SCOPE AE Validation Report",
        "",
        f"Run ID: `{run_id}`",
        "",
        f"Result: **{passed} passed, {failed} failed**",
        "",
        "| Status | Experiment | Metric | Actual | Paper | Tolerance |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for record in records:
        actual = record["actual"]
        expected = record["expected"]
        if isinstance(actual, float):
            actual = f"{actual:.6g}"
        if isinstance(expected, float):
            expected = f"{expected:.6g}"
        lines.append(
            f"| {record['status']} | {record['experiment']} | {record['metric']} | "
            f"{actual} | {expected} | {record['tolerance']} |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Validation: {passed} passed, {failed} failed")
    print(f"Report: {out_dir / 'REPORT.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
