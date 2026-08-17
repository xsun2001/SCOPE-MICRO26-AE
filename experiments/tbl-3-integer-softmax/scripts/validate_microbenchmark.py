from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METHOD_COLUMNS = {
    "fp_stream_softmax_u8": ("fp_stream_softmax_u8_ms", "fp_stream_softmax_u8_mean_ms", "fp_stream_softmax_u8_stddev_ms"),
    "illm_di_softmax_u8": ("illm_di_softmax_u8_ms", "illm_di_softmax_u8_mean_ms", "illm_di_softmax_u8_stddev_ms"),
    "intattention_idx_softmax_u8": ("intattention_idx_softmax_u8_ms", "intattention_idx_softmax_u8_mean_ms", "intattention_idx_softmax_u8_stddev_ms"),
    "memory_floor_1read_1write": ("memory_floor_1read_1write_ms", "memory_floor_1read_1write_mean_ms", "memory_floor_1read_1write_stddev_ms"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw repetitions from the Table 3 H100 softmax microbenchmark.")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    failures: list[str] = []

    metadata = json.loads((args.result_dir / "metadata.json").read_text())
    with (args.result_dir / "softmax_latency.csv").open() as handle:
        summary_rows = list(csv.DictReader(handle))
    with (args.result_dir / "raw_samples.csv").open() as handle:
        raw_rows = list(csv.DictReader(handle))

    if metadata.get("measurement") != "physical_h100_cuda" or "H100" not in str(metadata.get("device")):
        failures.append("metadata does not identify a physical H100 CUDA measurement")
    if metadata.get("correctness_check") is not True:
        failures.append("the Torch output correctness check was not enabled")
    seq_lens = [int(value) for value in metadata.get("seq_lens", [])]
    runs = int(summary_rows[0]["runs"]) if summary_rows else 0
    summaries: dict[int, dict[str, str]] = {}
    for row in summary_rows:
        seq_len = int(row["seq_len"])
        if seq_len in summaries:
            failures.append(f"duplicate summary row for seq_len={seq_len}")
        summaries[seq_len] = row
        try:
            max_abs_diff = int(row["max_abs_diff_fp_vs_torch"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"missing Torch correctness result for seq_len={seq_len}")
        else:
            if max_abs_diff > 1:
                failures.append(f"FP/Torch output difference exceeds one U8 level for seq_len={seq_len}: {max_abs_diff}")
    if set(summaries) != set(seq_lens):
        failures.append(f"summary sequence lengths {sorted(summaries)} do not match metadata {sorted(seq_lens)}")

    samples: dict[tuple[int, str], list[tuple[int, float]]] = {}
    seen: set[tuple[int, str, int]] = set()
    for row in raw_rows:
        key = (int(row["seq_len"]), row["method"], int(row["repetition"]))
        if key in seen:
            failures.append(f"duplicate raw repetition: {key}")
            continue
        seen.add(key)
        latency = float(row["latency_ms"])
        if not math.isfinite(latency) or latency <= 0:
            failures.append(f"invalid raw latency for {key}: {latency}")
        samples.setdefault(key[:2], []).append((key[2], latency))

    expected_keys = {(seq_len, method) for seq_len in seq_lens for method in METHOD_COLUMNS}
    missing_keys = sorted(expected_keys - set(samples))
    extra_keys = sorted(set(samples) - expected_keys)
    if missing_keys:
        failures.append(f"missing raw sample groups: {missing_keys}")
    if extra_keys:
        failures.append(f"unexpected raw sample groups: {extra_keys}")

    recomputed = 0
    for key in sorted(expected_keys & set(samples)):
        seq_len, method = key
        repetitions = sorted(samples[key])
        if [index for index, _ in repetitions] != list(range(1, runs + 1)):
            failures.append(f"non-contiguous repetitions for {key}")
            continue
        values = [value for _, value in repetitions]
        median_column, mean_column, stddev_column = METHOD_COLUMNS[method]
        summary = summaries[seq_len]
        expected_stats = (
            statistics.median(values),
            statistics.fmean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
        stored_stats = tuple(float(summary[column]) for column in (median_column, mean_column, stddev_column))
        for label, stored, expected in zip(("median", "mean", "stddev"), stored_stats, expected_stats):
            if not math.isclose(stored, expected, rel_tol=1e-12, abs_tol=1e-12):
                failures.append(f"{key} {label} mismatch: stored={stored} recomputed={expected}")
        recomputed += 1

    expected_raw_count = len(seq_lens) * len(METHOD_COLUMNS) * runs
    if len(raw_rows) != expected_raw_count:
        failures.append(f"expected {expected_raw_count} raw samples, found {len(raw_rows)}")

    payload = {
        "status": "fail" if failures else "pass",
        "sequence_lengths": len(seq_lens),
        "methods": len(METHOD_COLUMNS),
        "repetitions_per_method": runs,
        "raw_samples": len(raw_rows),
        "recomputed_groups": recomputed,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
