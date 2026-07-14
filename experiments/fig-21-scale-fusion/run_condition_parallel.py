import argparse
import concurrent.futures
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List


EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))
SOURCE_EXPERIMENT_DIR = EXPERIMENT_DIR

if SOURCE_EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, SOURCE_EXPERIMENT_DIR)

from simulate_prefill_attention import (  # type: ignore
    VARIANTS,
    ExperimentSpec,
    ModelSpec,
    build_speedup_rows,
    ensure_dir,
    ensure_scalesim_temp_dir,
    load_experiment_spec,
    load_single_card_system,
    profile_attention_variant,
    write_csv,
    write_summary,
)


def _build_worker_spec(base_spec: ExperimentSpec, case, prefill_length: int) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_name=base_spec.experiment_name,
        experiment_config_path=base_spec.experiment_config_path,
        compile_mode=base_spec.compile_mode,
        batch_size=base_spec.batch_size,
        adjust_global_buffer_to_memory_capacity=base_spec.adjust_global_buffer_to_memory_capacity,
        ignore_hbm_bottleneck=base_spec.ignore_hbm_bottleneck,
        ignore_onchip_io_bottleneck=base_spec.ignore_onchip_io_bottleneck,
        prefill_lengths=[prefill_length],
        model=ModelSpec(
            name=base_spec.model.name,
            hidden_size=base_spec.model.hidden_size,
            num_heads=base_spec.model.num_heads,
        ),
        cases=[case],
    )


def _profile_task(payload: Dict[str, object]) -> Dict[str, object]:
    os.environ["LLMCOMPASS_INT8_MATCH_FP16_TILES"] = "0"
    os.environ["LLMCOMPASS_CUSTOMSA_SEARCH_TILES"] = "0"
    os.environ["LLMCOMPASS_INT8_SOFTMAX_CONVERSION"] = str(payload["softmax_conversion"])

    spec = load_experiment_spec(str(payload["experiment_config"]))
    spec.compile_mode = str(payload["compile_mode"])
    spec.batch_size = int(payload["batch_size"])
    spec.adjust_global_buffer_to_memory_capacity = bool(
        payload["adjust_global_buffer_to_memory_capacity"]
    )
    spec.ignore_hbm_bottleneck = bool(payload["ignore_hbm_bottleneck"])
    spec.ignore_onchip_io_bottleneck = bool(payload["ignore_onchip_io_bottleneck"])

    case_name = str(payload["case_name"])
    case = next(case for case in spec.cases if case.name == case_name)
    prefill_length = int(payload["prefill_length"])
    variant_name = str(payload["variant_name"])
    variant_spec = next(variant for variant in VARIANTS if variant["name"] == variant_name)

    system, raw_specs = load_single_card_system(
        case.system_config_path,
        adjust_global_buffer_to_memory_capacity=spec.adjust_global_buffer_to_memory_capacity,
        ignore_onchip_io_bottleneck=spec.ignore_onchip_io_bottleneck,
    )
    configured_hbm_bandwidth = float(system.device.io_module.bandwidth)
    configured_onchip_bandwidth = float(
        system.device.compute_module.l2_bandwidth_per_cycle
        * system.device.compute_module.clock_freq
    )
    if spec.ignore_hbm_bottleneck:
        system.device.io_module.bandwidth = float("inf")
    active_hbm_bandwidth = float(system.device.io_module.bandwidth)
    active_onchip_bandwidth = float(
        system.device.compute_module.l2_bandwidth_per_cycle
        * system.device.compute_module.clock_freq
    )

    worker_spec = _build_worker_spec(spec, case, prefill_length)
    row = profile_attention_variant(
        worker_spec,
        case,
        system,
        raw_specs,
        configured_hbm_bandwidth,
        active_hbm_bandwidth,
        configured_onchip_bandwidth,
        active_onchip_bandwidth,
        prefill_length,
        variant_spec,
    )
    return {
        "row": row,
        "case_name": case_name,
        "raw_specs": raw_specs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--softmax-conversion", type=int, choices=[0, 1], required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--prefill-lengths")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ignore-hbm-bottleneck", action="store_true")
    parser.add_argument("--ignore-onchip-io-bottleneck", action="store_true")
    parser.add_argument(
        "--variants",
        help="Optional comma-separated subset of baseline,flashattention,customsa.",
    )
    args = parser.parse_args()

    ensure_scalesim_temp_dir()
    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    spec = load_experiment_spec(os.path.abspath(args.experiment_config))
    if args.prefill_lengths:
        spec.prefill_lengths = [int(item) for item in args.prefill_lengths.split(",") if item.strip()]
    if args.batch_size is not None:
        spec.batch_size = args.batch_size
    if args.ignore_hbm_bottleneck:
        spec.ignore_hbm_bottleneck = True
    if args.ignore_onchip_io_bottleneck:
        spec.ignore_onchip_io_bottleneck = True

    selected_variants = VARIANTS
    if args.variants:
        requested = {item.strip() for item in args.variants.split(",") if item.strip()}
        known = {variant["name"] for variant in VARIANTS}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown variants: {sorted(unknown)}")
        selected_variants = [variant for variant in VARIANTS if variant["name"] in requested]

    tasks: List[Dict[str, object]] = []
    for case in spec.cases:
        for prefill_length in spec.prefill_lengths:
            for variant in selected_variants:
                tasks.append(
                    {
                        "experiment_config": spec.experiment_config_path,
                        "compile_mode": spec.compile_mode,
                        "batch_size": spec.batch_size,
                        "adjust_global_buffer_to_memory_capacity": spec.adjust_global_buffer_to_memory_capacity,
                        "ignore_hbm_bottleneck": spec.ignore_hbm_bottleneck,
                        "ignore_onchip_io_bottleneck": spec.ignore_onchip_io_bottleneck,
                        "softmax_conversion": args.softmax_conversion,
                        "case_name": case.name,
                        "prefill_length": prefill_length,
                        "variant_name": variant["name"],
                    }
                )

    rows = []
    case_specs: Dict[str, Dict[str, object]] = {}
    max_workers = max(1, min(args.jobs, len(tasks)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_profile_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            row = result["row"]
            rows.append(row)
            case_specs.setdefault(str(result["case_name"]), result["raw_specs"])
            required_hbm = row["required_hbm_bandwidth_tbps"]
            required_hbm_value = float(required_hbm) if required_hbm is not None else 0.0
            print(
                f"case={row['case_name']:<10} dtype={row['data_type']:<4} seq={int(row['prefill_length']):>5d} "
                f"variant={row['variant']:<14} total_ms={float(row['total_latency_ms']):>10.3f} "
                f"bottleneck={row['bottleneck_component']:<11} component_ms={float(row['bottleneck_ms']):>10.3f} "
                f"req_hbm_tbps={required_hbm_value:>8.3f}",
                flush=True,
            )

    rows.sort(key=lambda row: (str(row["case_name"]), int(row["prefill_length"]), str(row["variant"])))
    speedup_rows = build_speedup_rows(rows) if "baseline" in {v["name"] for v in selected_variants} else []

    write_csv(os.path.join(output_dir, "attention_latency.csv"), rows)
    if speedup_rows:
        write_csv(os.path.join(output_dir, "case_speedups.csv"), speedup_rows)
        write_summary(os.path.join(output_dir, "summary.md"), rows, speedup_rows)
    else:
        with open(os.path.join(output_dir, "summary.md"), "w") as f:
            f.write("# Attention Subset Summary\n\n")
            f.write("This condition intentionally profiles only: ")
            f.write(", ".join(variant["name"] for variant in selected_variants))
            f.write(".\n")

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "experiment_name": spec.experiment_name,
                "experiment_config_path": spec.experiment_config_path,
                "compile_mode": spec.compile_mode,
                "batch_size": spec.batch_size,
                "jobs": max_workers,
                "adjust_global_buffer_to_memory_capacity": spec.adjust_global_buffer_to_memory_capacity,
                "ignore_hbm_bottleneck": spec.ignore_hbm_bottleneck,
                "ignore_onchip_io_bottleneck": spec.ignore_onchip_io_bottleneck,
                "prefill_lengths": spec.prefill_lengths,
                "variants": selected_variants,
                "attention_env": {
                    "LLMCOMPASS_INT8_MATCH_FP16_TILES": "0",
                    "LLMCOMPASS_INT8_SOFTMAX_CONVERSION": str(args.softmax_conversion),
                    "LLMCOMPASS_CUSTOMSA_SEARCH_TILES": "0",
                },
                "model": asdict(spec.model),
                "cases": [asdict(case) for case in spec.cases],
                "case_specs": case_specs,
            },
            f,
            indent=2,
        )

    with open(os.path.join(output_dir, "cache_stats.json"), "w") as f:
        json.dump({"aggregated": {}, "per_case": {}}, f, indent=2)


if __name__ == "__main__":
    main()
