import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))
LLMCOMPASS_ROOT = os.path.join(REPO_ROOT, "LLMCompass")
DEFAULT_EXPERIMENT_CONFIG = os.path.join(EXPERIMENT_DIR, "configs", "default.json")

if LLMCOMPASS_ROOT not in sys.path:
    sys.path.insert(0, LLMCOMPASS_ROOT)

from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.attention import build_attention, FlashAttention, FlashAttentionCustomSA
from software_model.matmul import Matmul
from software_model.utils import Tensor, data_type_dict


VARIANTS = [
    {
        "name": "baseline",
        "label": "Unfused Attention",
        "attention_variant": "trivial",
    },
    {
        "name": "flashattention",
        "label": "FlashAttention",
        "attention_variant": "flashattention",
    },
    {
        "name": "customsa",
        "label": "FlashAttention-CustomSA",
        "attention_variant": "flashattention_customsa",
    },
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_scalesim_temp_dir() -> None:
    ensure_dir(os.path.join(LLMCOMPASS_ROOT, "systolic_array_model", "temp"))


def reset_cache_stats() -> None:
    Matmul.reset_cache_stats()
    FlashAttention.reset_cache_stats()
    FlashAttentionCustomSA.reset_cache_stats()


def collect_cache_stats() -> Dict[str, Dict[str, float]]:
    return {
        "matmul": Matmul.cache_stats(),
        "flashattention": FlashAttention.cache_stats(),
        "flashattention_customsa": FlashAttentionCustomSA.cache_stats(),
    }


def aggregate_cache_stats(cache_stats_list: List[Dict[str, Dict[str, float]]]):
    aggregated = {}
    for cache_stats in cache_stats_list:
        for section, stats in cache_stats.items():
            target = aggregated.setdefault(section, {})
            for key, value in stats.items():
                if key.endswith("_ratio"):
                    continue
                target[key] = target.get(key, 0) + int(value)
    for section, stats in aggregated.items():
        prefixes = sorted(
            {key[: -len("_hits")] for key in stats if key.endswith("_hits")}
            | {key[: -len("_misses")] for key in stats if key.endswith("_misses")}
        )
        for prefix in prefixes:
            hits = stats.get(f"{prefix}_hits", 0)
            misses = stats.get(f"{prefix}_misses", 0)
            total = hits + misses
            stats[f"{prefix}_hit_ratio"] = hits / total if total else 0.0
    return aggregated


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    config_relative = os.path.abspath(os.path.join(base_dir, path))
    repo_relative = os.path.abspath(os.path.join(REPO_ROOT, path))
    if os.path.exists(config_relative):
        return config_relative
    if os.path.exists(repo_relative):
        return repo_relative
    return config_relative


@dataclass
class CaseSpec:
    name: str
    label: str
    system_config_path: str
    data_type_name: str

    @property
    def data_type(self):
        return data_type_dict[self.data_type_name]


@dataclass
class ModelSpec:
    name: str
    hidden_size: int
    num_heads: int

    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by num_heads={self.num_heads}."
            )
        return self.hidden_size // self.num_heads


@dataclass
class ExperimentSpec:
    experiment_name: str
    experiment_config_path: str
    compile_mode: str
    batch_size: int
    adjust_global_buffer_to_memory_capacity: bool
    ignore_hbm_bottleneck: bool
    ignore_onchip_io_bottleneck: bool
    prefill_lengths: List[int]
    model: ModelSpec
    cases: List[CaseSpec]


def load_experiment_spec(config_path: str) -> ExperimentSpec:
    with open(config_path, "r") as f:
        payload = json.load(f)
    config_dir = os.path.dirname(config_path)
    runtime_payload = payload.get("runtime", {})
    model_payload = payload["model"]
    sweep_payload = payload["sweep"]
    cases = [
        CaseSpec(
            name=case["name"],
            label=case["label"],
            system_config_path=resolve_path(config_dir, case["system_config"]),
            data_type_name=case["data_type"],
        )
        for case in payload["cases"]
    ]
    return ExperimentSpec(
        experiment_name=payload.get(
            "experiment_name",
            "prefill_attention_a100_h100_b200_sweep",
        ),
        experiment_config_path=config_path,
        compile_mode=runtime_payload.get("compile_mode", "heuristic-GPU"),
        batch_size=int(runtime_payload.get("batch_size", 1)),
        adjust_global_buffer_to_memory_capacity=bool(
            runtime_payload.get("adjust_global_buffer_to_memory_capacity", False)
        ),
        ignore_hbm_bottleneck=bool(
            runtime_payload.get("ignore_hbm_bottleneck", False)
        ),
        ignore_onchip_io_bottleneck=bool(
            runtime_payload.get("ignore_onchip_io_bottleneck", False)
        ),
        prefill_lengths=[int(item) for item in sweep_payload["prefill_lengths"]],
        model=ModelSpec(
            name=model_payload.get("name", "llama2_7b_attention"),
            hidden_size=int(model_payload["hidden_size"]),
            num_heads=int(model_payload["num_heads"]),
        ),
        cases=cases,
    )


def apply_cli_overrides(spec: ExperimentSpec, args) -> ExperimentSpec:
    if args.prefill_lengths:
        spec.prefill_lengths = parse_int_list(args.prefill_lengths)
    if args.batch_size is not None:
        spec.batch_size = int(args.batch_size)
    if args.ignore_hbm_bottleneck:
        spec.ignore_hbm_bottleneck = True
    if args.ignore_onchip_io_bottleneck:
        spec.ignore_onchip_io_bottleneck = True
    if not spec.prefill_lengths:
        raise ValueError("At least one prefill length must be provided.")
    return spec


def load_single_card_system(
    config_path: str,
    adjust_global_buffer_to_memory_capacity: bool = False,
    ignore_onchip_io_bottleneck: bool = False,
):
    specs = read_architecture_template(config_path)
    specs["device_count"] = 1
    specs["interconnect"]["link_count_per_device"] = 0
    specs["name"] = f"{specs['name']} single-card"
    system = template_to_system(specs)
    if adjust_global_buffer_to_memory_capacity:
        system.device.compute_module.l2_size = int(
            system.device.memory_module.memory_capacity
        )
    if ignore_onchip_io_bottleneck:
        system.device.compute_module.l2_bandwidth_per_cycle = float("inf")
    return system, specs


def maybe_int(value):
    if value in (None, "", 0):
        return value
    return int(value)


def maybe_float_ms(value) -> float:
    if value in (None, ""):
        return 0.0
    return float(value) * 1e3


def estimate_hbm_bytes(attention, mapping, configured_hbm_bandwidth: float) -> int:
    direct_bytes = int(getattr(attention, "hbm_bytes", 0) or 0)
    if direct_bytes > 0:
        return direct_bytes

    q_tile_size = getattr(mapping, "q_tile_size", None) if mapping is not None else None
    fused_hbm_bytes = getattr(attention, "_fused_hbm_bytes", None)
    if callable(fused_hbm_bytes) and q_tile_size:
        return int(fused_hbm_bytes(q_tile_size))

    active_hbm_latency = float(getattr(attention, "hbm_io_latency", 0.0) or 0.0)
    if (
        active_hbm_latency > 0.0
        and configured_hbm_bandwidth > 0.0
        and math.isfinite(configured_hbm_bandwidth)
    ):
        return int(round(active_hbm_latency * configured_hbm_bandwidth))
    return 0


def non_hbm_critical_ms(variant_name: str, components: Dict[str, float]) -> float:
    if variant_name == "flashattention":
        return max(components["compute_model_ms"], components["onchip_io_ms"])
    if variant_name == "customsa":
        return max(components["fused_core_ms"], components["onchip_io_ms"])
    return 0.0


def global_buffer_mb(raw_specs: Dict[str, object]) -> float:
    io_specs = raw_specs["device"]["io"]
    if "global_buffer_MB" in io_specs:
        return float(io_specs["global_buffer_MB"])
    if "physical_global_buffer_MB" in io_specs:
        return float(io_specs["physical_global_buffer_MB"])
    if "global_buffer_byte" in io_specs:
        return float(io_specs["global_buffer_byte"]) / (1024 * 1024)
    return 0.0


def classify_bottleneck(variant_name: str, components: Dict[str, float]) -> Tuple[str, float]:
    if variant_name == "baseline":
        candidates = {
            "q_mul_k": components["q_mul_k_ms"],
            "softmax": components["softmax_ms"],
            "a_mul_v": components["a_mul_v_ms"],
        }
    elif variant_name == "flashattention":
        candidates = {
            "matmul_pair": components["q_mul_k_ms"] + components["a_mul_v_ms"],
            "softmax": components["softmax_ms"],
            "hbm_io": components["hbm_io_ms"],
            "onchip_io": components["onchip_io_ms"],
        }
    elif variant_name == "customsa":
        candidates = {
            "fused_core": components["fused_core_ms"],
            "hbm_io": components["hbm_io_ms"],
            "onchip_io": components["onchip_io_ms"],
        }
    else:
        raise ValueError(f"Unsupported variant for bottleneck classification: {variant_name}")

    bottleneck_name, bottleneck_ms = max(candidates.items(), key=lambda item: item[1])
    return bottleneck_name, float(bottleneck_ms)


def profile_attention_variant(
    spec: ExperimentSpec,
    case: CaseSpec,
    system,
    raw_specs,
    configured_hbm_bandwidth: float,
    active_hbm_bandwidth: float,
    configured_onchip_bandwidth: float,
    active_onchip_bandwidth: float,
    prefill_length: int,
    variant_spec: Dict[str, str],
) -> Dict[str, object]:
    attention = build_attention(variant_spec["attention_variant"], case.data_type)
    head_dim = spec.model.head_dim()
    query = Tensor(
        [spec.batch_size, spec.model.num_heads, prefill_length, head_dim],
        case.data_type,
    )
    key = Tensor(
        [spec.batch_size, spec.model.num_heads, head_dim, prefill_length],
        case.data_type,
    )
    value = Tensor(
        [spec.batch_size, spec.model.num_heads, prefill_length, head_dim],
        case.data_type,
    )
    attention(query, key, value)
    total_latency = attention.compile_and_simulate(system.device, spec.compile_mode)
    mapping = getattr(attention, "best_mapping", None)
    tensor_tflops = system.device.compute_module.total_systolic_array_flops / 1e12
    vector_tflops = system.device.compute_module.total_vector_flops / 1e12

    component_values = {
        "q_mul_k_ms": float(getattr(attention, "q_mul_k_latency", 0.0)) * 1e3,
        "softmax_ms": float(getattr(attention, "softmax_latency", 0.0)) * 1e3,
        "a_mul_v_ms": float(getattr(attention, "a_mul_v_latency", 0.0)) * 1e3,
        "compute_model_ms": maybe_float_ms(
            getattr(attention, "compute_latency", 0.0)
        ),
        "fused_core_ms": maybe_float_ms(
            getattr(attention, "fused_core_latency", 0.0)
        ),
        "hbm_io_ms": maybe_float_ms(getattr(attention, "hbm_io_latency", 0.0)),
        "onchip_io_ms": maybe_float_ms(
            getattr(attention, "onchip_io_latency", 0.0)
        ),
    }
    bottleneck_component, bottleneck_ms = classify_bottleneck(
        variant_spec["name"],
        component_values,
    )
    hbm_bytes = estimate_hbm_bytes(attention, mapping, configured_hbm_bandwidth)
    configured_hbm_io_ms = (
        (hbm_bytes / configured_hbm_bandwidth) * 1e3
        if hbm_bytes > 0 and configured_hbm_bandwidth > 0.0 and math.isfinite(configured_hbm_bandwidth)
        else 0.0
    )
    non_hbm_ms = non_hbm_critical_ms(variant_spec["name"], component_values)
    required_hbm_bandwidth_tbps = (
        (hbm_bytes / (non_hbm_ms / 1e3)) / 1e12
        if hbm_bytes > 0 and non_hbm_ms > 0.0
        else None
    )
    required_hbm_bandwidth_over_config_x = (
        required_hbm_bandwidth_tbps / (configured_hbm_bandwidth / 1e12)
        if required_hbm_bandwidth_tbps is not None
        and configured_hbm_bandwidth > 0.0
        and math.isfinite(configured_hbm_bandwidth)
        else None
    )

    return {
        "case_name": case.name,
        "case_label": case.label,
        "system_config_path": case.system_config_path,
        "data_type": case.data_type_name,
        "prefill_length": prefill_length,
        "batch_size": spec.batch_size,
        "hidden_size": spec.model.hidden_size,
        "num_heads": spec.model.num_heads,
        "head_dim": head_dim,
        "compile_mode": spec.compile_mode,
        "variant": variant_spec["name"],
        "variant_label": variant_spec["label"],
        "attention_variant": variant_spec["attention_variant"],
        "total_latency_ms": total_latency * 1e3,
        "q_mul_k_ms": component_values["q_mul_k_ms"],
        "softmax_ms": component_values["softmax_ms"],
        "a_mul_v_ms": component_values["a_mul_v_ms"],
        "compute_model_ms": component_values["compute_model_ms"],
        "fused_core_ms": component_values["fused_core_ms"],
        "hbm_io_ms": component_values["hbm_io_ms"],
        "configured_hbm_io_ms": configured_hbm_io_ms,
        "onchip_io_ms": component_values["onchip_io_ms"],
        "kernel_overhead_ms": maybe_float_ms(
            getattr(attention, "kernel_overhead", 0.0)
        ),
        "non_hbm_critical_ms": non_hbm_ms,
        "bottleneck_component": bottleneck_component,
        "bottleneck_ms": bottleneck_ms,
        "bottleneck_share_of_total": bottleneck_ms / (total_latency * 1e3)
        if total_latency
        else 0.0,
        "software_q_tile_size": maybe_int(getattr(mapping, "q_tile_size", None)),
        "software_kv_tile_size": maybe_int(getattr(mapping, "kv_tile_size", None)),
        "logical_q_tile_size": maybe_int(
            getattr(mapping, "logical_q_tile_size", None)
        ),
        "tensor_tflops": tensor_tflops,
        "vector_tflops": vector_tflops,
        "tensor_vector_ratio": tensor_tflops / vector_tflops if vector_tflops else 0.0,
        "hbm_bytes": hbm_bytes,
        "hbm_mib": hbm_bytes / (1024.0 * 1024.0),
        "configured_hbm_bandwidth_tbps": configured_hbm_bandwidth / 1e12,
        "active_hbm_bandwidth_tbps": active_hbm_bandwidth / 1e12,
        "configured_onchip_bandwidth_tbps": configured_onchip_bandwidth / 1e12,
        "active_onchip_bandwidth_tbps": active_onchip_bandwidth / 1e12,
        "required_hbm_bandwidth_tbps": required_hbm_bandwidth_tbps,
        "required_hbm_bandwidth_over_config_x": required_hbm_bandwidth_over_config_x,
        "ignore_hbm_bottleneck": int(spec.ignore_hbm_bottleneck),
        "ignore_onchip_io_bottleneck": int(spec.ignore_onchip_io_bottleneck),
        "sram_kib": raw_specs["device"]["compute_chiplet"]["core"]["SRAM_KB"],
        "l2_mb": global_buffer_mb(raw_specs),
    }


def simulate_case(
    case: CaseSpec,
    compile_mode: str,
    batch_size: int,
    adjust_global_buffer_to_memory_capacity: bool,
    ignore_hbm_bottleneck: bool,
    ignore_onchip_io_bottleneck: bool,
    prefill_lengths: List[int],
    hidden_size: int,
    num_heads: int,
):
    reset_cache_stats()
    system, raw_specs = load_single_card_system(
        case.system_config_path,
        adjust_global_buffer_to_memory_capacity=adjust_global_buffer_to_memory_capacity,
        ignore_onchip_io_bottleneck=ignore_onchip_io_bottleneck,
    )
    configured_hbm_bandwidth = float(system.device.io_module.bandwidth)
    configured_onchip_bandwidth = float(
        system.device.compute_module.l2_bandwidth_per_cycle
        * system.device.compute_module.clock_freq
    )
    if ignore_hbm_bottleneck:
        system.device.io_module.bandwidth = float("inf")
    active_hbm_bandwidth = float(system.device.io_module.bandwidth)
    active_onchip_bandwidth = float(
        system.device.compute_module.l2_bandwidth_per_cycle
        * system.device.compute_module.clock_freq
    )
    spec = ExperimentSpec(
        experiment_name="worker_local",
        experiment_config_path="",
        compile_mode=compile_mode,
        batch_size=batch_size,
        adjust_global_buffer_to_memory_capacity=adjust_global_buffer_to_memory_capacity,
        ignore_hbm_bottleneck=ignore_hbm_bottleneck,
        ignore_onchip_io_bottleneck=ignore_onchip_io_bottleneck,
        prefill_lengths=prefill_lengths,
        model=ModelSpec(
            name="llama2_7b_attention",
            hidden_size=hidden_size,
            num_heads=num_heads,
        ),
        cases=[case],
    )
    rows = []
    for prefill_length in prefill_lengths:
        for variant_spec in VARIANTS:
            row = profile_attention_variant(
                spec,
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
            rows.append(row)
            print(
                f"case={case.name:<10} dtype={case.data_type_name:<4} seq={prefill_length:>5d} "
                f"variant={variant_spec['name']:<14} total_ms={row['total_latency_ms']:>10.3f} "
                f"bottleneck={row['bottleneck_component']:<11} component_ms={row['bottleneck_ms']:>10.3f} "
                f"req_hbm_tbps={row['required_hbm_bandwidth_tbps'] if row['required_hbm_bandwidth_tbps'] is not None else 0.0:>8.3f}",
                flush=True,
            )
    return {
        "case_name": case.name,
        "rows": rows,
        "raw_specs": raw_specs,
        "cache_stats": collect_cache_stats(),
    }


def build_speedup_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = {}
    for row in rows:
        key = (row["case_name"], int(row["prefill_length"]))
        grouped.setdefault(key, {})[str(row["variant"])] = row

    speedup_rows = []
    for (case_name, prefill_length), variants in sorted(grouped.items()):
        baseline = variants["baseline"]
        flash = variants["flashattention"]
        customsa = variants["customsa"]
        for candidate in [flash, customsa]:
            speedup_rows.append(
                {
                    "case_name": case_name,
                    "case_label": baseline["case_label"],
                    "data_type": baseline["data_type"],
                    "prefill_length": prefill_length,
                    "variant": candidate["variant"],
                    "variant_label": candidate["variant_label"],
                    "baseline_total_ms": float(baseline["total_latency_ms"]),
                    "variant_total_ms": float(candidate["total_latency_ms"]),
                    "speedup_vs_baseline_x": float(baseline["total_latency_ms"])
                    / float(candidate["total_latency_ms"]),
                    "customsa_vs_flash_x": float(flash["total_latency_ms"])
                    / float(customsa["total_latency_ms"]),
                    "tensor_vector_ratio": float(baseline["tensor_vector_ratio"]),
                }
            )
    return speedup_rows


def write_csv(path: str, rows) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: str, rows, speedup_rows) -> None:
    longest_context = max(int(row["prefill_length"]) for row in rows)
    grouped = {}
    for row in rows:
        if int(row["prefill_length"]) != longest_context:
            continue
        grouped[row["case_name"]] = grouped.get(row["case_name"], {})
        grouped[row["case_name"]][row["variant"]] = row
    speedup_index = {
        (row["case_name"], row["variant"], int(row["prefill_length"])): row
        for row in speedup_rows
    }

    lines = [
        "# Attention Summary",
        "",
        f"Longest context: `{longest_context}`",
        "",
        "| case | dtype | ratio | baseline ms | flash ms | customsa ms | flash bottleneck | customsa bottleneck | flash/customsa |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for case_name in sorted(grouped):
        baseline = grouped[case_name]["baseline"]
        flash = grouped[case_name]["flashattention"]
        customsa = grouped[case_name]["customsa"]
        flash_speed = speedup_index[(case_name, "flashattention", longest_context)]
        lines.append(
            f"| {baseline['case_label']} | {baseline['data_type']} | "
            f"{baseline['tensor_vector_ratio']:.2f} | "
            f"{baseline['total_latency_ms']:.3f} | {flash['total_latency_ms']:.3f} | "
            f"{customsa['total_latency_ms']:.3f} | "
            f"{flash['bottleneck_component']} ({flash['bottleneck_ms']:.3f} ms) | "
            f"{customsa['bottleneck_component']} ({customsa['bottleneck_ms']:.3f} ms) | "
            f"{flash_speed['customsa_vs_flash_x']:.3f}x |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="Path to the experiment JSON config.",
    )
    parser.add_argument(
        "--prefill-lengths",
        help="Comma-separated override for prefill lengths.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the batch size.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where raw outputs should be written.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="Worker count for case-level parallel execution. Defaults to the number of cases.",
    )
    parser.add_argument(
        "--ignore-hbm-bottleneck",
        action="store_true",
        help="Run the simulation with effectively infinite HBM bandwidth while still reporting the configured and required HBM bandwidth.",
    )
    parser.add_argument(
        "--ignore-onchip-io-bottleneck",
        action="store_true",
        help="Run the simulation with effectively infinite on-chip/global-buffer bandwidth while still reporting the configured on-chip bandwidth.",
    )
    args = parser.parse_args()

    ensure_scalesim_temp_dir()
    reset_cache_stats()
    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    config_path = os.path.abspath(args.experiment_config)
    spec = apply_cli_overrides(load_experiment_spec(config_path), args)

    rows = []
    case_specs = {}
    per_case_cache_stats = {}
    worker_count = args.jobs if args.jobs is not None else min(len(spec.cases), os.cpu_count() or 1)
    worker_count = max(1, min(worker_count, len(spec.cases)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                simulate_case,
                case,
                spec.compile_mode,
                spec.batch_size,
                spec.adjust_global_buffer_to_memory_capacity,
                spec.ignore_hbm_bottleneck,
                spec.ignore_onchip_io_bottleneck,
                spec.prefill_lengths,
                spec.model.hidden_size,
                spec.model.num_heads,
            ): case
            for case in spec.cases
        }
        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            result = future.result()
            case_specs[case.name] = result["raw_specs"]
            per_case_cache_stats[case.name] = result["cache_stats"]
            rows.extend(result["rows"])

    rows.sort(
        key=lambda row: (str(row["case_name"]), int(row["prefill_length"]), str(row["variant"]))
    )
    speedup_rows = build_speedup_rows(rows)

    latency_csv_path = os.path.join(output_dir, "attention_latency.csv")
    speedup_csv_path = os.path.join(output_dir, "case_speedups.csv")
    summary_path = os.path.join(output_dir, "summary.md")
    metadata_path = os.path.join(output_dir, "metadata.json")
    cache_stats_path = os.path.join(output_dir, "cache_stats.json")

    write_csv(latency_csv_path, rows)
    write_csv(speedup_csv_path, speedup_rows)
    write_summary(summary_path, rows, speedup_rows)
    cache_stats = {
        "aggregated": aggregate_cache_stats(list(per_case_cache_stats.values())),
        "per_case": per_case_cache_stats,
    }
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "experiment_name": spec.experiment_name,
                "experiment_config_path": spec.experiment_config_path,
                "compile_mode": spec.compile_mode,
                "batch_size": spec.batch_size,
                "jobs": worker_count,
                "adjust_global_buffer_to_memory_capacity": spec.adjust_global_buffer_to_memory_capacity,
                "ignore_hbm_bottleneck": spec.ignore_hbm_bottleneck,
                "ignore_onchip_io_bottleneck": spec.ignore_onchip_io_bottleneck,
                "prefill_lengths": spec.prefill_lengths,
                "variants": VARIANTS,
                "attention_env": {
                    "LLMCOMPASS_INT8_MATCH_FP16_TILES": os.environ.get(
                        "LLMCOMPASS_INT8_MATCH_FP16_TILES",
                        "0",
                    ),
                    "LLMCOMPASS_INT8_SOFTMAX_CONVERSION": os.environ.get(
                        "LLMCOMPASS_INT8_SOFTMAX_CONVERSION",
                        "0",
                    ),
                    "LLMCOMPASS_CUSTOMSA_SEARCH_TILES": os.environ.get(
                        "LLMCOMPASS_CUSTOMSA_SEARCH_TILES",
                        "0",
                    ),
                    "LLMCOMPASS_CUSTOMSA_STAGE_OVERHEAD_CYCLES": os.environ.get(
                        "LLMCOMPASS_CUSTOMSA_STAGE_OVERHEAD_CYCLES",
                        str(FlashAttentionCustomSA._DEFAULT_STAGE_OVERHEAD_CYCLES),
                    ),
                },
                "model": {
                    "name": spec.model.name,
                    "hidden_size": spec.model.hidden_size,
                    "num_heads": spec.model.num_heads,
                    "head_dim": spec.model.head_dim(),
                },
                "cases": [
                    {
                        "name": case.name,
                        "label": case.label,
                        "system_config_path": case.system_config_path,
                        "data_type": case.data_type_name,
                        "resolved_specs": case_specs[case.name],
                    }
                    for case in spec.cases
                ],
                "outputs": {
                    "output_dir": output_dir,
                    "attention_latency_csv": latency_csv_path,
                    "case_speedups_csv": speedup_csv_path,
                    "summary_md": summary_path,
                    "cache_stats_json": cache_stats_path,
                },
                "cache_stats": cache_stats,
            },
            f,
            indent=2,
        )
    with open(cache_stats_path, "w") as f:
        json.dump(cache_stats, f, indent=2)

    print("cache_stats:", json.dumps(cache_stats, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
