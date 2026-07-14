import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))
LLMCOMPASS_ROOT = os.path.join(REPO_ROOT, "LLMCompass")

if LLMCOMPASS_ROOT not in sys.path:
    sys.path.insert(0, LLMCOMPASS_ROOT)

from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.attention import (
    FlashAttention,
    FlashAttentionCustomSA,
    FlashAttentionILLM,
    FlashAttentionIntAttention,
    TrivialAttention,
)
from software_model.utils import Tensor, data_type_dict


DEFAULT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

DEFAULT_VARIANTS = [
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

INT_SOFTMAX_VARIANTS = [
    {
        "name": "baseline",
        "label": "FlashAttention",
        "attention_variant": "flashattention",
    },
    {
        "name": "customsa",
        "label": "FlashAttention-CustomSA",
        "attention_variant": "flashattention_customsa",
    },
    {
        "name": "illm",
        "label": "FlashAttention + I-LLM",
        "attention_variant": "flashattention_illm",
    },
    {
        "name": "intattention",
        "label": "FlashAttention + IntAttention",
        "attention_variant": "flashattention_intattention",
    },
]

FLASH_VARIANT_CLASSES = {
    "flashattention": FlashAttention,
    "flashattention_illm": FlashAttentionILLM,
    "flashattention_illm_di": FlashAttentionILLM,
    "flashattention_intattention": FlashAttentionIntAttention,
}


@dataclass(frozen=True)
class DeviceSpec:
    key: str
    case_name: str
    case_label: str
    system_config: str
    data_type_name: str
    attention_dir: str
    q_tile_size: int
    kv_tile_size: int
    adjust_global_buffer_to_memory_capacity: bool
    ignore_hbm_bottleneck: bool
    ignore_onchip_io_bottleneck: bool


DEVICES = {
    "b200": DeviceSpec(
        key="b200",
        case_name="b200_fp16",
        case_label="B200 FP16",
        system_config="LLMCompass/configs/ours/B200.json",
        data_type_name="fp16",
        attention_dir="",
        q_tile_size=32,
        kv_tile_size=256,
        adjust_global_buffer_to_memory_capacity=False,
        ignore_hbm_bottleneck=True,
        ignore_onchip_io_bottleneck=True,
    ),
    "b300": DeviceSpec(
        key="b300",
        case_name="b300_fp16",
        case_label="B300 FP16",
        system_config="LLMCompass/configs/ours/B300.json",
        data_type_name="fp16",
        attention_dir="",
        q_tile_size=32,
        kv_tile_size=256,
        adjust_global_buffer_to_memory_capacity=False,
        ignore_hbm_bottleneck=True,
        ignore_onchip_io_bottleneck=True,
    ),
    "awsv4": DeviceSpec(
        key="awsv4",
        case_name="awsv4_fp16",
        case_label="AWSv4 FP16",
        system_config="LLMCompass/configs/ours/AWSv4.json",
        data_type_name="fp16",
        attention_dir="",
        q_tile_size=512,
        kv_tile_size=4096,
        adjust_global_buffer_to_memory_capacity=True,
        ignore_hbm_bottleneck=True,
        ignore_onchip_io_bottleneck=False,
    ),
    "tpuv6e": DeviceSpec(
        key="tpuv6e",
        case_name="tpuv6e_fp16",
        case_label="TPUv6e FP16",
        system_config="LLMCompass/configs/ours/TPUv6e.json",
        data_type_name="fp16",
        attention_dir="",
        q_tile_size=512,
        kv_tile_size=4096,
        adjust_global_buffer_to_memory_capacity=True,
        ignore_hbm_bottleneck=True,
        ignore_onchip_io_bottleneck=False,
    ),
    "b200_int8": DeviceSpec(
        key="b200_int8",
        case_name="b200_int8",
        case_label="B200 INT8",
        system_config="LLMCompass/configs/ours/B200.json",
        data_type_name="int8",
        attention_dir="",
        q_tile_size=32,
        kv_tile_size=512,
        adjust_global_buffer_to_memory_capacity=False,
        ignore_hbm_bottleneck=True,
        ignore_onchip_io_bottleneck=True,
    ),
    "b300_int8": DeviceSpec(
        key="b300_int8",
        case_name="b300_int8",
        case_label="B300 INT8",
        system_config="LLMCompass/configs/ours/B300.json",
        data_type_name="int8",
        attention_dir="",
        q_tile_size=32,
        kv_tile_size=512,
        adjust_global_buffer_to_memory_capacity=False,
        ignore_hbm_bottleneck=True,
        ignore_onchip_io_bottleneck=True,
    ),
    "h100_int8": DeviceSpec(
        key="h100_int8",
        case_name="h100_int8",
        case_label="H100 INT8",
        system_config="LLMCompass/configs/ours/H100.json",
        data_type_name="int8",
        attention_dir="",
        q_tile_size=128,
        kv_tile_size=512,
        adjust_global_buffer_to_memory_capacity=False,
        ignore_hbm_bottleneck=True,
        ignore_onchip_io_bottleneck=True,
    ),
    "awsv4_int8": DeviceSpec(
        key="awsv4_int8",
        case_name="awsv4_int8",
        case_label="AWSv4 INT8",
        system_config="LLMCompass/configs/ours/AWSv4.json",
        data_type_name="int8",
        attention_dir="",
        q_tile_size=128,
        kv_tile_size=16384,
        adjust_global_buffer_to_memory_capacity=True,
        ignore_hbm_bottleneck=True,
        ignore_onchip_io_bottleneck=False,
    ),
    "tpuv6e_int8": DeviceSpec(
        key="tpuv6e_int8",
        case_name="tpuv6e_int8",
        case_label="TPUv6e INT8",
        system_config="LLMCompass/configs/ours/TPUv6e.json",
        data_type_name="int8",
        attention_dir="",
        q_tile_size=512,
        kv_tile_size=8192,
        adjust_global_buffer_to_memory_capacity=True,
        ignore_hbm_bottleneck=True,
        ignore_onchip_io_bottleneck=False,
    ),
}


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def load_single_card_system(spec: DeviceSpec):
    raw_specs = read_architecture_template(resolve_path(spec.system_config))
    raw_specs["device_count"] = 1
    raw_specs["interconnect"]["link_count_per_device"] = 0
    raw_specs["name"] = f"{raw_specs['name']} single-card"
    system = template_to_system(raw_specs)
    if spec.adjust_global_buffer_to_memory_capacity:
        system.device.compute_module.l2_size = int(
            system.device.memory_module.memory_capacity
        )
    if spec.ignore_hbm_bottleneck:
        system.device.io_module.bandwidth = float("inf")
    if spec.ignore_onchip_io_bottleneck:
        system.device.compute_module.l2_bandwidth_per_cycle = float("inf")
    return system, raw_specs


def read_source_rows(
    spec: DeviceSpec,
    source_dir_override: Optional[str] = None,
) -> Dict[Tuple[int, str], Dict[str, str]]:
    source_dir = source_dir_override or spec.attention_dir
    path = os.path.join(resolve_path(source_dir), "attention_latency.csv")
    rows: Dict[Tuple[int, str], Dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["case_name"] != spec.case_name:
                continue
            rows[(int(row["prefill_length"]), row["variant"])] = row
    return rows


def build_attention_inputs(length: int, dtype):
    return (
        Tensor([1, 32, length, 128], dtype),
        Tensor([1, 32, 128, length], dtype),
        Tensor([1, 32, length, 128], dtype),
    )


def evaluate_flash(
    length: int,
    device,
    dtype,
    q_tile_size: int,
    kv_tile_size: int,
    is_double_buffering: bool,
    attention_cls=FlashAttention,
):
    attention = attention_cls(dtype)
    attention(*build_attention_inputs(length, dtype))
    (
        total,
        mapping,
        q_mul_k,
        softmax,
        a_mul_v,
        compute,
        hbm,
        onchip,
        kernel_overhead,
    ) = attention._estimate_mapping_latency(
        q_tile_size,
        kv_tile_size,
        device,
        is_double_buffering,
    )
    scale_fn = getattr(attention, "_softmax_scale_for_sequence", None)
    return {
        "total_latency_ms": total * 1e3,
        "q_mul_k_ms": q_mul_k * 1e3,
        "softmax_ms": softmax * 1e3,
        "softmax_scale": float(scale_fn(device)) if scale_fn else 1.0,
        "a_mul_v_ms": a_mul_v * 1e3,
        "compute_model_ms": compute * 1e3,
        "fused_core_ms": 0.0,
        "hbm_io_ms": hbm * 1e3,
        "onchip_io_ms": onchip * 1e3,
        "kernel_overhead_ms": kernel_overhead * 1e3,
        "workspace_bytes": mapping.workspace_bytes,
    }


def evaluate_customsa(
    length: int,
    device,
    dtype,
    q_tile_size: int,
    kv_tile_size: int,
):
    attention = FlashAttentionCustomSA(dtype)
    attention(*build_attention_inputs(length, dtype))
    (
        total,
        mapping,
        fused_core_cycles,
        fused_core,
        hbm,
        onchip,
        hbm_bytes,
        kernel_overhead,
        steady_cycles,
        drain_cycles,
    ) = attention._estimate_mapping_latency(q_tile_size, kv_tile_size, device)
    return {
        "total_latency_ms": total * 1e3,
        "q_mul_k_ms": 0.0,
        "softmax_ms": 0.0,
        "softmax_scale": 1.0,
        "a_mul_v_ms": 0.0,
        "compute_model_ms": 0.0,
        "fused_core_ms": fused_core * 1e3,
        "hbm_io_ms": hbm * 1e3,
        "onchip_io_ms": onchip * 1e3,
        "kernel_overhead_ms": kernel_overhead * 1e3,
        "workspace_bytes": mapping.workspace_bytes,
        "hbm_bytes": hbm_bytes,
        "fused_core_cycles": fused_core_cycles,
        "steady_cycles_per_tile": steady_cycles,
        "drain_cycles_per_tile": drain_cycles,
    }


def evaluate_baseline(length: int, device, dtype):
    attention = TrivialAttention(dtype)
    attention(*build_attention_inputs(length, dtype))
    total = attention.roofline_model(device)
    return {
        "total_latency_ms": total * 1e3,
        "q_mul_k_ms": attention.q_mul_k_latency * 1e3,
        "softmax_ms": attention.softmax_latency * 1e3,
        "softmax_scale": 1.0,
        "a_mul_v_ms": attention.a_mul_v_latency * 1e3,
        "compute_model_ms": 0.0,
        "fused_core_ms": 0.0,
        "hbm_io_ms": 0.0,
        "onchip_io_ms": 0.0,
        "kernel_overhead_ms": 0.0,
        "workspace_bytes": 0,
    }


def source_component(row: Dict[str, str], name: str) -> float:
    value = row.get(name, "")
    if value == "" or value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def row_from_source(spec: DeviceSpec, row: Dict[str, str], source_tag: str) -> Dict[str, object]:
    return {
        "case_name": spec.case_name,
        "case_label": spec.case_label,
        "system_config_path": resolve_path(spec.system_config),
        "data_type": spec.data_type_name,
        "prefill_length": int(row["prefill_length"]),
        "batch_size": 1,
        "hidden_size": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "compile_mode": "fixed-tile",
        "variant": row["variant"],
        "variant_label": row["variant_label"],
        "attention_variant": row["attention_variant"],
        "total_latency_ms": float(row["total_latency_ms"]),
        "q_mul_k_ms": source_component(row, "q_mul_k_ms"),
        "softmax_ms": source_component(row, "softmax_ms"),
        "softmax_scale": source_component(row, "softmax_scale") or 1.0,
        "a_mul_v_ms": source_component(row, "a_mul_v_ms"),
        "compute_model_ms": source_component(row, "compute_model_ms"),
        "fused_core_ms": source_component(row, "fused_core_ms"),
        "hbm_io_ms": source_component(row, "hbm_io_ms"),
        "onchip_io_ms": source_component(row, "onchip_io_ms"),
        "kernel_overhead_ms": source_component(row, "kernel_overhead_ms"),
        "software_q_tile_size": row.get("software_q_tile_size", ""),
        "software_kv_tile_size": row.get("software_kv_tile_size", ""),
        "logical_q_tile_size": row.get("logical_q_tile_size", ""),
        "tensor_tflops": source_component(row, "tensor_tflops"),
        "vector_tflops": source_component(row, "vector_tflops"),
        "tensor_vector_ratio": source_component(row, "tensor_vector_ratio"),
        "source": source_tag,
    }


def classify_bottleneck(variant: str, components: Dict[str, float]) -> Tuple[str, float]:
    if variant == "baseline":
        if components.get("fused_core_ms", 0.0):
            candidates = {
                "fused_core": components["fused_core_ms"],
                "hbm_io": components["hbm_io_ms"],
                "onchip_io": components["onchip_io_ms"],
            }
        elif components.get("compute_model_ms", 0.0):
            candidates = {
                "matmul_pair": components["q_mul_k_ms"] + components["a_mul_v_ms"],
                "softmax": components["softmax_ms"],
                "hbm_io": components["hbm_io_ms"],
                "onchip_io": components["onchip_io_ms"],
            }
        else:
            candidates = {
                "q_mul_k": components["q_mul_k_ms"],
                "softmax": components["softmax_ms"],
                "a_mul_v": components["a_mul_v_ms"],
            }
    elif variant in {"flashattention", "illm", "intattention"}:
        candidates = {
            "matmul_pair": components["q_mul_k_ms"] + components["a_mul_v_ms"],
            "softmax": components["softmax_ms"],
            "hbm_io": components["hbm_io_ms"],
            "onchip_io": components["onchip_io_ms"],
        }
    else:
        candidates = {
            "fused_core": components["fused_core_ms"],
            "hbm_io": components["hbm_io_ms"],
            "onchip_io": components["onchip_io_ms"],
        }
    return max(candidates.items(), key=lambda item: item[1])


def build_modeled_row(
    spec: DeviceSpec,
    length: int,
    variant_spec: Dict[str, str],
    components: Dict[str, float],
    system,
) -> Dict[str, object]:
    variant = variant_spec["name"]
    bottleneck, bottleneck_ms = classify_bottleneck(variant, components)
    total = float(components["total_latency_ms"])
    logical_q_tile_size = spec.q_tile_size if variant == "customsa" else ""
    dtype = data_type_dict[spec.data_type_name]
    return {
        "case_name": spec.case_name,
        "case_label": spec.case_label,
        "system_config_path": resolve_path(spec.system_config),
        "data_type": spec.data_type_name,
        "prefill_length": length,
        "batch_size": 1,
        "hidden_size": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "compile_mode": "fixed-tile",
        "variant": variant,
        "variant_label": variant_spec["label"],
        "attention_variant": variant_spec["attention_variant"],
        "total_latency_ms": total,
        "q_mul_k_ms": components["q_mul_k_ms"],
        "softmax_ms": components["softmax_ms"],
        "softmax_scale": components["softmax_scale"],
        "a_mul_v_ms": components["a_mul_v_ms"],
        "compute_model_ms": components["compute_model_ms"],
        "fused_core_ms": components["fused_core_ms"],
        "hbm_io_ms": components["hbm_io_ms"],
        "onchip_io_ms": components["onchip_io_ms"],
        "kernel_overhead_ms": components["kernel_overhead_ms"],
        "bottleneck_component": bottleneck,
        "bottleneck_ms": bottleneck_ms,
        "bottleneck_share_of_total": bottleneck_ms / total if total else 0.0,
        "software_q_tile_size": spec.q_tile_size if variant != "baseline" else "",
        "software_kv_tile_size": spec.kv_tile_size if variant != "baseline" else "",
        "logical_q_tile_size": logical_q_tile_size,
        "tensor_tflops": system.device.compute_module.total_systolic_array_flops_for(
            dtype
        )
        / 1e12,
        "vector_tflops": system.device.compute_module.total_vector_flops / 1e12,
        "tensor_vector_ratio": (
            system.device.compute_module.total_systolic_array_flops_for(dtype)
            / system.device.compute_module.total_vector_flops
        ),
        "source": "fixed_tile_model",
    }


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_speedup_rows(
    rows: List[Dict[str, object]],
    variants_to_compare: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    grouped: Dict[int, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["prefill_length"]), {})[str(row["variant"])] = row
    baseline_name = variants_to_compare[0]["name"]
    speedups = []
    for length, variants in sorted(grouped.items()):
        baseline = variants[baseline_name]
        for variant_spec in variants_to_compare:
            candidate = variants[variant_spec["name"]]
            speedups.append(
                {
                    "case_name": candidate["case_name"],
                    "case_label": candidate["case_label"],
                    "data_type": candidate["data_type"],
                    "prefill_length": length,
                    "baseline_variant": baseline_name,
                    "baseline_label": baseline["variant_label"],
                    "variant": candidate["variant"],
                    "variant_label": candidate["variant_label"],
                    "baseline_total_ms": baseline["total_latency_ms"],
                    "variant_total_ms": candidate["total_latency_ms"],
                    "speedup_vs_baseline_x": (
                        float(baseline["total_latency_ms"]) / float(candidate["total_latency_ms"])
                    ),
                    "tensor_vector_ratio": candidate["tensor_vector_ratio"],
                    "source": candidate["source"],
                }
            )
    return speedups


def choose_double_buffering(
    spec: DeviceSpec,
    source_rows: Dict[Tuple[int, str], Dict[str, str]],
    system,
    dtype,
) -> bool:
    reference = source_rows.get((32768, "flashattention"))
    if reference is None:
        return True
    target = float(reference["total_latency_ms"])
    candidates = []
    for flag in [True, False]:
        modeled = evaluate_flash(
            32768,
            system.device,
            dtype,
            spec.q_tile_size,
            spec.kv_tile_size,
            flag,
        )["total_latency_ms"]
        candidates.append((abs(modeled - target), flag, modeled))
    return min(candidates, key=lambda item: item[0])[1]


def run_device(
    spec: DeviceSpec,
    lengths: List[int],
    output_root: str,
    variants_to_compare: List[Dict[str, str]],
    use_source_rows: bool,
    source_dir_override: Optional[str] = None,
) -> str:
    output_dir = os.path.join(output_root, spec.key)
    os.makedirs(output_dir, exist_ok=True)
    system, raw_specs = load_single_card_system(spec)
    dtype = data_type_dict[spec.data_type_name]
    source_rows = (
        read_source_rows(spec, source_dir_override)
        if use_source_rows and source_dir_override
        else {}
    )
    is_double_buffering = choose_double_buffering(
        spec,
        source_rows,
        system,
        dtype,
    )

    rows = []
    for length in lengths:
        for variant_spec in variants_to_compare:
            variant = variant_spec["name"]
            attention_variant = variant_spec["attention_variant"]
            source = source_rows.get((length, variant)) if use_source_rows else None
            if source is not None:
                rows.append(row_from_source(spec, source, "source_csv"))
                continue
            if attention_variant == "trivial":
                components = evaluate_baseline(length, system.device, dtype)
            elif attention_variant in FLASH_VARIANT_CLASSES:
                components = evaluate_flash(
                    length,
                    system.device,
                    dtype,
                    spec.q_tile_size,
                    spec.kv_tile_size,
                    is_double_buffering,
                    attention_cls=FLASH_VARIANT_CLASSES[attention_variant],
                )
            else:
                components = evaluate_customsa(
                    length,
                    system.device,
                    dtype,
                    spec.q_tile_size,
                    spec.kv_tile_size,
                )
            rows.append(build_modeled_row(spec, length, variant_spec, components, system))

    rows.sort(key=lambda row: (int(row["prefill_length"]), str(row["variant"])))
    speedups = build_speedup_rows(rows, variants_to_compare)
    write_csv(os.path.join(output_dir, "attention_latency.csv"), rows)
    write_csv(os.path.join(output_dir, "case_speedups.csv"), speedups)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "device": spec.key,
                "case_name": spec.case_name,
                "case_label": spec.case_label,
                "system_config": spec.system_config,
                "system_name": raw_specs["name"],
                "data_type": spec.data_type_name,
                "source_attention_dir": source_dir_override,
                "lengths": lengths,
                "variants": variants_to_compare,
                "q_tile_size": spec.q_tile_size,
                "kv_tile_size": spec.kv_tile_size,
                "flashattention_is_double_buffering": is_double_buffering,
                "adjust_global_buffer_to_memory_capacity": spec.adjust_global_buffer_to_memory_capacity,
                "ignore_hbm_bottleneck": spec.ignore_hbm_bottleneck,
                "ignore_onchip_io_bottleneck": spec.ignore_onchip_io_bottleneck,
                "note": "Rows at source lengths reuse the supplied attention_latency.csv; longer rows use the fixed tile mapping from the source run.",
            },
            f,
            indent=2,
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--devices", default="b200,awsv4,tpuv6e")
    parser.add_argument(
        "--lengths",
        default=",".join(str(item) for item in DEFAULT_LENGTHS),
    )
    parser.add_argument("--int-softmax-comparison", action="store_true")
    parser.add_argument("--no-source-rows", action="store_true")
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        metavar="DEVICE=PATH",
        help="Fresh attention source directory for one device; may be repeated.",
    )
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    lengths = parse_int_list(args.lengths)
    device_keys = [item.strip() for item in args.devices.split(",") if item.strip()]
    variants_to_compare = (
        INT_SOFTMAX_VARIANTS if args.int_softmax_comparison else DEFAULT_VARIANTS
    )
    use_source_rows = not args.no_source_rows and not args.int_softmax_comparison
    source_dir_overrides = {}
    for item in args.source_dir:
        if "=" not in item:
            raise ValueError(f"Expected DEVICE=PATH for --source-dir, got {item!r}")
        key, path = item.split("=", 1)
        if key not in DEVICES:
            raise ValueError(f"Unknown source-dir device {key!r}")
        source_dir_overrides[key] = path
    outputs = {}
    for key in device_keys:
        outputs[key] = run_device(
            DEVICES[key],
            lengths,
            output_root,
            variants_to_compare,
            use_source_rows,
            source_dir_overrides.get(key),
        )
        print(f"{key}: {outputs[key]}", flush=True)
    with open(os.path.join(output_root, "metadata.json"), "w") as f:
        json.dump(
            {
                "outputs": outputs,
                "lengths": lengths,
                "variants": variants_to_compare,
                "use_source_rows": use_source_rows,
                "source_dir_overrides": source_dir_overrides,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
