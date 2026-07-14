import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List


EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))
LLMCOMPASS_ROOT = os.path.join(REPO_ROOT, "LLMCompass")
DEFAULT_CONFIG = os.path.join(EXPERIMENT_DIR, "configs", "default.json")

if LLMCOMPASS_ROOT not in sys.path:
    sys.path.insert(0, LLMCOMPASS_ROOT)

from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.utils import data_type_dict


@dataclass
class ModelSpec:
    name: str
    hidden_size: int
    num_heads: int
    num_layers: int
    intermediate_size: int
    vocab_size: int
    logit_positions: int
    data_type_name: str


@dataclass
class ExperimentSpec:
    experiment_name: str
    system_config_path: str
    batch_size: int
    ignore_hbm_bottleneck: bool
    ignore_onchip_io_bottleneck: bool
    model: ModelSpec
    attention_variants: List[Dict[str, str]]


DEFAULT_ATTENTION_VARIANTS = [
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


def load_attention_variants(payload: Dict[str, object]) -> List[Dict[str, str]]:
    raw_variants = payload.get("attention_variants", DEFAULT_ATTENTION_VARIANTS)
    variants = []
    seen = set()
    for item in raw_variants:
        variant = {
            "name": str(item["name"]),
            "label": str(item["label"]),
            "attention_variant": str(item["attention_variant"]),
        }
        if variant["name"] in seen:
            raise ValueError(f"Duplicate attention variant name '{variant['name']}'.")
        seen.add(variant["name"])
        variants.append(variant)
    if not variants:
        raise ValueError("At least one attention variant must be configured.")
    return variants


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


def load_spec(config_path: str) -> ExperimentSpec:
    with open(config_path, "r") as f:
        payload = json.load(f)
    config_dir = os.path.dirname(config_path)
    runtime = payload.get("runtime", {})
    model = payload["model"]
    return ExperimentSpec(
        experiment_name=payload.get(
            "experiment_name",
            "llama3_8b_full_model_attention_b200",
        ),
        system_config_path=resolve_path(config_dir, payload["system_config"]),
        batch_size=int(runtime.get("batch_size", 1)),
        ignore_hbm_bottleneck=bool(runtime.get("ignore_hbm_bottleneck", False)),
        ignore_onchip_io_bottleneck=bool(
            runtime.get("ignore_onchip_io_bottleneck", False)
        ),
        model=ModelSpec(
            name=model.get("name", "llama3_8b"),
            hidden_size=int(model["hidden_size"]),
            num_heads=int(model["num_heads"]),
            num_layers=int(model["num_layers"]),
            intermediate_size=int(model["intermediate_size"]),
            vocab_size=int(model["vocab_size"]),
            logit_positions=int(model.get("logit_positions", 1)),
            data_type_name=model.get("data_type", "fp16"),
        ),
        attention_variants=load_attention_variants(payload),
    )


def load_single_card_system(config_path: str):
    specs = read_architecture_template(config_path)
    specs["device_count"] = 1
    specs["interconnect"]["link_count_per_device"] = 0
    specs["name"] = f"{specs['name']} single-card"
    return template_to_system(specs), specs


def load_single_card_system_with_options(
    config_path: str,
    adjust_global_buffer_to_memory_capacity: bool,
):
    system, specs = load_single_card_system(config_path)
    if adjust_global_buffer_to_memory_capacity:
        system.device.compute_module.l2_size = int(
            system.device.memory_module.memory_capacity
        )
    return system, specs


def read_attention_rows(
    attention_dir: str,
    case_name: str,
    required_variants: List[str],
) -> Dict[int, Dict[str, Dict[str, str]]]:
    path = os.path.join(attention_dir, "attention_latency.csv")
    grouped: Dict[int, Dict[str, Dict[str, str]]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["case_name"] != case_name:
                continue
            length = int(row["prefill_length"])
            grouped.setdefault(length, {})[row["variant"]] = row
    required = set(required_variants)
    for length, variants in grouped.items():
        missing = required - set(variants)
        if missing:
            raise ValueError(f"Missing variants for length {length}: {sorted(missing)}")
    return grouped


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def attention_useful_flops(spec: ExperimentSpec, context_length: int) -> float:
    heads_total = spec.batch_size * spec.model.num_heads
    head_dim = spec.model.hidden_size // spec.model.num_heads
    qk_flops = 2.0 * heads_total * context_length * context_length * head_dim
    av_flops = qk_flops
    return qk_flops + av_flops


def throughput_tflops(flops: float, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return flops / (latency_ms / 1000.0) / 1e12


def tokens_per_second(tokens: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return tokens / (latency_ms / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-config", default=DEFAULT_CONFIG)
    parser.add_argument("--case-name", default="b200_fp16")
    parser.add_argument("--system-label", default="B200 FP16")
    parser.add_argument("--system-config")
    parser.add_argument("--adjust-global-buffer-to-memory-capacity", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    spec = load_spec(os.path.abspath(args.experiment_config))
    if args.system_config:
        spec.system_config_path = resolve_path(os.getcwd(), args.system_config)
    system, system_specs = load_single_card_system_with_options(
        spec.system_config_path,
        adjust_global_buffer_to_memory_capacity=args.adjust_global_buffer_to_memory_capacity,
    )
    dtype = data_type_dict[spec.model.data_type_name]
    tensor_flops = system.device.compute_module.total_systolic_array_flops_for(dtype)
    vector_unit = system.device.compute_module
    overhead = system.device.compute_module.overhead

    attention_by_length = read_attention_rows(
        os.path.abspath(args.attention_dir),
        args.case_name,
        [variant["name"] for variant in spec.attention_variants],
    )

    def matmul_ms(m: int, n: int, k: int) -> float:
        flops = 2 * m * n * k
        return (flops / tensor_flops + overhead.matmul) * 1e3

    def vector_ms(elements: int, flop_per_element: float, overhead_seconds: float = 0.0) -> float:
        flops = elements * flop_per_element
        return (vector_unit.vector_latency(flops) + overhead_seconds) * 1e3

    latency_rows: List[Dict[str, object]] = []
    speedup_rows: List[Dict[str, object]] = []
    b = spec.batch_size
    h = spec.model.hidden_size
    intermediate = spec.model.intermediate_size

    for length in sorted(attention_by_length):
        token_rows = attention_by_length[length]
        hidden_elements = b * length * h
        ffn_elements = b * length * intermediate

        attn_norm_ms = vector_ms(hidden_elements, 4, overhead.layernorm)
        residual_add1_ms = vector_ms(hidden_elements, 1)
        ffn_norm_ms = vector_ms(hidden_elements, 4, overhead.layernorm)
        residual_add2_ms = vector_ms(hidden_elements, 1)
        q_proj_ms = matmul_ms(b * length, h, h)
        k_proj_ms = matmul_ms(b * length, h, h)
        v_proj_ms = matmul_ms(b * length, h, h)
        o_proj_ms = matmul_ms(b * length, h, h)
        gate_proj_ms = matmul_ms(b * length, intermediate, h)
        up_proj_ms = matmul_ms(b * length, intermediate, h)
        silu_ms = vector_ms(ffn_elements, 8, overhead.gelu)
        gate_mul_ms = vector_ms(ffn_elements, 1)
        down_proj_ms = matmul_ms(b * length, h, intermediate)
        final_norm_ms = vector_ms(hidden_elements, 4, overhead.layernorm)
        lm_head_ms = matmul_ms(
            b * spec.model.logit_positions,
            spec.model.vocab_size,
            h,
        )

        ffn_total_ms = (
            ffn_norm_ms
            + gate_proj_ms
            + up_proj_ms
            + silu_ms
            + gate_mul_ms
            + down_proj_ms
            + residual_add2_ms
        )
        non_attention_per_layer_ms = attn_norm_ms + residual_add1_ms + ffn_total_ms
        projection_total_ms = q_proj_ms + k_proj_ms + v_proj_ms + o_proj_ms
        useful_attention_flops = attention_useful_flops(spec, length)
        tokens = b * length

        by_variant = {}
        for variant_spec in spec.attention_variants:
            variant = variant_spec["name"]
            variant_label = variant_spec["label"]
            attention_core_ms = float(token_rows[variant]["total_latency_ms"])
            attention_total_ms = projection_total_ms + attention_core_ms
            layer_total_ms = non_attention_per_layer_ms + attention_total_ms
            decoder_stack_ms = layer_total_ms * spec.model.num_layers
            model_total_ms = decoder_stack_ms + final_norm_ms + lm_head_ms
            row = {
                "model": spec.model.name,
                "system": args.system_label,
                "data_type": spec.model.data_type_name,
                "batch_size": b,
                "context_length": length,
                "variant": variant,
                "variant_label": variant_label,
                "num_layers": spec.model.num_layers,
                "attention_core_ms": attention_core_ms,
                "q_proj_ms": q_proj_ms,
                "k_proj_ms": k_proj_ms,
                "v_proj_ms": v_proj_ms,
                "o_proj_ms": o_proj_ms,
                "attention_total_ms": attention_total_ms,
                "attn_norm_ms": attn_norm_ms,
                "residual_add1_ms": residual_add1_ms,
                "ffn_norm_ms": ffn_norm_ms,
                "gate_proj_ms": gate_proj_ms,
                "up_proj_ms": up_proj_ms,
                "silu_ms": silu_ms,
                "gate_mul_ms": gate_mul_ms,
                "down_proj_ms": down_proj_ms,
                "residual_add2_ms": residual_add2_ms,
                "ffn_total_ms": ffn_total_ms,
                "non_attention_per_layer_ms": non_attention_per_layer_ms,
                "layer_total_ms": layer_total_ms,
                "decoder_stack_ms": decoder_stack_ms,
                "final_norm_ms": final_norm_ms,
                "lm_head_ms": lm_head_ms,
                "model_total_ms": model_total_ms,
                "attention_share_of_layer": attention_total_ms / layer_total_ms,
                "attention_share_of_model": (
                    attention_total_ms * spec.model.num_layers / model_total_ms
                ),
                "q_mul_k_ms": float(token_rows[variant].get("q_mul_k_ms") or 0.0),
                "softmax_ms": float(token_rows[variant].get("softmax_ms") or 0.0),
                "softmax_scale": float(
                    token_rows[variant].get("softmax_scale") or 1.0
                ),
                "a_mul_v_ms": float(token_rows[variant].get("a_mul_v_ms") or 0.0),
                "fused_core_ms": float(
                    token_rows[variant].get("fused_core_ms") or 0.0
                ),
                "attention_useful_tflops_core": throughput_tflops(
                    useful_attention_flops,
                    attention_core_ms,
                ),
                "attention_useful_tflops_total": throughput_tflops(
                    useful_attention_flops,
                    attention_total_ms,
                ),
                "model_prefill_tokens_per_s": tokens_per_second(
                    tokens,
                    model_total_ms,
                ),
                "decoder_stack_tokens_per_s": tokens_per_second(
                    tokens,
                    decoder_stack_ms,
                ),
            }
            latency_rows.append(row)
            by_variant[variant] = row

        baseline_spec = spec.attention_variants[0]
        baseline = by_variant[baseline_spec["name"]]
        for variant_spec in spec.attention_variants:
            candidate = by_variant[variant_spec["name"]]
            speedup_rows.append(
                {
                    "context_length": length,
                    "baseline_variant": baseline_spec["name"],
                    "baseline_label": baseline_spec["label"],
                    "variant": variant_spec["name"],
                    "variant_label": variant_spec["label"],
                    "baseline_model_ms": baseline["model_total_ms"],
                    "variant_model_ms": candidate["model_total_ms"],
                    "model_speedup_vs_baseline_x": (
                        baseline["model_total_ms"] / candidate["model_total_ms"]
                    ),
                    "baseline_layer_ms": baseline["layer_total_ms"],
                    "variant_layer_ms": candidate["layer_total_ms"],
                    "layer_speedup_vs_baseline_x": (
                        baseline["layer_total_ms"] / candidate["layer_total_ms"]
                    ),
                    "baseline_attention_core_ms": baseline["attention_core_ms"],
                    "variant_attention_core_ms": candidate["attention_core_ms"],
                    "attention_core_speedup_vs_baseline_x": (
                        baseline["attention_core_ms"]
                        / candidate["attention_core_ms"]
                    ),
                    "baseline_attention_total_ms": baseline["attention_total_ms"],
                    "variant_attention_total_ms": candidate["attention_total_ms"],
                    "attention_total_speedup_vs_baseline_x": (
                        baseline["attention_total_ms"]
                        / candidate["attention_total_ms"]
                    ),
                    "baseline_attention_useful_tflops_core": baseline[
                        "attention_useful_tflops_core"
                    ],
                    "variant_attention_useful_tflops_core": candidate[
                        "attention_useful_tflops_core"
                    ],
                    "baseline_model_prefill_tokens_per_s": baseline[
                        "model_prefill_tokens_per_s"
                    ],
                    "variant_model_prefill_tokens_per_s": candidate[
                        "model_prefill_tokens_per_s"
                    ],
                    "attention_share_of_model": candidate["attention_share_of_model"],
                }
            )

    latency_csv = os.path.join(output_dir, "full_model_latency.csv")
    speedup_csv = os.path.join(output_dir, "full_model_speedup.csv")
    summary_md = os.path.join(output_dir, "summary.md")
    metadata_json = os.path.join(output_dir, "metadata.json")

    write_csv(latency_csv, latency_rows)
    write_csv(speedup_csv, speedup_rows)

    metadata = {
        "experiment_name": spec.experiment_name,
        "attention_dir": os.path.relpath(os.path.abspath(args.attention_dir), REPO_ROOT),
        "experiment_config": os.path.relpath(os.path.abspath(args.experiment_config), REPO_ROOT),
        "system_config": os.path.relpath(os.path.abspath(spec.system_config_path), REPO_ROOT),
        "case_name": args.case_name,
        "system_label": args.system_label,
        "system_name": system_specs["name"],
        "tensor_tflops": tensor_flops / 1e12,
        "vector_tflops": system.device.compute_module.total_vector_flops / 1e12,
        "batch_size": b,
        "model": {
            "name": spec.model.name,
            "hidden_size": h,
            "num_heads": spec.model.num_heads,
            "num_layers": spec.model.num_layers,
            "intermediate_size": intermediate,
            "vocab_size": spec.model.vocab_size,
            "logit_positions": spec.model.logit_positions,
            "data_type": spec.model.data_type_name,
            "attention_limitation": "GQA is not modeled; attention core uses the active 32-head MHA simulator interface.",
        },
        "attention_variants": spec.attention_variants,
        "scope": "32 * one_decoder_layer + final_norm + lm_head_for_last_token",
        "common_model": "Projection/MLP/final costs use the selected system tensor/vector roofline with LLMCompass operator overheads; attention core uses the supplied attention_latency.csv.",
        "outputs": {
            "latency_csv": latency_csv,
            "speedup_csv": speedup_csv,
            "summary_md": summary_md,
        },
    }

    with open(summary_md, "w") as f:
        f.write("# Llama 3 8B Derived Full-Model Attention Summary\n\n")
        f.write(f"- Attention source: `{metadata['attention_dir']}`\n")
        f.write(f"- Attention case: `{metadata['case_name']}`\n")
        f.write(f"- System: `{metadata['system_name']}`\n")
        f.write(f"- Tensor throughput: `{metadata['tensor_tflops']:.4f}` TFLOP/s\n")
        f.write(f"- Vector throughput: `{metadata['vector_tflops']:.4f}` TFLOP/s\n")
        f.write(f"- Scope: `{metadata['scope']}`\n")
        f.write(
            "- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.\n\n"
        )
        f.write(f"- Baseline for speedups: `{spec.attention_variants[0]['label']}`\n\n")
        f.write(
            "| Context | Variant | Model ms | Tok/s | Attn core ms | "
            "Attn TFLOP/s | Model speedup | Attn-core speedup |\n"
        )
        f.write("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in speedup_rows:
            f.write(
                f"| {row['context_length']} | "
                f"{row['variant_label']} | "
                f"{row['variant_model_ms']:.3f} | "
                f"{row['variant_model_prefill_tokens_per_s']:.3f} | "
                f"{row['variant_attention_core_ms']:.3f} | "
                f"{row['variant_attention_useful_tflops_core']:.3f} | "
                f"{row['model_speedup_vs_baseline_x']:.5f}x | "
                f"{row['attention_core_speedup_vs_baseline_x']:.5f}x |\n"
            )

    with open(metadata_json, "w") as f:
        json.dump(
            {
                **metadata,
                "latency_rows": latency_rows,
                "speedup_rows": speedup_rows,
            },
            f,
            indent=2,
        )

    for row in speedup_rows:
        print(
            f"context={row['context_length']:>5} "
            f"variant={row['variant']:<14} "
            f"model_ms={row['variant_model_ms']:>12.3f} "
            f"tok_s={row['variant_model_prefill_tokens_per_s']:>12.3f} "
            f"model_speedup={row['model_speedup_vs_baseline_x']:.5f}x "
            f"attention_core_speedup={row['attention_core_speedup_vs_baseline_x']:.5f}x",
            flush=True,
        )
    print(f"[outputs] {output_dir}")


if __name__ == "__main__":
    main()
