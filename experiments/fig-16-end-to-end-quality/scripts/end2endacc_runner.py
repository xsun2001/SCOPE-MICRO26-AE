from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(os.environ.get("BUNDLE_ROOT", Path(__file__).resolve().parents[3])).resolve()
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", BUNDLE_ROOT / "models")).resolve()
END2ENDACC_ROOT = BUNDLE_ROOT / "end2endacc"
WIKITEXT_SCRIPT = END2ENDACC_ROOT / "scripts" / "evaluation_wikitext.sh"
LMEVAL_SCRIPT = END2ENDACC_ROOT / "scripts" / "evaluation_lm_eval.sh"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def resolve_path(value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        resolved = path
    else:
        if path.parts and path.parts[0] == "models":
            resolved = (MODEL_ROOT.joinpath(*path.parts[1:])).resolve()
        else:
            resolved = (BUNDLE_ROOT / path).resolve()
    # Some legacy Llama-3 experiment configs point at a non-existent
    # Meta-Llama-3-* leaf under the local model directory. When that
    # leaf is missing but the parent model dir exists, use the parent.
    if (
        not resolved.exists()
        and resolved.parent.exists()
        and resolved.name.startswith("Meta-Llama-3-")
    ):
        return str(resolved.parent)
    return str(resolved)


def set_env(env: dict[str, str], key: str, value: Any) -> None:
    if value is None:
        env.pop(key, None)
        return
    env[key] = str(value)


def set_bool_env(env: dict[str, str], key: str, value: bool) -> None:
    env[key] = "1" if value else "0"


def build_common_env(config: dict[str, Any], output_dir: Path) -> dict[str, str]:
    runtime = config.get("runtime", {})
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_env(env, "MODEL", resolve_path(runtime.get("model", "models/Llama-2-7b-hf")))
    set_env(env, "TORCH_DTYPE", runtime.get("torch_dtype", "bfloat16"))
    set_env(env, "JOB_NAME", runtime.get("job_name"))
    set_bool_env(env, "TRUST_REMOTE_CODE", bool(runtime.get("trust_remote_code", False)))
    set_bool_env(env, "USE_FAST_TOKENIZER", bool(runtime.get("use_fast_tokenizer", False)))
    set_env(env, "OUTPUT_DIR", str(output_dir))

    approximation = config.get("approximation", {})
    pinn = config.get("pinn", {})
    pinn_quant = config.get("pinn_quant", {})

    approx_backend = approximation.get("backend")
    if approx_backend is None:
        approx_backend = "pinn" if pinn.get("enabled", False) else "none"
    approx_scope = approximation.get("scope")
    if approx_backend != "none" and approx_scope is None:
        raise ValueError(
            "Experiment configs with approximation enabled must set `approximation.scope` explicitly. "
            "Use `\"scope\": \"attn\"` for exp-only comparisons."
        )
    if approx_scope is None:
        approx_scope = "attn"
    approx_quant_weights = approximation.get("quant_weights")
    if approx_quant_weights is None:
        approx_quant_weights = pinn_quant.get("weights", False)
    approx_quant_activations = approximation.get("quant_activations")
    if approx_quant_activations is None:
        approx_quant_activations = pinn_quant.get("activations", False)

    set_env(env, "APPROX_BACKEND", approx_backend)
    set_env(env, "APPROX_SCOPE", approx_scope)
    set_env(env, "APPROX_EXP_LUT_PATH", resolve_path(approximation.get("artifacts", {}).get("exp_lut")))
    set_env(env, "APPROX_EXP_LUT_BITS", approximation.get("exp_lut_bits"))
    set_bool_env(env, "QUANT_APPROX_WEIGHTS", bool(approx_quant_weights))
    set_bool_env(env, "QUANT_APPROX_ACTIVATIONS", bool(approx_quant_activations))

    set_bool_env(env, "PINN", approx_backend == "pinn")
    set_env(env, "PINN_DIM", approximation.get("dimension", pinn.get("dimension", 16)))
    set_bool_env(env, "QUANT_PINN_WEIGHTS", bool(approx_quant_weights))
    set_env(env, "W_BITS", approximation.get("weight_bits", pinn_quant.get("weight_bits", 8)))
    set_env(
        env,
        "W_MANTISSA_BIT",
        approximation.get("weight_mantissa_bits", pinn_quant.get("weight_mantissa_bits", 2)),
    )
    set_bool_env(
        env,
        "W_PER_TENSOR",
        bool(approximation.get("weight_per_tensor", pinn_quant.get("weight_per_tensor", False))),
    )
    set_bool_env(env, "QUANT_PINN_ACTIVATIONS", bool(approx_quant_activations))
    set_env(env, "A_BITS", approximation.get("activation_bits", pinn_quant.get("activation_bits", 8)))
    set_env(
        env,
        "A_MANTISSA_BIT",
        approximation.get("activation_mantissa_bits", pinn_quant.get("activation_mantissa_bits", 2)),
    )
    set_bool_env(
        env,
        "A_PER_TENSOR",
        bool(approximation.get("activation_per_tensor", pinn_quant.get("activation_per_tensor", True))),
    )
    set_bool_env(env, "FPQ", bool(approximation.get("fpq", pinn_quant.get("fpq", False))))

    backbone = config.get("backbone_quant", {})
    set_bool_env(env, "QUANT_BACKBONE", bool(backbone.get("enabled", False)))
    set_env(env, "BACKBONE_W_BITS", backbone.get("weight_bits", 8))
    set_env(env, "BACKBONE_A_BITS", backbone.get("activation_bits", 8))
    set_env(env, "BACKBONE_WEIGHT_DTYPE", backbone.get("weight_dtype", "int8"))
    set_env(env, "BACKBONE_ACTIVATION_DTYPE", backbone.get("activation_dtype", "int8"))
    set_env(env, "BACKBONE_WEIGHT_SCHEME", backbone.get("weight_scheme", "per_channel"))
    set_env(env, "BACKBONE_ACT_SCHEME", backbone.get("activation_scheme", "per_tensor"))
    set_env(env, "BACKBONE_CALIBRATION", backbone.get("calibration_mode", "static"))
    set_env(env, "BACKBONE_CALIBRATION_SAMPLES", backbone.get("calibration_samples"))
    set_env(env, "BACKBONE_CALIBRATION_SEQ_LEN", backbone.get("calibration_seq_len"))
    set_bool_env(env, "BACKBONE_SMOOTHQUANT", bool(backbone.get("smoothquant", False)))
    set_env(env, "BACKBONE_SMOOTHQUANT_ALPHA", backbone.get("smoothquant_alpha"))
    set_bool_env(env, "QUANTIZE_LM_HEAD", bool(backbone.get("quantize_lm_head", False)))
    return env


def build_wikitext_env(config: dict[str, Any], output_dir: Path) -> dict[str, str]:
    env = build_common_env(config, output_dir)
    dataset = config.get("dataset", {})
    limits = config.get("limits", {})
    set_env(env, "DATASET_NAME", dataset.get("name", "wikitext"))
    set_env(env, "DATASET_CONFIG", dataset.get("config", "wikitext-2-raw-v1"))
    set_env(env, "DATASET_SPLIT", dataset.get("split", "test"))
    set_env(env, "DATASET_TEXT_COLUMN", dataset.get("text_column", "text"))
    set_env(env, "DATASET_JOINER", dataset.get("joiner", "\n\n"))
    set_env(env, "SEQUENCE_LENGTH", dataset.get("sequence_length", 2048))
    set_env(env, "BATCH_SIZE", dataset.get("batch_size", 1))
    set_env(env, "NUM_SAMPLES", limits.get("num_samples"))
    set_env(env, "MAX_BLOCKS", limits.get("max_blocks"))
    set_env(env, "MAX_CHUNKS", limits.get("max_chunks"))
    set_env(env, "MAX_TOKENS", limits.get("max_tokens"))
    return env


def build_lmeval_env(config: dict[str, Any], output_dir: Path) -> dict[str, str]:
    env = build_common_env(config, output_dir)
    eval_config = config.get("evaluation", {})
    set_env(env, "TASK_GROUP", eval_config.get("task_group", "group1"))
    set_env(env, "LIMIT", eval_config.get("limit"))
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an end2endacc experiment from a JSON config.")
    parser.add_argument("--kind", choices=["wikitext", "lm_eval"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    config = load_json(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "experiment_config.json",
        {
            "kind": args.kind,
            "config_path": str(args.config.resolve()),
            "config": config,
        },
    )

    if args.kind == "wikitext":
        env = build_wikitext_env(config, args.output_dir)
        script = WIKITEXT_SCRIPT
    else:
        env = build_lmeval_env(config, args.output_dir)
        script = LMEVAL_SCRIPT

    completed = subprocess.run(["bash", str(script)], cwd=END2ENDACC_ROOT, env=env)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
