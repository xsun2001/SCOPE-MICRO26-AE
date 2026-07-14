from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PINNacle.approximation_wrapper import apply_approximation_replacement
from PINNacle.gqalut.common import approximation_artifact_metadata as gqalut_artifact_metadata
from PINNacle.nli.common import approximation_artifact_metadata as nli_artifact_metadata
from PINNacle.nnlut.common import approximation_artifact_metadata, resolve_repo_path
from PINNacle.quantization.calibration import calibrate_static_act
from PINNacle.quantization.model_quant_wrapper import (
    apply_backbone_smoothquant,
    apply_backbone_quantization,
    collect_backbone_calibration_stats,
    collect_backbone_smoothquant_stats,
    export_backbone_calibration_stats,
)
from PINNacle.quantization.quantizer import ActQuantizer


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype '{dtype_name}'.")


def ensure_output_dir(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_output_dir(kind: str) -> str:
    base_name = "end2endacc_llama2_7b_int8_ppl" if kind == "wikitext" else "end2endacc_llama2_7b_int8_lm_eval"
    return os.path.join(repo_root(), "experiments", base_name, "results", timestamp())


def git_commit(cwd: Optional[str] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or repo_root(),
            text=True,
        ).strip()
    except Exception:
        return None


def write_json(path: str, payload: Dict[str, Any]) -> None:
    def _default(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    ensure_output_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_default)
        f.write("\n")


def normalize_args(args) -> Any:
    normalized = vars(args).copy()
    approx_backend = normalized.get("approx_backend", "none")
    if approx_backend in {None, ""}:
        approx_backend = "none"
    if approx_backend == "none" and normalized.get("pinn"):
        approx_backend = "pinn"
    normalized["approx_backend"] = approx_backend
    normalized["approx_enabled"] = approx_backend != "none"
    if normalized["approx_enabled"]:
        approx_scope = normalized.get("approx_scope")
        if approx_scope in {None, ""}:
            raise ValueError(
                "Approximation runs must pass `--approx_scope` explicitly. "
                "Use `--approx_scope attn` for exp-only comparisons."
            )
        normalized["approx_scope"] = approx_scope
    else:
        normalized["approx_scope"] = normalized.get("approx_scope") or "attn"

    normalized["quant_approx_weights"] = bool(
        normalized.get("quant_approx_weights")
        or normalized.get("quant_pinn_weights")
        or normalized.get("quant_pinn")
    )
    normalized["quant_approx_activations"] = bool(
        normalized.get("quant_approx_activations")
        or normalized.get("quant_pinn_activations")
        or normalized.get("quant_act")
    )
    normalized["quant_pinn_weights"] = normalized["quant_approx_weights"]
    normalized["quant_pinn_activations"] = normalized["quant_approx_activations"]
    normalized["quant_pinn"] = normalized["quant_approx_weights"]
    normalized["quant_act"] = normalized["quant_approx_activations"]
    normalized["pinn"] = approx_backend == "pinn"
    normalized["quant_backbone"] = bool(
        normalized.get("quant_backbone")
        or int(normalized.get("backbone_w_bits", 16)) < 16
        or int(normalized.get("backbone_a_bits", 16)) < 16
    )
    normalized["backbone_weight_dtype"] = normalized.get("backbone_weight_dtype", "int8")
    normalized["backbone_activation_dtype"] = normalized.get("backbone_activation_dtype", "int8")
    if normalized["backbone_w_bits"] not in {8, 16}:
        raise ValueError("`--backbone_w_bits` currently supports only 8 or 16.")
    if normalized["backbone_a_bits"] not in {8, 16}:
        raise ValueError("`--backbone_a_bits` currently supports only 8 or 16.")
    if normalized["backbone_weight_dtype"] not in {"int8", "fp8"}:
        raise ValueError("`--backbone_weight_dtype` currently supports only `int8` or `fp8`.")
    if normalized["backbone_activation_dtype"] not in {"int8", "fp8"}:
        raise ValueError("`--backbone_activation_dtype` currently supports only `int8` or `fp8`.")
    if normalized["backbone_weight_dtype"] == "fp8" and normalized["backbone_w_bits"] != 8:
        raise ValueError("`--backbone_weight_dtype fp8` requires `--backbone_w_bits 8`.")
    if normalized["backbone_activation_dtype"] == "fp8" and normalized["backbone_a_bits"] != 8:
        raise ValueError("`--backbone_activation_dtype fp8` requires `--backbone_a_bits 8`.")
    if normalized["backbone_act_scheme"] == "per_channel" and normalized["backbone_calibration"] != "static":
        raise ValueError("`--backbone_act_scheme per_channel` currently supports only static calibration.")
    if normalized.get("backbone_smoothquant", False):
        if not normalized["quant_backbone"]:
            raise ValueError("`--backbone_smoothquant` requires `--quant_backbone`.")
        if normalized["backbone_a_bits"] >= 16:
            raise ValueError("`--backbone_smoothquant` requires `--backbone_a_bits 8`.")
        if normalized["backbone_calibration"] == "static":
            if normalized["backbone_act_scheme"] != "per_channel":
                raise ValueError(
                    "`--backbone_smoothquant` with static activation quantization expects "
                    "`--backbone_act_scheme per_channel` so the offline activation scales stay channel-wise."
                )
        elif normalized["backbone_calibration"] == "dynamic":
            if normalized["backbone_act_scheme"] == "per_channel":
                raise ValueError(
                    "`--backbone_smoothquant` with dynamic activation quantization supports "
                    "`per_token` or `per_tensor`, not `per_channel`."
                )
        else:
            raise ValueError(
                f"Unsupported `--backbone_calibration {normalized['backbone_calibration']}` for SmoothQuant."
            )
    if not normalized["approx_enabled"] and (
        normalized["quant_approx_weights"] or normalized["quant_approx_activations"]
    ):
        raise ValueError("Approximation quantization flags require an enabled approximation backend.")
    if normalized["approx_backend"] in {"nnlut", "gqalut", "nli"} and not normalized.get("approx_exp_lut_path"):
        raise ValueError(f"{normalized['approx_backend'].upper()} backend requires `--approx_exp_lut_path`.")
    if normalized["approx_backend"] in {"nnlut", "gqalut", "nli"} and normalized["approx_scope"] != "attn":
        raise ValueError(f"{normalized['approx_backend'].upper()} backend currently supports only `--approx_scope attn`.")
    if normalized["approx_backend"] == "gqalut" and normalized.get("approx_exp_lut_bits") is None:
        normalized["approx_exp_lut_bits"] = 6
    if normalized["quant_approx_activations"] and not normalized.get("calibrate_static_act", False):
        raise ValueError(
            "Approximation activation quantization requires `--calibrate_static_act` because ActQuantizer only supports calibrated static inference here."
        )
    if normalized.get("approx_exp_lut_path"):
        normalized["approx_exp_lut_path"] = str(resolve_repo_path(normalized["approx_exp_lut_path"]))
    return SimpleNamespace(**normalized)


def add_shared_runtime_args(parser) -> None:
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", "-t", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")

    parser.add_argument(
        "--approx_backend",
        type=str,
        default="none",
        choices=["none", "pinn", "nnlut", "gqalut", "nli"],
    )
    parser.add_argument(
        "--approx_scope",
        type=str,
        default=None,
        choices=["attn", "all"],
    )
    parser.add_argument("--approx_exp_lut_path", type=str, default=None)
    parser.add_argument("--approx_exp_lut_bits", type=int, default=None)
    parser.add_argument("--quant_approx_weights", action="store_true", default=False)
    parser.add_argument("--quant_approx_activations", action="store_true", default=False)

    parser.add_argument("--pinn", action="store_true", default=False)
    parser.add_argument("--pinn_dim", type=int, default=16, choices=[8, 16, 32])
    parser.add_argument("--quant_pinn", action="store_true", default=False)
    parser.add_argument("--quant_act", action="store_true", default=False)
    parser.add_argument("--quant_pinn_weights", action="store_true", default=False)
    parser.add_argument("--quant_pinn_activations", action="store_true", default=False)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--w_zero_point", action="store_true", default=False)
    parser.add_argument("--w_group_size", type=int, default=-1)
    parser.add_argument("--w_mantissa_bit", type=int, default=2)
    parser.add_argument("--w_clip", action="store_true", default=False)
    parser.add_argument("--w_per_tensor", action="store_true", default=False)
    parser.add_argument("--a_bits", type=int, default=16)
    parser.add_argument("--a_group_size", type=int, default=-1)
    parser.add_argument("--a_mantissa_bit", type=int, default=2)
    parser.add_argument("--a_clip", action="store_true", default=False)
    parser.add_argument("--a_per_tensor", action="store_true", default=False)
    parser.add_argument("--fpq", action="store_true", default=False)
    parser.add_argument("--calibrate_static_act", action="store_true", default=False)

    parser.add_argument("--quant_backbone", action="store_true", default=False)
    parser.add_argument("--backbone_w_bits", type=int, default=16)
    parser.add_argument("--backbone_a_bits", type=int, default=16)
    parser.add_argument(
        "--backbone_weight_dtype",
        type=str,
        default="int8",
        choices=["int8", "fp8"],
    )
    parser.add_argument(
        "--backbone_activation_dtype",
        type=str,
        default="int8",
        choices=["int8", "fp8"],
    )
    parser.add_argument(
        "--backbone_weight_scheme",
        type=str,
        default="per_channel",
        choices=["per_channel", "per_tensor"],
    )
    parser.add_argument(
        "--backbone_act_scheme",
        type=str,
        default="per_tensor",
        choices=["per_tensor", "per_token", "per_channel"],
    )
    parser.add_argument(
        "--backbone_calibration",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
    )
    parser.add_argument("--backbone_calibration_dataset", type=str, default="mit-han-lab/pile-val-backup")
    parser.add_argument("--backbone_calibration_config", type=str, default=None)
    parser.add_argument("--backbone_calibration_split", type=str, default="validation")
    parser.add_argument("--backbone_calibration_text_column", type=str, default="text")
    parser.add_argument("--backbone_calibration_samples", type=int, default=512)
    parser.add_argument("--backbone_calibration_seq_len", type=int, default=2048)
    parser.add_argument("--backbone_smoothquant", action="store_true", default=False)
    parser.add_argument("--backbone_smoothquant_alpha", type=float, default=0.85)
    parser.add_argument("--quantize_lm_head", action="store_true", default=False)


def runtime_config_payload(args, *, command: str) -> Dict[str, Any]:
    return {
        "command": command,
        "argv": sys.argv,
        "git_commit": git_commit(),
        "timestamp": timestamp(),
        "args": vars(args),
    }


@dataclass
class RuntimeMetadata:
    model_path: str
    device: str
    dtype: str
    torch_dtype: str
    quant_backbone: bool
    backbone_quantization: Optional[Dict[str, Any]]
    backbone_calibration_path: Optional[str]
    approx_enabled: bool
    approx_backend: str
    approx_scope: str
    approx_quantized_weights: bool
    approx_quantized_activations: bool
    approx_artifacts: Optional[Dict[str, Any]]
    approx_calibration_path: Optional[str]
    approx_state_audit: Optional[Dict[str, Any]]


def load_model_and_tokenizer(args) -> tuple[Any, Any, torch.dtype]:
    transformers.set_seed(args.seed)
    torch_dtype = resolve_torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast_tokenizer,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(args.device)
    model.eval()
    return model, tokenizer, torch_dtype


def export_approx_calibration(model) -> Dict[str, Any]:
    quantizers: Dict[str, Dict[str, float]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, ActQuantizer):
            continue
        quantizers[name] = {
            "mode": module.mode,
            "scale": float(module.scale.detach().float().item()),
            "amax": float(module.amax.detach().float().item()),
            "emit_int_codes": bool(getattr(module, "emit_int_codes", False)),
        }
    return {
        "num_quantizers": len(quantizers),
        "quantizers": quantizers,
    }


def collect_approx_state_audit(model) -> Dict[str, Any]:
    modules: Dict[str, Dict[str, Any]] = {}
    int_code_quantizers: Dict[str, Dict[str, Any]] = {}

    for name, module in model.named_modules():
        if isinstance(module, ActQuantizer):
            int_code_quantizers[name] = {
                "mode": module.mode,
                "scale_dtype": str(module.scale.dtype),
                "emit_int_codes": bool(getattr(module, "emit_int_codes", False)),
            }

        if not hasattr(module, "metadata_dict"):
            continue

        buffers = {
            buffer_name: {
                "dtype": str(buffer.dtype),
                "numel": int(buffer.numel()),
            }
            for buffer_name, buffer in module.named_buffers(recurse=False)
        }
        parameters = {
            param_name: {
                "dtype": str(param.dtype),
                "numel": int(param.numel()),
            }
            for param_name, param in module.named_parameters(recurse=False)
        }
        metadata = module.metadata_dict() if callable(module.metadata_dict) else {}
        modules[name] = {
            "module_type": module.__class__.__name__,
            "metadata": metadata,
            "buffers": buffers,
            "parameters": parameters,
        }

    return {
        "num_modules": len(modules),
        "num_quantizers": len(int_code_quantizers),
        "modules": modules,
        "quantizers": int_code_quantizers,
    }


def collect_approx_artifacts(args) -> Optional[Dict[str, Any]]:
    if not getattr(args, "approx_enabled", False):
        return None

    payload: Dict[str, Any] = {
        "backend": args.approx_backend,
        "scope": args.approx_scope,
    }
    if args.approx_backend == "nnlut" and args.approx_exp_lut_path:
        payload["exp_lut"] = approximation_artifact_metadata(args.approx_exp_lut_path)
    if args.approx_backend == "gqalut" and args.approx_exp_lut_path:
        payload["exp_lut"] = gqalut_artifact_metadata(args.approx_exp_lut_path, args.approx_exp_lut_bits)
    if args.approx_backend == "nli" and args.approx_exp_lut_path:
        payload["exp_lut"] = nli_artifact_metadata(args.approx_exp_lut_path)
    return payload


def build_runtime(
    args,
    *,
    output_dir: str,
) -> tuple[Any, Any, Dict[str, Any], RuntimeMetadata]:
    model, tokenizer, torch_dtype = load_model_and_tokenizer(args)

    approx_artifacts = collect_approx_artifacts(args)
    if args.approx_enabled:
        model = apply_approximation_replacement(model, args, torch_dtype)
        model.eval()

    backbone_quantization = None
    backbone_calibration_path = None
    if args.quant_backbone:
        activation_scales = None
        backbone_calibration = None
        if args.backbone_smoothquant:
            if args.backbone_calibration == "static":
                activation_scales, backbone_calibration = collect_backbone_smoothquant_stats(
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    dataset_name=args.backbone_calibration_dataset,
                    dataset_config=args.backbone_calibration_config,
                    dataset_split=args.backbone_calibration_split,
                    text_column=args.backbone_calibration_text_column,
                    num_samples=args.backbone_calibration_samples,
                    seq_len=args.backbone_calibration_seq_len,
                    act_scheme=args.backbone_act_scheme,
                    activation_dtype_name=args.backbone_activation_dtype,
                    quantize_lm_head=args.quantize_lm_head,
                    alpha=args.backbone_smoothquant_alpha,
                )
            else:
                backbone_calibration = apply_backbone_smoothquant(
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    dataset_name=args.backbone_calibration_dataset,
                    dataset_config=args.backbone_calibration_config,
                    dataset_split=args.backbone_calibration_split,
                    text_column=args.backbone_calibration_text_column,
                    num_samples=args.backbone_calibration_samples,
                    seq_len=args.backbone_calibration_seq_len,
                    act_scheme=args.backbone_act_scheme,
                    activation_dtype_name=args.backbone_activation_dtype,
                    alpha=args.backbone_smoothquant_alpha,
                )
            backbone_calibration_path = os.path.join(output_dir, "backbone_calibration.json")
            write_json(backbone_calibration_path, backbone_calibration)
        elif args.backbone_a_bits < 16 and args.backbone_calibration == "static":
            activation_scales = collect_backbone_calibration_stats(
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                dataset_name=args.backbone_calibration_dataset,
                dataset_config=args.backbone_calibration_config,
                dataset_split=args.backbone_calibration_split,
                text_column=args.backbone_calibration_text_column,
                num_samples=args.backbone_calibration_samples,
                seq_len=args.backbone_calibration_seq_len,
                act_scheme=args.backbone_act_scheme,
                activation_dtype_name=args.backbone_activation_dtype,
                quantize_lm_head=args.quantize_lm_head,
            )
            backbone_calibration = export_backbone_calibration_stats(
                scales=activation_scales,
                dataset_name=args.backbone_calibration_dataset,
                dataset_config=args.backbone_calibration_config,
                dataset_split=args.backbone_calibration_split,
                text_column=args.backbone_calibration_text_column,
                num_samples=args.backbone_calibration_samples,
                seq_len=args.backbone_calibration_seq_len,
                act_scheme=args.backbone_act_scheme,
                activation_dtype_name=args.backbone_activation_dtype,
            )
            backbone_calibration_path = os.path.join(output_dir, "backbone_calibration.json")
            write_json(backbone_calibration_path, backbone_calibration)
        replaced = apply_backbone_quantization(
            model,
            weight_bits=args.backbone_w_bits,
            activation_bits=args.backbone_a_bits,
            weight_dtype_name=args.backbone_weight_dtype,
            activation_dtype_name=args.backbone_activation_dtype,
            weight_scheme=args.backbone_weight_scheme,
            activation_scheme=args.backbone_act_scheme,
            activation_quant_mode=args.backbone_calibration,
            activation_scales=activation_scales,
            quantize_lm_head=args.quantize_lm_head,
        )
        backbone_quantization = {
            "weight_bits": args.backbone_w_bits,
            "activation_bits": args.backbone_a_bits,
            "weight_dtype": args.backbone_weight_dtype,
            "activation_dtype": args.backbone_activation_dtype,
            "weight_scheme": args.backbone_weight_scheme,
            "activation_scheme": args.backbone_act_scheme,
            "activation_quant_mode": args.backbone_calibration,
            "quantize_lm_head": args.quantize_lm_head,
            "num_replaced_modules": len(replaced),
            "replaced_modules": replaced,
            "attention_path_note": "backbone linear W/A quantization only; attention QK^T, AV, and exact softmax path remain exact in v1",
        }
        if args.backbone_smoothquant:
            backbone_quantization["smoothquant"] = {
                "enabled": True,
                "alpha": args.backbone_smoothquant_alpha,
            }

    approx_calibration_path = None
    approx_calibration = None
    if args.quant_approx_activations:
        calibrate_static_act(
            model,
            tokenizer,
            num_samples=args.backbone_calibration_samples,
            seq_len=args.backbone_calibration_seq_len,
        )
        approx_calibration = export_approx_calibration(model)
        approx_calibration_path = os.path.join(output_dir, "approx_activation_calibration.json")
        write_json(approx_calibration_path, approx_calibration)

    approx_state_audit = collect_approx_state_audit(model) if args.approx_enabled else None

    metadata = RuntimeMetadata(
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        torch_dtype=str(torch_dtype),
        quant_backbone=args.quant_backbone,
        backbone_quantization=backbone_quantization,
        backbone_calibration_path=backbone_calibration_path,
        approx_enabled=args.approx_enabled,
        approx_backend=args.approx_backend,
        approx_scope=args.approx_scope,
        approx_quantized_weights=args.quant_approx_weights,
        approx_quantized_activations=args.quant_approx_activations,
        approx_artifacts=approx_artifacts,
        approx_calibration_path=approx_calibration_path,
        approx_state_audit=approx_state_audit,
    )
    extras = {
        "backbone_quantization": backbone_quantization,
        "approximation": approx_artifacts,
        "approx_calibration": approx_calibration,
        "approx_state_audit": approx_state_audit,
    }
    return model, tokenizer, extras, metadata


def save_runtime_config(output_dir: str, payload: Dict[str, Any], metadata: RuntimeMetadata) -> None:
    config_payload = dict(payload)
    config_payload["runtime"] = asdict(metadata)
    write_json(os.path.join(output_dir, "config.json"), config_payload)
