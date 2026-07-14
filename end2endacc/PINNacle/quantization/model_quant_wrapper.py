from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from datasets import load_dataset
from torch import nn
from tqdm import tqdm

from quant.core.observers import VectorAbsMaxObserver
from quant.core.smoothquant import SMOOTHQUANT_GROUPS as LLAMA_SMOOTHQUANT_GROUPS
from quant.core.smoothquant import smooth_layernorm_and_linears

from .quant_linear import BackboneQuantLinear

_STATIC_SCALE_EPS = 1e-8

LLAMA_TARGET_LINEAR_SUFFIXES = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}

OPT_TARGET_LINEAR_SUFFIXES = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "fc1",
    "fc2",
}

MODEL_TARGET_LINEAR_SUFFIXES = {
    "llama": LLAMA_TARGET_LINEAR_SUFFIXES,
    "opt": OPT_TARGET_LINEAR_SUFFIXES,
}

TARGET_LINEAR_SUFFIXES = set().union(*MODEL_TARGET_LINEAR_SUFFIXES.values())

OPT_SMOOTHQUANT_GROUPS = {
    "self_attn_layer_norm": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ),
    "final_layer_norm": ("fc1",),
}

MODEL_SMOOTHQUANT_LAYOUTS = {
    "llama": {
        "groups": LLAMA_SMOOTHQUANT_GROUPS,
        "root_prefix": "model.layers",
        "layers_attr": ("model", "layers"),
    },
    "opt": {
        "groups": OPT_SMOOTHQUANT_GROUPS,
        "root_prefix": "model.decoder.layers",
        "layers_attr": ("model", "decoder", "layers"),
    },
}


def _model_family(model) -> str:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if isinstance(model_type, str):
        normalized = model_type.lower()
        if normalized in MODEL_TARGET_LINEAR_SUFFIXES:
            return normalized

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama"
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return "opt"

    raise ValueError(
        "Unsupported model family for backbone quantization. "
        f"Expected one of {sorted(MODEL_TARGET_LINEAR_SUFFIXES)}, "
        f"got model_type={model_type!r}."
    )


def _target_linear_suffixes_for_model(model) -> set[str]:
    return MODEL_TARGET_LINEAR_SUFFIXES[_model_family(model)]


def _resolve_attr_path(root: nn.Module, parts: Tuple[str, ...]):
    current = root
    for part in parts:
        current = getattr(current, part)
    return current


def _resolve_module(root: nn.Module, module_name: str) -> nn.Module:
    current = root
    for part in module_name.split("."):
        current = getattr(current, part)
    return current


def _smoothquant_layout(model):
    family = _model_family(model)
    layout = MODEL_SMOOTHQUANT_LAYOUTS[family]
    try:
        layers = _resolve_attr_path(model, layout["layers_attr"])
    except AttributeError as exc:
        raise ValueError(
            f"SmoothQuant expects decoder layers at {'.'.join(layout['layers_attr'])} "
            f"for model family '{family}'."
        ) from exc
    return family, layers, layout["root_prefix"], layout["groups"]


def _is_opt_decoder_mlp_linear(name: str) -> bool:
    parts = name.split(".")
    return (
        len(parts) == 5
        and parts[0] == "model"
        and parts[1] == "decoder"
        and parts[2] == "layers"
        and parts[3].isdigit()
        and parts[4] in {"fc1", "fc2"}
    )


def _is_target_linear(
    name: str,
    module: nn.Module,
    quantize_lm_head: bool,
    model_family: str,
    target_linear_suffixes: set[str],
) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if name == "lm_head":
        return quantize_lm_head
    if model_family == "opt" and (name.endswith(".fc1") or name.endswith(".fc2")):
        return _is_opt_decoder_mlp_linear(name)
    return any(name.endswith(suffix) for suffix in target_linear_suffixes)


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent, attr_name = _get_parent_module(root, module_name)
    setattr(parent, attr_name, new_module)


def _iter_calibration_texts(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_column: str,
    limit: int,
):
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    emitted = 0
    for row in dataset:
        text = row[text_column].strip()
        if not text:
            continue
        yield text
        emitted += 1
        if emitted >= limit:
            return


def _collect_packed_calibration_batches(
    tokenizer,
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_column: str,
    limit: int,
    seq_len: int,
    joiner: str = "\n\n",
) -> list[torch.Tensor]:
    texts = list(
        _iter_calibration_texts(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            text_column=text_column,
            limit=limit,
        )
    )
    if not texts:
        return []
    input_ids = tokenizer(joiner.join(texts), return_tensors="pt").input_ids
    if input_ids.numel() == 0:
        return []

    total_tokens = input_ids.shape[-1]
    batches: list[torch.Tensor] = []
    for start in range(0, total_tokens, seq_len):
        end = min(start + seq_len, total_tokens)
        block = input_ids[:, start:end]
        if block.numel() > 0:
            batches.append(block)
    return batches


def _reduce_channel_absmax(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 1:
        raise ValueError("Expected tensor with at least one dimension for channel reduction.")
    source = tensor.detach().float().abs()
    if tensor.ndim == 1:
        return source
    return source.amax(dim=tuple(range(source.ndim - 1)))


def _reduce_activation_stat(tensor: torch.Tensor, act_scheme: str) -> torch.Tensor:
    source = tensor.detach().float().abs()
    if act_scheme == "per_tensor":
        return source.max().reshape(())
    if act_scheme == "per_channel":
        return _reduce_channel_absmax(source)
    if act_scheme == "per_token":
        token_max = source.amax(dim=-1, keepdim=True)
        if token_max.ndim > 2:
            reduce_dims = tuple(range(token_max.ndim - 2))
            token_max = token_max.amax(dim=reduce_dims)
        return token_max
    raise ValueError(
        f"Unsupported static backbone activation scheme '{act_scheme}'. "
        "Supported values: per_tensor, per_channel, per_token."
    )


def _merge_activation_stat(
    previous: Optional[torch.Tensor | float],
    current: torch.Tensor,
) -> torch.Tensor:
    current_tensor = current.detach().float().cpu()
    if previous is None:
        return current_tensor

    if isinstance(previous, torch.Tensor):
        previous_tensor = previous.detach().float().cpu()
    else:
        previous_tensor = torch.tensor(float(previous), dtype=torch.float32)

    if previous_tensor.ndim == 0 and current_tensor.ndim == 0:
        return torch.maximum(previous_tensor, current_tensor)
    if previous_tensor.shape == current_tensor.shape:
        return torch.maximum(previous_tensor, current_tensor)
    if (
        previous_tensor.ndim == 2
        and current_tensor.ndim == 2
        and previous_tensor.shape[1] == 1
        and current_tensor.shape[1] == 1
    ):
        max_len = max(previous_tensor.shape[0], current_tensor.shape[0])
        merged = torch.zeros(max_len, 1, dtype=torch.float32)
        merged[: previous_tensor.shape[0]] = previous_tensor
        merged[: current_tensor.shape[0]] = torch.maximum(merged[: current_tensor.shape[0]], current_tensor)
        return merged
    raise ValueError(
        f"Could not merge activation statistics with shapes "
        f"{tuple(previous_tensor.shape)} and {tuple(current_tensor.shape)}."
    )


def _finalize_activation_stats(
    stats: Dict[str, torch.Tensor | float],
    *,
    activation_dtype_name: str,
) -> Dict[str, torch.Tensor | float]:
    if activation_dtype_name == "fp8":
        scale_divisor = float(torch.finfo(torch.float8_e4m3fn).max)
    elif activation_dtype_name == "int8":
        scale_divisor = 127.0
    else:
        raise ValueError(f"Unsupported activation dtype for calibration export: {activation_dtype_name!r}")
    resolved: Dict[str, torch.Tensor | float] = {}
    for name, max_abs in stats.items():
        if isinstance(max_abs, torch.Tensor):
            if max_abs.numel() == 0:
                continue
            if max_abs.max().item() <= 0.0:
                continue
            scale = torch.clamp(max_abs.detach().float().cpu() / scale_divisor, min=_STATIC_SCALE_EPS)
            resolved[name] = float(scale.item()) if scale.ndim == 0 else scale
            continue
        if max_abs > 0.0:
            resolved[name] = max(max_abs / scale_divisor, _STATIC_SCALE_EPS)
    return resolved


def _collect_backbone_calibration_stats_from_batches(
    model,
    *,
    batches: Iterable[torch.Tensor],
    total_batches: int,
    unit: str,
    desc: str,
    device: str,
    act_scheme: str,
    activation_dtype_name: str,
    quantize_lm_head: bool,
) -> Dict[str, torch.Tensor | float]:
    stats: Dict[str, torch.Tensor | float] = {}
    hooks = []
    model_family = _model_family(model)
    target_linear_suffixes = _target_linear_suffixes_for_model(model)

    def _make_hook(module_name: str):
        def _hook(_module, inputs):
            if not inputs:
                return
            tensor = inputs[0]
            if tensor.numel() == 0:
                return
            reduced = _reduce_activation_stat(tensor, act_scheme)
            stats[module_name] = _merge_activation_stat(stats.get(module_name), reduced)

        return _hook

    for module_name, module in model.named_modules():
        if not _is_target_linear(
            module_name,
            module,
            quantize_lm_head,
            model_family,
            target_linear_suffixes,
        ):
            continue
        hooks.append(module.register_forward_pre_hook(_make_hook(module_name)))

    previous_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    model.eval()
    try:
        with torch.inference_mode():
            for input_ids in tqdm(batches, total=total_batches, desc=desc, unit=unit):
                if input_ids.numel() == 0:
                    continue
                model(input_ids.to(device))
    finally:
        model.config.use_cache = previous_use_cache
        for hook in hooks:
            hook.remove()

    return _finalize_activation_stats(stats, activation_dtype_name=activation_dtype_name)


def _register_smoothquant_observers(
    model,
) -> Tuple[Dict[str, VectorAbsMaxObserver], list[torch.utils.hooks.RemovableHandle]]:
    _, layers, root_prefix, smoothquant_groups = _smoothquant_layout(model)
    observers: Dict[str, VectorAbsMaxObserver] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx, layer in enumerate(layers):
        for norm_suffix in smoothquant_groups:
            module_name = f"{root_prefix}.{layer_idx}.{norm_suffix}"
            observer = VectorAbsMaxObserver()
            observers[module_name] = observer
            module = getattr(layer, norm_suffix)

            def _make_hook(store: VectorAbsMaxObserver):
                def _hook(_module, _inputs, output):
                    store.update(_reduce_channel_absmax(output))

                return _hook

            hooks.append(module.register_forward_hook(_make_hook(observer)))

    return observers, hooks


def _apply_smoothquant_scales(
    model,
    *,
    activation_scales_by_module: Dict[str, list[float]],
    alpha: float,
) -> Dict[str, list[float]]:
    _, layers, root_prefix, smoothquant_groups = _smoothquant_layout(model)
    applied_scales: Dict[str, list[float]] = {}

    for layer_idx, layer in enumerate(layers):
        for norm_suffix, linear_suffixes in smoothquant_groups.items():
            norm_name = f"{root_prefix}.{layer_idx}.{norm_suffix}"
            if norm_name not in activation_scales_by_module:
                continue
            layernorm = getattr(layer, norm_suffix)
            linear_modules = [
                _resolve_module(layer, linear_suffix)
                for linear_suffix in linear_suffixes
            ]
            scale_tensor = torch.tensor(
                activation_scales_by_module[norm_name],
                dtype=torch.float32,
            )
            applied = smooth_layernorm_and_linears(
                layernorm=layernorm,
                linear_modules=linear_modules,
                activation_scale=scale_tensor,
                alpha=alpha,
            )
            applied_scales[norm_name] = [float(value) for value in applied.tolist()]
    return applied_scales


def _apply_smoothquant_from_batches(
    model,
    *,
    batches: Iterable[torch.Tensor],
    total_batches: int,
    device: str,
    alpha: float,
) -> Dict[str, object]:
    smoothquant_observers, smoothquant_hooks = _register_smoothquant_observers(model)
    previous_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    model.eval()
    try:
        with torch.inference_mode():
            for input_ids in tqdm(batches, total=total_batches, desc="smoothquant", unit="block"):
                model(input_ids.to(device))
    finally:
        model.config.use_cache = previous_use_cache
        for hook in smoothquant_hooks:
            hook.remove()

    layernorm_activation_scales = {
        name: observer.to_list() for name, observer in smoothquant_observers.items()
    }
    applied_scales = _apply_smoothquant_scales(
        model,
        activation_scales_by_module=layernorm_activation_scales,
        alpha=alpha,
    )
    return {
        "enabled": True,
        "alpha": alpha,
        "layernorm_activation_scales": layernorm_activation_scales,
        "applied_scales": applied_scales,
    }


def apply_backbone_smoothquant(
    model,
    tokenizer,
    *,
    device: str,
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_split: str,
    text_column: str,
    num_samples: int,
    seq_len: int,
    act_scheme: str,
    activation_dtype_name: str,
    alpha: float,
) -> Dict[str, object]:
    batches = _collect_packed_calibration_batches(
        tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=dataset_split,
        text_column=text_column,
        limit=num_samples,
        seq_len=seq_len,
    )
    if not batches:
        raise ValueError("SmoothQuant calibration produced no packed token blocks.")

    smoothquant = _apply_smoothquant_from_batches(
        model,
        batches=batches,
        total_batches=len(batches),
        device=device,
        alpha=alpha,
    )
    return export_backbone_calibration_stats(
        scales={},
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        text_column=text_column,
        num_samples=num_samples,
        seq_len=seq_len,
        act_scheme=act_scheme,
        activation_dtype_name=activation_dtype_name,
        calibration_layout="packed_blocks",
        calibration_blocks=len(batches),
        smoothquant=smoothquant,
    )


def collect_backbone_smoothquant_stats(
    model,
    tokenizer,
    *,
    device: str,
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_split: str,
    text_column: str,
    num_samples: int,
    seq_len: int,
    act_scheme: str,
    activation_dtype_name: str,
    quantize_lm_head: bool,
    alpha: float,
) -> Tuple[Dict[str, torch.Tensor | float], Dict[str, object]]:
    if act_scheme != "per_channel":
        raise ValueError(
            "SmoothQuant backbone calibration expects `backbone_act_scheme=per_channel` "
            "so the post-smoothing activation scales stay channel-wise and static."
        )

    batches = _collect_packed_calibration_batches(
        tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=dataset_split,
        text_column=text_column,
        limit=num_samples,
        seq_len=seq_len,
    )
    if not batches:
        raise ValueError("SmoothQuant calibration produced no packed token blocks.")

    smoothquant = _apply_smoothquant_from_batches(
        model,
        batches=batches,
        total_batches=len(batches),
        device=device,
        alpha=alpha,
    )

    activation_scales = _collect_backbone_calibration_stats_from_batches(
        model,
        batches=batches,
        total_batches=len(batches),
        unit="block",
        desc="backbone-calibration",
        device=device,
        act_scheme=act_scheme,
        activation_dtype_name=activation_dtype_name,
        quantize_lm_head=quantize_lm_head,
    )
    payload = export_backbone_calibration_stats(
        scales=activation_scales,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        text_column=text_column,
        num_samples=num_samples,
        seq_len=seq_len,
        act_scheme=act_scheme,
        activation_dtype_name=activation_dtype_name,
        calibration_layout="packed_blocks",
        calibration_blocks=len(batches),
        smoothquant=smoothquant,
    )
    return activation_scales, payload


def collect_backbone_calibration_stats(
    model,
    tokenizer,
    *,
    device: str,
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_split: str,
    text_column: str,
    num_samples: int,
    seq_len: int,
    act_scheme: str,
    activation_dtype_name: str,
    quantize_lm_head: bool,
) -> Dict[str, torch.Tensor | float]:
    sample_batches = []
    iterator = _iter_calibration_texts(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=dataset_split,
        text_column=text_column,
        limit=num_samples,
    )
    for text in iterator:
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
        ).input_ids
        if input_ids.numel() > 0:
            sample_batches.append(input_ids)
    return _collect_backbone_calibration_stats_from_batches(
        model,
        batches=sample_batches,
        total_batches=len(sample_batches),
        unit="sample",
        desc="backbone-calibration",
        device=device,
        act_scheme=act_scheme,
        activation_dtype_name=activation_dtype_name,
        quantize_lm_head=quantize_lm_head,
    )


def export_backbone_calibration_stats(
    *,
    scales: Dict[str, object],
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_split: str,
    text_column: str,
    num_samples: int,
    seq_len: int,
    act_scheme: str,
    activation_dtype_name: str,
    calibration_layout: str = "sample_rows",
    calibration_blocks: Optional[int] = None,
    smoothquant: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    payload = {
        "activation_scales": scales,
        "calibration": {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "text_column": text_column,
            "num_samples": num_samples,
            "seq_len": seq_len,
            "activation_scheme": act_scheme,
            "activation_dtype": activation_dtype_name,
            "layout": calibration_layout,
        },
    }
    if calibration_blocks is not None:
        payload["calibration"]["num_blocks"] = calibration_blocks
    if smoothquant is not None:
        payload["smoothquant"] = smoothquant
    return payload


def apply_backbone_quantization(
    model,
    *,
    weight_bits: int,
    activation_bits: int,
    weight_dtype_name: str,
    activation_dtype_name: str,
    weight_scheme: str,
    activation_scheme: str,
    activation_quant_mode: str,
    activation_scales: Optional[Dict[str, torch.Tensor | float]],
    quantize_lm_head: bool,
) -> Dict[str, Dict[str, object]]:
    replaced: Dict[str, Dict[str, object]] = {}
    model_family = _model_family(model)
    target_linear_suffixes = _target_linear_suffixes_for_model(model)
    for module_name, module in model.named_modules():
        if not _is_target_linear(
            module_name,
            module,
            quantize_lm_head,
            model_family,
            target_linear_suffixes,
        ):
            continue
        scale_tensor = None
        if activation_bits < 16 and activation_quant_mode == "static":
            if activation_scales is None or module_name not in activation_scales:
                raise ValueError(
                    f"Missing static backbone activation scale for module '{module_name}'."
                )
            scale_value = activation_scales[module_name]
            if isinstance(scale_value, torch.Tensor):
                scale_tensor = scale_value.detach().float().clone()
            else:
                scale_tensor = torch.tensor([scale_value], dtype=torch.float32)
        quant_linear = BackboneQuantLinear.from_linear(
            module,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            weight_dtype_name=weight_dtype_name,
            activation_dtype_name=activation_dtype_name,
            weight_scheme=weight_scheme,
            activation_scheme=activation_scheme,
            activation_quant_mode=activation_quant_mode,
            activation_scale=scale_tensor,
        )
        _replace_module(model, module_name, quant_linear)
        replaced[module_name] = quant_linear.quant_metadata()
    return replaced
