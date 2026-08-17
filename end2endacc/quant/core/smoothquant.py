"""Offline LayerNorm/linear rescaling used by static SmoothQuant runs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .rtn import EPS


SMOOTHQUANT_GROUPS = {
    "input_layernorm": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ),
    "post_attention_layernorm": (
        "mlp.gate_proj",
        "mlp.up_proj",
    ),
}


def _linear_input_absmax(linear_modules: Sequence[nn.Linear]) -> torch.Tensor:
    if not linear_modules:
        raise ValueError("SmoothQuant needs at least one downstream linear module.")
    input_features = linear_modules[0].weight.shape[1]
    maxima: list[torch.Tensor] = []
    for linear in linear_modules:
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"SmoothQuant expected nn.Linear, got {type(linear).__name__}.")
        if linear.weight.shape[1] != input_features:
            raise ValueError("All linears in a SmoothQuant group must share input features.")
        maxima.append(linear.weight.detach().float().abs().amax(dim=0).cpu())
    return torch.stack(maxima).amax(dim=0)


def smooth_layernorm_and_linears(
    *,
    layernorm: nn.Module,
    linear_modules: Sequence[nn.Linear],
    activation_scale: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply the function-preserving SmoothQuant channel transformation."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"SmoothQuant alpha must be in [0, 1], got {alpha}.")
    if not hasattr(layernorm, "weight") or layernorm.weight is None:
        raise TypeError("SmoothQuant expects a LayerNorm/RMSNorm with a weight parameter.")

    act = activation_scale.detach().float().reshape(-1).cpu().clamp(min=EPS)
    weight = _linear_input_absmax(linear_modules).clamp(min=EPS)
    if act.shape != weight.shape:
        raise ValueError(
            f"Activation scale shape {tuple(act.shape)} does not match linear input shape "
            f"{tuple(weight.shape)}."
        )
    scale = act.pow(alpha) / weight.pow(1.0 - alpha)
    scale = scale.clamp(min=EPS)

    with torch.no_grad():
        norm_scale = scale.to(device=layernorm.weight.device, dtype=layernorm.weight.dtype)
        layernorm.weight.div_(norm_scale)
        bias = getattr(layernorm, "bias", None)
        if bias is not None:
            bias.div_(norm_scale.to(device=bias.device, dtype=bias.dtype))
        for linear in linear_modules:
            linear_scale = scale.to(device=linear.weight.device, dtype=linear.weight.dtype)
            linear.weight.mul_(linear_scale.reshape(1, -1))
    return scale
