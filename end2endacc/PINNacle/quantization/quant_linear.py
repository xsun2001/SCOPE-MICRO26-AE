from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from quant.core.rtn import (
    EPS,
    compute_symmetric_scale_per_axis,
    compute_symmetric_scale_per_tensor,
    compute_symmetric_scale_per_token,
)


_SUPPORTED_BITS = {8, 16}
_SUPPORTED_DTYPES = {"int8", "fp8"}


def _ensure_supported_bits(bits: int, label: str) -> None:
    if bits not in _SUPPORTED_BITS:
        raise ValueError(
            f"{label} only supports {sorted(_SUPPORTED_BITS)} bits in v1, got {bits}."
        )


def _ensure_supported_quant_dtype(dtype_name: str, label: str) -> None:
    if dtype_name not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"{label} only supports {sorted(_SUPPORTED_DTYPES)} in v1, got {dtype_name!r}."
        )


def _fp8_dtype() -> torch.dtype:
    return torch.float8_e4m3fn


def _fp8_limit() -> float:
    return float(torch.finfo(_fp8_dtype()).max)


def _resolve_weight_scale(weight: torch.Tensor, scheme: str) -> torch.Tensor:
    if scheme == "per_channel":
        return compute_symmetric_scale_per_axis(weight.detach().float(), spec=_INT8_SPEC, axis=0)
    if scheme == "per_tensor":
        return compute_symmetric_scale_per_tensor(weight.detach().float(), spec=_INT8_SPEC).reshape(1)
    raise ValueError(
        f"Unsupported backbone weight quantization scheme '{scheme}'. "
        "Supported values: per_channel, per_tensor."
    )


def _broadcast_weight_scale(scale: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if scale.numel() == 1:
        return scale.reshape(1, 1)
    return scale.reshape(weight.shape[0], 1)


def _resolve_fp8_weight_scale(weight: torch.Tensor, scheme: str) -> torch.Tensor:
    source = weight.detach().float()
    limit = _fp8_limit()
    if scheme == "per_channel":
        reduce_dims = tuple(dim for dim in range(source.ndim) if dim != 0)
        scale = source.abs().amax(dim=reduce_dims)
        return torch.clamp(scale / limit, min=EPS)
    if scheme == "per_tensor":
        scale = source.abs().max()
        return torch.clamp(scale / limit, min=EPS).reshape(1)
    raise ValueError(
        f"Unsupported backbone weight quantization scheme '{scheme}'. "
        "Supported values: per_channel, per_tensor."
    )


def _resolve_dynamic_activation_scale(inputs: torch.Tensor, scheme: str) -> torch.Tensor:
    if scheme == "per_token":
        return compute_symmetric_scale_per_token(inputs.detach().float(), spec=_INT8_SPEC)
    if scheme == "per_tensor":
        return compute_symmetric_scale_per_tensor(inputs.detach().float(), spec=_INT8_SPEC).reshape(1)
    raise ValueError(
        f"Unsupported backbone activation quantization scheme '{scheme}'. "
        "Supported values: per_tensor, per_token."
    )


def _resolve_dynamic_fp8_activation_scale(inputs: torch.Tensor, scheme: str) -> torch.Tensor:
    source = inputs.detach().float()
    limit = _fp8_limit()
    if scheme == "per_token":
        scale = source.abs().amax(dim=-1, keepdim=True)
        return torch.clamp(scale / limit, min=EPS)
    if scheme == "per_tensor":
        scale = source.abs().max()
        return torch.clamp(scale / limit, min=EPS).reshape(1)
    raise ValueError(
        f"Unsupported backbone activation quantization scheme '{scheme}'. "
        "Supported values: per_tensor, per_token."
    )


def _broadcast_activation_scale(scale: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 0 or scale.numel() == 1:
        return scale.reshape([1] * inputs.ndim)
    if scale.ndim == 1 and scale.shape[0] == inputs.shape[-1]:
        return scale.reshape([1] * (inputs.ndim - 1) + [scale.shape[0]])
    if scale.shape == inputs.shape[:-1] + (1,):
        return scale
    if scale.ndim == inputs.ndim and scale.shape[1:] == inputs.shape[1:-1] + (1,) and scale.shape[0] == 1:
        return scale
    if scale.ndim == inputs.ndim - 1:
        return scale.unsqueeze(-1)
    raise ValueError(
        f"Could not broadcast activation scale with shape {tuple(scale.shape)} "
        f"to input tensor shape {tuple(inputs.shape)}."
    )


class _Int8Spec:
    qmin = -127
    qmax = 127


_INT8_SPEC = _Int8Spec()


class BackboneQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bits: int,
        activation_bits: int,
        *,
        weight_dtype_name: str = "int8",
        activation_dtype_name: str = "int8",
        weight_scheme: str = "per_channel",
        activation_scheme: str = "per_tensor",
        activation_quant_mode: str = "dynamic",
        activation_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        _ensure_supported_bits(weight_bits, "Backbone weight quantization")
        _ensure_supported_bits(activation_bits, "Backbone activation quantization")
        _ensure_supported_quant_dtype(weight_dtype_name, "Backbone weight dtype")
        _ensure_supported_quant_dtype(activation_dtype_name, "Backbone activation dtype")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_dtype_name = weight_dtype_name
        self.activation_dtype_name = activation_dtype_name
        self.weight_scheme = weight_scheme
        self.activation_scheme = activation_scheme
        self.activation_quant_mode = activation_quant_mode

        if weight_bits < 16:
            qweight_dtype = torch.int8 if weight_dtype_name == "int8" else torch.uint8
            self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=qweight_dtype))
            self.register_buffer("weight_scale", torch.ones(out_features, dtype=torch.float32))
            self.register_buffer("weight_fp", None)
        else:
            self.register_buffer("qweight", None)
            self.register_buffer("weight_scale", None)
            self.register_buffer("weight_fp", torch.empty(out_features, in_features, dtype=torch.float32))

        if activation_bits < 16:
            self.register_buffer("activation_scale", None if activation_scale is None else activation_scale.detach().float().cpu().contiguous())
        else:
            self.register_buffer("activation_scale", None)
        self.register_buffer("bias", None if bias is None else bias.detach().cpu().contiguous())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        weight_bits: int,
        activation_bits: int,
        weight_dtype_name: str = "int8",
        activation_dtype_name: str = "int8",
        weight_scheme: str = "per_channel",
        activation_scheme: str = "per_tensor",
        activation_quant_mode: str = "dynamic",
        activation_scale: Optional[torch.Tensor] = None,
    ) -> "BackboneQuantLinear":
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            weight_dtype_name=weight_dtype_name,
            activation_dtype_name=activation_dtype_name,
            weight_scheme=weight_scheme,
            activation_scheme=activation_scheme,
            activation_quant_mode=activation_quant_mode,
            activation_scale=activation_scale,
            bias=linear.bias,
        )
        device = linear.weight.device
        compute_dtype = linear.weight.dtype
        if weight_bits < 16:
            if weight_dtype_name == "fp8":
                weight_scale = _resolve_fp8_weight_scale(linear.weight, weight_scheme).to(torch.float32)
                normalized = (
                    linear.weight.detach().float()
                    / _broadcast_weight_scale(weight_scale, linear.weight)
                ).clamp(-_fp8_limit(), _fp8_limit())
                qweight = normalized.to(_fp8_dtype()).contiguous().view(torch.uint8)
            else:
                weight_scale = _resolve_weight_scale(linear.weight, weight_scheme).to(torch.float32)
                qweight = torch.round(
                    linear.weight.detach().float() / _broadcast_weight_scale(weight_scale, linear.weight)
                ).clamp(_INT8_SPEC.qmin, _INT8_SPEC.qmax).to(torch.int8)
            module.qweight = qweight.to(device=device)
            module.weight_scale = weight_scale.to(device=device)
        else:
            module.weight_fp = linear.weight.detach().to(device=device, dtype=compute_dtype).contiguous()
        if module.activation_scale is not None:
            module.activation_scale = module.activation_scale.to(device=device, dtype=torch.float32)
        if module.bias is not None and linear.bias is not None:
            module.bias = module.bias.to(device=device, dtype=linear.bias.dtype)
        return module

    def dequantize_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.weight_bits >= 16:
            return self.weight_fp.to(device=device, dtype=dtype)
        scale = self.weight_scale.to(device=device, dtype=torch.float32)
        qweight = self.qweight.to(device=device)
        if self.weight_dtype_name == "fp8":
            dequant = qweight.contiguous().view(_fp8_dtype()).to(torch.float32)
            return (dequant * _broadcast_weight_scale(scale, qweight)).to(dtype)
        return (qweight.to(torch.float32) * _broadcast_weight_scale(scale, qweight)).to(dtype)

    def quantize_inputs(self, inputs: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        if self.activation_bits >= 16:
            return inputs.to(compute_dtype)
        if self.activation_quant_mode == "static":
            if self.activation_scale is None:
                raise ValueError(
                    "Static activation quantization requested, but activation_scale was not provided."
                )
            scale = self.activation_scale.to(device=inputs.device, dtype=torch.float32).clamp(min=EPS)
            if self.activation_scheme == "per_token" and scale.ndim >= 2:
                seq_len = inputs.shape[-2]
                if scale.shape[0] < seq_len:
                    raise ValueError(
                        f"Static per-token activation scale has only {scale.shape[0]} token positions, "
                        f"but the input requires {seq_len}."
                    )
                scale = scale[:seq_len]
                expand_prefix = [1] * max(inputs.ndim - scale.ndim, 0)
                scale = scale.reshape(expand_prefix + list(scale.shape))
                if self.activation_dtype_name == "fp8":
                    normalized = (inputs.to(torch.float32) / scale).clamp(-_fp8_limit(), _fp8_limit())
                    qinputs = normalized.to(_fp8_dtype()).contiguous().view(torch.uint8)
                    return (qinputs.contiguous().view(_fp8_dtype()).to(torch.float32) * scale).to(compute_dtype)
                qinputs = torch.round(inputs.to(torch.float32) / scale).clamp(_INT8_SPEC.qmin, _INT8_SPEC.qmax)
                return (qinputs * scale).to(compute_dtype)
        elif self.activation_quant_mode == "dynamic":
            if self.activation_dtype_name == "fp8":
                scale = _resolve_dynamic_fp8_activation_scale(inputs, self.activation_scheme).to(
                    device=inputs.device,
                    dtype=torch.float32,
                )
            else:
                scale = _resolve_dynamic_activation_scale(inputs, self.activation_scheme).to(
                    device=inputs.device,
                    dtype=torch.float32,
                )
        else:
            raise ValueError(
                f"Unsupported backbone activation quantization mode '{self.activation_quant_mode}'. "
                "Supported values: dynamic, static."
            )
        if self.activation_dtype_name == "fp8":
            broadcast_scale = _broadcast_activation_scale(scale, inputs)
            normalized = (inputs.to(torch.float32) / broadcast_scale).clamp(-_fp8_limit(), _fp8_limit())
            qinputs = normalized.to(_fp8_dtype()).contiguous().view(torch.uint8)
            return (qinputs.contiguous().view(_fp8_dtype()).to(torch.float32) * broadcast_scale).to(compute_dtype)
        qinputs = torch.round(
            inputs.to(torch.float32) / _broadcast_activation_scale(scale, inputs)
        ).clamp(_INT8_SPEC.qmin, _INT8_SPEC.qmax)
        return (qinputs * _broadcast_activation_scale(scale, inputs)).to(compute_dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        compute_dtype = inputs.dtype if inputs.dtype.is_floating_point else torch.float32
        weight = self.dequantize_weight(compute_dtype, inputs.device)
        quantized_inputs = self.quantize_inputs(inputs, compute_dtype)
        bias = self.bias
        if bias is not None:
            bias = bias.to(device=inputs.device, dtype=compute_dtype)
        return F.linear(quantized_inputs, weight, bias)

    def quant_metadata(self) -> Dict[str, object]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "weight_dtype": self.weight_dtype_name,
            "activation_dtype": self.activation_dtype_name,
            "weight_scheme": self.weight_scheme,
            "activation_scheme": self.activation_scheme,
            "activation_quant_mode": self.activation_quant_mode,
            "has_bias": self.bias is not None,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_bits={self.weight_bits}, activation_bits={self.activation_bits}, "
            f"weight_dtype='{self.weight_dtype_name}', activation_dtype='{self.activation_dtype_name}', "
            f"weight_scheme='{self.weight_scheme}', activation_scheme='{self.activation_scheme}', "
            f"activation_quant_mode='{self.activation_quant_mode}', bias={self.bias is not None}"
        )
