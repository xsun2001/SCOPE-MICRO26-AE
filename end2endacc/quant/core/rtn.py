"""Scale calculation for symmetric round-to-nearest quantization."""

from __future__ import annotations

from typing import Protocol

import torch


EPS = 1e-8


class QuantSpec(Protocol):
    qmin: int
    qmax: int


def _positive_limit(spec: QuantSpec) -> float:
    limit = max(abs(int(spec.qmin)), abs(int(spec.qmax)))
    if limit <= 0:
        raise ValueError("A symmetric quantization specification needs a nonzero range.")
    return float(limit)


def compute_symmetric_scale_per_tensor(tensor: torch.Tensor, *, spec: QuantSpec) -> torch.Tensor:
    source = tensor.detach().float()
    return torch.clamp(source.abs().max() / _positive_limit(spec), min=EPS)


def compute_symmetric_scale_per_axis(
    tensor: torch.Tensor,
    *,
    spec: QuantSpec,
    axis: int,
) -> torch.Tensor:
    source = tensor.detach().float()
    normalized_axis = axis % source.ndim
    reduce_dims = tuple(dim for dim in range(source.ndim) if dim != normalized_axis)
    absmax = source.abs() if not reduce_dims else source.abs().amax(dim=reduce_dims)
    return torch.clamp(absmax / _positive_limit(spec), min=EPS)


def compute_symmetric_scale_per_token(tensor: torch.Tensor, *, spec: QuantSpec) -> torch.Tensor:
    if tensor.ndim == 0:
        raise ValueError("Per-token quantization expects at least one tensor dimension.")
    source = tensor.detach().float()
    return torch.clamp(source.abs().amax(dim=-1, keepdim=True) / _positive_limit(spec), min=EPS)
