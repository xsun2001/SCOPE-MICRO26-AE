from __future__ import annotations

import torch


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def hadamard_transform(x: torch.Tensor, scale=1.0) -> torch.Tensor:
    """Torch fallback for Dao-AILab fast_hadamard_transform.hadamard_transform."""

    n = x.shape[-1]
    if not _is_power_of_two(n):
        raise ValueError(f"Hadamard transform requires a power-of-two last dimension, got {n}")

    original_shape = x.shape
    y = x.contiguous().reshape(-1, n)
    block = 1
    while block < n:
        y = y.view(-1, n // (2 * block), 2 * block)
        left = y[..., :block].clone()
        right = y[..., block : 2 * block].clone()
        y[..., :block] = left + right
        y[..., block : 2 * block] = left - right
        y = y.view(-1, n)
        block *= 2

    if isinstance(scale, torch.Tensor):
        scale = scale.to(device=y.device, dtype=y.dtype)
    return (y * scale).reshape(original_shape)
