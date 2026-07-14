from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def load_lut_payload(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def quantize_tensor_to_int_codes(tensor: torch.Tensor, *, n_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_bits < 2:
        raise ValueError("Integer code quantization requires at least 2 bits.")

    q_max = 2 ** (n_bits - 1) - 1
    q_min = -(2 ** (n_bits - 1))
    amax = float(tensor.detach().abs().max().item())
    if amax <= 0.0:
        scale = torch.tensor(1.0, dtype=torch.float32)
        codes = torch.zeros_like(tensor, dtype=torch.int32)
        return codes, scale

    scale = torch.tensor(amax / q_max, dtype=torch.float32)
    codes = torch.round(tensor / scale).clamp(q_min, q_max).to(torch.int32)
    return codes, scale


def dequantize_tensor_from_codes(codes: torch.Tensor, scale: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    return codes.to(device=device, dtype=torch.float32) * scale.to(device=device, dtype=torch.float32)
