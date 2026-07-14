from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..quantization.clip import pseudo_quantize_tensor_clip


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ScalingSpec:
    enabled: bool
    threshold: float
    factor: float


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def load_lut_payload(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def approximation_artifact_metadata(path: str | Path) -> dict[str, Any]:
    payload = load_lut_payload(path)
    return {
        "path": str(resolve_repo_path(path)),
        "target": payload.get("target"),
        "description": payload.get("description"),
        "lut_entries": payload.get("lut_entries"),
        "train_range": payload.get("train_range"),
        "eval_range": payload.get("eval_range"),
        "scaling": payload.get("scaling", {}),
    }


def _quantize_vector(
    tensor: torch.Tensor,
    *,
    n_bits: int,
    zero_point: bool,
    q_group_size: int,
    per_tensor: bool,
    fpq: bool,
    mantissa_bit: int,
    clip: bool,
) -> torch.Tensor:
    quantized = pseudo_quantize_tensor_clip(
        tensor.detach().clone().reshape(1, -1),
        n_bits=n_bits,
        zero_point=zero_point,
        q_group_size=q_group_size,
        per_tensor=per_tensor,
        fpq=fpq,
        mantissa_bit=mantissa_bit,
        clip=clip,
    )
    return quantized.reshape_as(tensor).to(dtype=tensor.dtype)


class ScalarNnLut(nn.Module):
    def __init__(self, lut_path: str | Path, *, quantize_weights: bool, args: Any) -> None:
        super().__init__()
        payload = load_lut_payload(lut_path)
        self.lut_path = str(resolve_repo_path(lut_path))
        self.target = str(payload.get("target", "unknown"))
        self.description = str(payload.get("description", ""))
        self.lut_entries = int(payload.get("lut_entries", 0))

        scaling = payload.get("scaling", {}) or {}
        self.scaling = ScalingSpec(
            enabled=bool(scaling.get("enabled", False)),
            threshold=float(scaling.get("threshold", 1.0)),
            factor=float(scaling.get("factor", 1.0)),
        )

        breakpoints = torch.tensor(payload["breakpoints"], dtype=torch.float32)
        slopes = torch.tensor(payload["slopes"], dtype=torch.float32)
        intercepts = torch.tensor(payload["intercepts"], dtype=torch.float32)

        if slopes.numel() != intercepts.numel():
            raise ValueError("NNLUT slopes and intercepts must have the same length.")
        if breakpoints.numel() + 1 != slopes.numel():
            raise ValueError("NNLUT expects len(breakpoints) + 1 == len(slopes).")

        if quantize_weights:
            breakpoints = _quantize_vector(
                breakpoints,
                n_bits=args.w_bits,
                zero_point=True,
                q_group_size=-1,
                per_tensor=True,
                fpq=args.fpq,
                mantissa_bit=args.w_mantissa_bit,
                clip=False,
            )
            breakpoints = torch.sort(breakpoints).values
            slopes = _quantize_vector(
                slopes,
                n_bits=args.w_bits,
                zero_point=args.w_zero_point,
                q_group_size=args.w_group_size,
                per_tensor=args.w_per_tensor,
                fpq=args.fpq,
                mantissa_bit=args.w_mantissa_bit,
                clip=args.w_clip,
            )
            intercepts = _quantize_vector(
                intercepts,
                n_bits=args.w_bits,
                zero_point=args.w_zero_point,
                q_group_size=args.w_group_size,
                per_tensor=args.w_per_tensor,
                fpq=args.fpq,
                mantissa_bit=args.w_mantissa_bit,
                clip=args.w_clip,
            )

        self.register_buffer("breakpoints", breakpoints)
        self.register_buffer("slopes", slopes)
        self.register_buffer("intercepts", intercepts)

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "path": self.lut_path,
            "target": self.target,
            "description": self.description,
            "lut_entries": self.lut_entries,
            "scaling": {
                "enabled": self.scaling.enabled,
                "threshold": self.scaling.threshold,
                "factor": self.scaling.factor,
            },
        }

    def _apply_runtime_scaling(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.scaling.enabled:
            return x, torch.ones_like(x)

        scaled_x = x.clone()
        output_scale = torch.ones_like(x)
        mask = x < self.scaling.threshold
        if mask.any():
            scaled_x[mask] = x[mask] * self.scaling.factor
            output_scale[mask] = math.sqrt(self.scaling.factor)
        return scaled_x, output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_x, output_scale = self._apply_runtime_scaling(x)
        breakpoints = self.breakpoints.to(device=scaled_x.device, dtype=scaled_x.dtype)
        slopes = self.slopes.to(device=scaled_x.device, dtype=scaled_x.dtype)
        intercepts = self.intercepts.to(device=scaled_x.device, dtype=scaled_x.dtype)
        segment_idx = torch.bucketize(scaled_x, breakpoints)
        return (slopes[segment_idx] * scaled_x + intercepts[segment_idx]) * output_scale
