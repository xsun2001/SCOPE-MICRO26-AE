from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..lut_quantization import (
    dequantize_tensor_from_codes,
    load_lut_payload,
    quantize_tensor_to_int_codes,
    resolve_repo_path,
)


def approximation_artifact_metadata(path: str | Path, decimal_bits: int | None) -> dict[str, Any]:
    payload = load_lut_payload(path)
    if len(payload) != 1:
        raise ValueError("GQA-LUT payload must contain exactly one target function root key.")
    act_func = next(iter(payload))
    available_bits = sorted(int(bit) for bit in payload[act_func].keys())
    selected_bits = available_bits[-1] if decimal_bits is None else int(decimal_bits)
    if str(selected_bits) not in payload[act_func]:
        raise ValueError(
            f"GQA-LUT payload at {resolve_repo_path(path)} does not contain decimal-bit table {selected_bits}. "
            f"Available: {available_bits}"
        )
    return {
        "path": str(resolve_repo_path(path)),
        "target": act_func,
        "available_decimal_bits": available_bits,
        "selected_decimal_bits": selected_bits,
        "num_segments": len(payload[act_func][str(selected_bits)]["slopes"]),
        "runtime_domain": "integer_q",
    }
class ScalarGqaLut(nn.Module):
    def __init__(
        self,
        lut_path: str | Path,
        *,
        decimal_bits: int | None,
        quantize_weights: bool = False,
        args: Any | None = None,
    ) -> None:
        super().__init__()
        payload = load_lut_payload(lut_path)
        if len(payload) != 1:
            raise ValueError("GQA-LUT payload must contain exactly one target function root key.")

        self.lut_path = str(resolve_repo_path(lut_path))
        self.target = next(iter(payload))
        self.available_decimal_bits = sorted(int(bit) for bit in payload[self.target].keys())
        self.decimal_bits = self.available_decimal_bits[-1] if decimal_bits is None else int(decimal_bits)
        if str(self.decimal_bits) not in payload[self.target]:
            raise ValueError(
                f"GQA-LUT payload {self.lut_path} does not contain decimal-bit table {self.decimal_bits}. "
                f"Available: {self.available_decimal_bits}"
            )

        table = payload[self.target][str(self.decimal_bits)]
        breakpoints = torch.tensor(table["breakpoints"], dtype=torch.float32)
        slopes = torch.tensor(table["slopes"], dtype=torch.float32)
        intercepts = torch.tensor(table["intercepts"], dtype=torch.float32)
        if slopes.numel() != intercepts.numel():
            raise ValueError("GQA-LUT slopes and intercepts must have the same length.")
        if breakpoints.numel() + 1 != slopes.numel():
            raise ValueError("GQA-LUT expects len(breakpoints) + 1 == len(slopes).")

        scale = 2.0 ** (-self.decimal_bits)
        breakpoint_q = torch.round(breakpoints / scale).to(torch.int32)
        intercept_q = torch.round(intercepts / scale).to(torch.int32)

        if quantize_weights:
            if args is None:
                raise ValueError("GQA-LUT runtime weight quantization requires approximation args.")
            slope_codes, slope_scale = quantize_tensor_to_int_codes(slopes, n_bits=args.w_bits)
            self.register_buffer("slope_codes", slope_codes)
            self.register_buffer("slope_scale", slope_scale)
            self.register_buffer("slopes", torch.empty(0, dtype=torch.float32))
            self.weight_encoding = f"int{args.w_bits}"
        else:
            self.register_buffer("slope_codes", torch.empty(0, dtype=torch.int32))
            self.register_buffer("slope_scale", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer("slopes", slopes)
            self.weight_encoding = "fp32"

        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("breakpoints_q", breakpoint_q)
        self.register_buffer("intercepts_q", intercept_q)

    def _apply(self, fn):
        scale = self.scale.detach().float().clone()
        slope_scale = self.slope_scale.detach().float().clone()
        slopes = self.slopes.detach().float().clone()
        super()._apply(fn)
        device = self.breakpoints_q.device
        self.scale = scale.to(device=device)
        self.slope_scale = slope_scale.to(device=device)
        self.slopes = slopes.to(device=device)
        return self

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "path": self.lut_path,
            "target": self.target,
            "available_decimal_bits": self.available_decimal_bits,
            "selected_decimal_bits": self.decimal_bits,
            "num_segments": int(self.breakpoints_q.numel() + 1),
            "scale": float(self.scale.item()),
            "runtime_domain": "integer_q",
            "weight_encoding": self.weight_encoding,
        }

    def forward(self, x: torch.Tensor, *, input_scale: torch.Tensor | None = None) -> torch.Tensor:
        input_is_integer = not x.is_floating_point()
        output_dtype = torch.float32 if input_is_integer else x.dtype
        scale = self.scale.to(device=x.device, dtype=torch.float32)

        if input_is_integer:
            q_values = x.to(torch.int32)
            if input_scale is not None:
                ratio = torch.as_tensor(input_scale, device=x.device, dtype=torch.float32).clamp(min=1e-8) / scale
                q_values = torch.round(q_values.to(torch.float32) * ratio).to(torch.int32)
        else:
            x_fp = x.to(torch.float32)
            q_values = torch.round(x_fp / scale).to(torch.int32)

        breakpoints_q = self.breakpoints_q.to(device=x.device)
        intercepts_q = self.intercepts_q.to(device=x.device)
        segment_idx = torch.bucketize(q_values, breakpoints_q, right=True)
        q_values_fp = q_values.to(torch.float32)
        if self.slope_codes.numel() > 0:
            slopes = dequantize_tensor_from_codes(self.slope_codes, self.slope_scale, device=x.device)
        else:
            slopes = self.slopes.to(device=x.device, dtype=torch.float32)
        out = (slopes[segment_idx] * q_values_fp + intercepts_q[segment_idx].to(torch.float32)) * scale
        return out.to(output_dtype)
