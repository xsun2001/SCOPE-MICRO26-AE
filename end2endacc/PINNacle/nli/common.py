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


def normalize_nli_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("backend") == "nli":
        return payload

    if "function" in payload and "result" in payload:
        function = payload["function"]
        result = payload["result"]
        args = payload.get("args", {})
        return {
            "backend": "nli",
            "target": function.get("name", "unknown"),
            "description": function.get("description", ""),
            "l_range": args.get("l_range"),
            "r_range": args.get("r_range"),
            "macro_cutpoints": args.get("macro_cutpoints", len(result.get("macro_x", []))),
            "micro_bins": args.get("micro_bins"),
            "lut_entries": len(result.get("lut_y", [])),
            "macro_x": result["macro_x"],
            "lut_x": result.get("lut_x"),
            "lut_y": result["lut_y"],
            "interval_bins": result["interval_bins"],
            "interval_scales": result["interval_scales"],
            "interval_bases": result["interval_bases"],
            "metrics": result.get("metrics"),
        }

    raise ValueError("Unsupported NLI payload format. Expected `artifact.json` or `summary.json` from train/baselines/nli.py.")


def approximation_artifact_metadata(path: str | Path) -> dict[str, Any]:
    payload = normalize_nli_payload(load_lut_payload(path))
    return {
        "path": str(resolve_repo_path(path)),
        "target": payload.get("target"),
        "description": payload.get("description"),
        "macro_cutpoints": payload.get("macro_cutpoints"),
        "micro_bins": payload.get("micro_bins"),
        "lut_entries": payload.get("lut_entries"),
        "domain": [payload.get("l_range"), payload.get("r_range")],
        "metrics": payload.get("metrics"),
        "runtime_domain": "mixed_float_or_integer_q",
    }
class ScalarNliLut(nn.Module):
    def __init__(
        self,
        lut_path: str | Path,
        *,
        quantize_weights: bool,
        args: Any,
        use_integer_control: bool = False,
    ) -> None:
        super().__init__()
        payload = normalize_nli_payload(load_lut_payload(lut_path))
        self.lut_path = str(resolve_repo_path(lut_path))
        self.target = str(payload.get("target", "unknown"))
        self.description = str(payload.get("description", ""))
        self.macro_cutpoints = int(payload.get("macro_cutpoints", 0))
        self.micro_bins = int(payload.get("micro_bins", 0))
        self.lut_entries = int(payload.get("lut_entries", 0))
        self.use_integer_control = bool(use_integer_control)

        source_macro_x = torch.tensor(payload["macro_x"], dtype=torch.float32)
        lut_y = torch.tensor(payload["lut_y"], dtype=torch.float32)
        interval_bins = torch.tensor(payload["interval_bins"], dtype=torch.int64)
        interval_bases = torch.tensor(payload["interval_bases"], dtype=torch.int64)

        if lut_y.numel() < 2:
            raise ValueError("NLI LUT must contain at least two entries.")
        if source_macro_x.numel() < 2:
            raise ValueError("NLI macro cutpoints must contain at least two entries.")
        if interval_bins.numel() != source_macro_x.numel() - 1:
            raise ValueError("NLI interval_bins must have len(macro_x) - 1 entries.")
        if interval_bases.numel() != interval_bins.numel():
            raise ValueError("NLI interval_bases must match interval count.")

        if quantize_weights:
            lut_y_codes, lut_y_scale = quantize_tensor_to_int_codes(lut_y, n_bits=args.w_bits)
            self.register_buffer("lut_y_codes", lut_y_codes)
            self.register_buffer("lut_y_scale", lut_y_scale)
            self.register_buffer("lut_y", torch.empty(0, dtype=torch.float32))
            self.weight_encoding = f"int{args.w_bits}"
        else:
            self.register_buffer("lut_y_codes", torch.empty(0, dtype=torch.int32))
            self.register_buffer("lut_y_scale", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer("lut_y", lut_y)
            self.weight_encoding = "fp32"

        self.register_buffer("macro_x_source", source_macro_x.detach().float().clone())
        if self.use_integer_control:
            self.register_buffer("macro_x", torch.empty(0, dtype=torch.float32))
            self.register_buffer("interval_scales", torch.empty(0, dtype=torch.float32))
            self.register_buffer("macro_x_q", torch.empty(0, dtype=torch.int32))
            self.register_buffer("interval_width_q", torch.empty(0, dtype=torch.int32))
            self.control_encoding = "int32_from_input_scale"
            self.control_source_encoding = "fp32_artifact_macro_x"
        else:
            interval_scales = torch.tensor(payload["interval_scales"], dtype=torch.float32)
            if interval_scales.numel() != interval_bins.numel():
                raise ValueError("NLI interval_scales must match interval count.")
            self.register_buffer("macro_x", source_macro_x)
            self.register_buffer("interval_scales", interval_scales)
            self.register_buffer("macro_x_q", torch.empty(0, dtype=torch.int32))
            self.register_buffer("interval_width_q", torch.empty(0, dtype=torch.int32))
            self.control_encoding = "fp32_macro_x_scales"
            self.control_source_encoding = "registered_fp32"

        self.register_buffer("interval_bins", interval_bins)
        self.register_buffer("interval_bases", interval_bases)
        self.register_buffer("control_input_scale", torch.tensor(0.0, dtype=torch.float32))

    def _apply(self, fn):
        lut_y = self.lut_y.detach().float().clone()
        interval_scales = self.interval_scales.detach().float().clone()
        lut_y_scale = self.lut_y_scale.detach().float().clone()
        control_input_scale = self.control_input_scale.detach().float().clone()
        macro_x = self.macro_x.detach().float().clone()
        macro_x_source = self.macro_x_source.detach().float().clone()
        super()._apply(fn)
        device = self.interval_bins.device
        self.lut_y = lut_y.to(device=device)
        self.interval_scales = interval_scales.to(device=device)
        self.lut_y_scale = lut_y_scale.to(device=device)
        self.control_input_scale = control_input_scale.to(device=device)
        self.macro_x = macro_x.to(device=device)
        self.macro_x_source = macro_x_source.to(device=device)
        return self

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "path": self.lut_path,
            "target": self.target,
            "description": self.description,
            "macro_cutpoints": self.macro_cutpoints,
            "micro_bins": self.micro_bins,
            "lut_entries": self.lut_entries,
            "runtime_domain": "integer_q" if self.use_integer_control else "float",
            "control_encoding": self.control_encoding,
            "control_source_encoding": self.control_source_encoding,
            "weight_encoding": self.weight_encoding,
        }

    def _lut_y_float(self, device: torch.device) -> torch.Tensor:
        if self.lut_y_codes.numel() > 0:
            return dequantize_tensor_from_codes(self.lut_y_codes, self.lut_y_scale, device=device)
        return self.lut_y.to(device=device, dtype=torch.float32)

    def _refresh_integer_control(self, input_scale: torch.Tensor | float, device: torch.device) -> None:
        scale_value = float(torch.as_tensor(input_scale, dtype=torch.float32).clamp(min=1e-8).item())
        if (
            self.macro_x_q.device == device
            and self.macro_x_q.numel() == self.macro_x_source.numel()
            and abs(float(self.control_input_scale.detach().float().item()) - scale_value) <= 1e-12
        ):
            return

        macro_x_q = torch.round(self.macro_x_source.to(device=device) / scale_value).to(torch.int32)
        if macro_x_q.numel() > 1:
            macro_x_q = macro_x_q.clone()
            for idx in range(1, macro_x_q.numel()):
                minimum = macro_x_q[idx - 1] + 1
                if macro_x_q[idx] < minimum:
                    macro_x_q[idx] = minimum
        interval_width_q = macro_x_q[1:] - macro_x_q[:-1]
        interval_width_q = torch.maximum(interval_width_q, torch.ones_like(interval_width_q))

        self.macro_x_q = macro_x_q.to(device=device)
        self.interval_width_q = interval_width_q.to(device=device)
        self.control_input_scale = torch.tensor(scale_value, dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor, *, input_scale: torch.Tensor | None = None) -> torch.Tensor:
        interval_bins = self.interval_bins.to(device=x.device)
        interval_bases = self.interval_bases.to(device=x.device)
        lut_y = self._lut_y_float(x.device)

        if not x.is_floating_point():
            if input_scale is None:
                raise ValueError("Integer-domain NLI lookup requires the activation quantizer scale.")
            self._refresh_integer_control(input_scale, x.device)
            macro_x_q = self.macro_x_q.to(device=x.device)
            interval_width_q = self.interval_width_q.to(device=x.device)
            clipped_q = torch.clamp(
                x.to(torch.int32),
                min=int(macro_x_q[0].item()),
                max=int(macro_x_q[-1].item()),
            )
            interval_idx = torch.bucketize(clipped_q, macro_x_q[1:-1], right=True)
            bins = interval_bins[interval_idx].to(torch.int64)
            widths = interval_width_q[interval_idx].to(torch.int64)
            bases = interval_bases[interval_idx]
            left_q = macro_x_q[:-1][interval_idx].to(torch.int64)

            scaled_position = (clipped_q.to(torch.int64) - left_q) * bins
            address = torch.div(scaled_position, widths, rounding_mode="floor")
            address = torch.minimum(torch.maximum(address, torch.zeros_like(address)), bins - 1)
            decimal_numerator = scaled_position - address * widths
            decimal = decimal_numerator.to(torch.float32) / widths.to(torch.float32).clamp(min=1.0)

            global_index = bases + address.to(torch.int64)
            left_values = lut_y[global_index]
            right_values = lut_y[global_index + 1]
            output = left_values + decimal * (right_values - left_values)
            return output.to(torch.float32)

        input_dtype = x.dtype
        x_fp = x.to(torch.float32)
        if self.use_integer_control:
            macro_x = self.macro_x_source.to(device=x.device)
            interval_scales = interval_bins.to(torch.float32) / (macro_x[1:] - macro_x[:-1]).clamp(min=1e-12)
        else:
            macro_x = self.macro_x.to(device=x.device)
            interval_scales = self.interval_scales.to(device=x.device)

        clipped = torch.clamp(x_fp, min=macro_x[0].item(), max=macro_x[-1].item())
        interval_idx = torch.bucketize(clipped, macro_x[1:-1], right=True)
        left = macro_x[:-1][interval_idx]
        scales = interval_scales[interval_idx]
        bins = interval_bins[interval_idx]
        bases = interval_bases[interval_idx]

        position = (clipped - left) * scales
        address = torch.floor(position).to(torch.int64)
        address = torch.minimum(torch.maximum(address, torch.zeros_like(address)), bins - 1)
        decimal = position - address.to(position.dtype)

        global_index = bases + address
        left_values = lut_y[global_index]
        right_values = lut_y[global_index + 1]
        output = left_values + decimal * (right_values - left_values)
        return output.to(input_dtype)
