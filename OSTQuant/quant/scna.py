from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _default_artifact_path(dim: int) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root
        / "end2endacc"
        / "PINNacle"
        / "pinn"
        / "sw"
        / "attn"
        / f"d{dim}"
        / f"best_mlp1_scalar_expo_{dim}.pth"
    )


class SCNAExpApproximator(nn.Module):
    """Scalar exp(x) approximator: sum_i c_i * ReLU(w_i * x + b_i)."""

    def __init__(self, dim: int, artifact_root: str | None = None):
        super().__init__()
        if dim not in {8, 16, 32}:
            raise ValueError(f"Unsupported SCNA dimension: {dim}")

        if artifact_root:
            weight_path = Path(artifact_root) / f"d{dim}" / f"best_mlp1_scalar_expo_{dim}.pth"
        else:
            weight_path = _default_artifact_path(dim)
        if not weight_path.exists():
            raise FileNotFoundError(f"SCNA weight file not found: {weight_path}")

        state_dict = _load_state_dict(weight_path)
        self.dim = dim
        self.artifact_path = str(weight_path)
        self.register_buffer("fc1_weight", torch.exp(state_dict["fc1_weight_raw"].float()).view(dim))
        self.register_buffer("fc1_bias", torch.exp(state_dict["fc1_bias_raw"].float()).view(dim))
        self.register_buffer("fc2_weight", torch.exp(state_dict["fc2_weight_raw"].float()).view(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x.float()
        weight = self.fc1_weight.to(device=work.device, dtype=work.dtype)
        bias = self.fc1_bias.to(device=work.device, dtype=work.dtype)
        out_weight = self.fc2_weight.to(device=work.device, dtype=work.dtype)
        return _SCNAExpFunction.apply(work, weight, bias, out_weight)


class _SCNAExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, work, weight, bias, out_weight):
        out = torch.zeros_like(work)
        for idx in range(weight.numel()):
            out.add_(out_weight[idx] * F.relu(work * weight[idx] + bias[idx]))
        ctx.save_for_backward(work, weight, bias, out_weight)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        work, weight, bias, out_weight = ctx.saved_tensors
        grad_work = torch.zeros_like(work)
        grad = grad_output.to(dtype=work.dtype)
        for idx in range(weight.numel()):
            active = (work * weight[idx] + bias[idx]) > 0
            grad_work.add_(grad * (out_weight[idx] * weight[idx]) * active.to(dtype=work.dtype))
        return grad_work, None, None, None
