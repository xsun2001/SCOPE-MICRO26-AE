from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TensorFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ScalingSpec:
    enabled: bool = False
    threshold: float = 1.0
    factor: float = 1.0
    train_lo: float | None = None
    train_hi: float | None = None


@dataclass(frozen=True)
class TargetSpec:
    name: str
    description: str
    target_fn: TensorFn
    eval_lo: float
    eval_hi: float
    n_sign: str
    b_sign: str
    scaling: ScalingSpec = ScalingSpec()


@dataclass(frozen=True)
class LutTable:
    breakpoints: torch.Tensor
    slopes: torch.Tensor
    intercepts: torch.Tensor


def target_gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="none")


def target_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


def target_divide(x: torch.Tensor) -> torch.Tensor:
    return torch.reciprocal(x)


def target_rsqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.rsqrt(x)


TARGETS: dict[str, TargetSpec] = {
    "gelu": TargetSpec(
        name="gelu",
        description="GELU direct approximation",
        target_fn=target_gelu,
        eval_lo=-5.0,
        eval_hi=5.0,
        n_sign="random",
        b_sign="random",
    ),
    "exp": TargetSpec(
        name="exp",
        description="Softmax exponent sub-function",
        target_fn=target_exp,
        eval_lo=-256.0,
        eval_hi=0.0,
        n_sign="positive",
        b_sign="positive",
    ),
    "divide": TargetSpec(
        name="divide",
        description="Softmax reciprocal sub-function",
        target_fn=target_divide,
        eval_lo=1.0,
        eval_hi=1024.0,
        n_sign="negative",
        b_sign="positive",
    ),
    "rsqrt": TargetSpec(
        name="rsqrt",
        description="LayerNorm reciprocal square-root sub-function",
        target_fn=target_rsqrt,
        eval_lo=0.1,
        eval_hi=1024.0,
        n_sign="negative",
        b_sign="positive",
        scaling=ScalingSpec(
            enabled=True,
            threshold=1.0,
            factor=1024.0,
            train_lo=1.0,
            train_hi=1024.0,
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export a paper-style NN-LUT scalar approximator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        choices=["all", *TARGETS.keys()],
        default="all",
        help="Target scalar function to fit.",
    )
    parser.add_argument(
        "--lut-entries",
        type=int,
        default=16,
        help="Number of LUT entries (hidden width is entries - 1).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100_000,
        help="Number of uniformly spaced training samples.",
    )
    parser.add_argument(
        "--eval-points",
        type=int,
        default=32_768,
        help="Number of dense evaluation points.",
    )
    parser.add_argument("--epochs", type=int, default=3000, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--milestones",
        default="0.5,0.75,0.9",
        help="Comma-separated MultiStepLR milestones. Values in (0, 1) are treated as epoch fractions.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="Learning-rate decay factor for MultiStepLR.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=500,
        help="Stop if the best MAE does not improve for this many epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-8,
        help="Minimum MAE improvement required to reset the early stopper.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to train/baselines/runs/<timestamp>-nnlut-e<entries>.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional suffix for the output folder name.",
    )
    parser.add_argument(
        "--disable-rsqrt-scaling",
        action="store_true",
        help="Disable the paper's input scaling trick for rsqrt.",
    )
    parser.add_argument(
        "--linear-fit-points",
        type=int,
        default=2048,
        help="Dense grid size used to fit the uniform linear-LUT baseline.",
    )
    args = parser.parse_args()

    if args.lut_entries < 2:
        raise ValueError("--lut-entries must be at least 2.")
    if args.num_samples < args.lut_entries:
        raise ValueError("--num-samples must be >= --lut-entries.")
    if args.eval_points <= 1:
        raise ValueError("--eval-points must be greater than 1.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if not 0.0 < args.gamma < 1.0:
        raise ValueError("--gamma must be in (0, 1).")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be non-negative.")
    if args.linear_fit_points < args.lut_entries:
        raise ValueError("--linear-fit-points must be >= --lut-entries.")

    return args


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        name = f"{timestamp}-nnlut-e{args.lut_entries}"
        if args.run_name:
            name = f"{name}-{args.run_name}"
        output_dir = Path(__file__).resolve().parent / "runs" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_milestones(raw: str, epochs: int) -> list[int]:
    milestones: list[int] = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        value = float(item)
        if 0.0 < value < 1.0:
            milestone = int(round(value * epochs))
        else:
            milestone = int(round(value))
        if 0 < milestone < epochs:
            milestones.append(milestone)
    return sorted(set(milestones))


class ScalarNnLut(nn.Module):
    def __init__(
        self,
        *,
        hidden_units: int,
        domain_lo: float,
        domain_hi: float,
        n_sign: str,
        b_sign: str,
        target_fn: TensorFn,
        init_points: int = 4096,
    ) -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.domain_lo = float(domain_lo)
        self.domain_hi = float(domain_hi)
        self.n_sign = n_sign
        self.b_sign = b_sign

        self.m = nn.Parameter(torch.empty(hidden_units))
        self.raw_n = nn.Parameter(torch.empty(hidden_units))
        self.raw_b = nn.Parameter(torch.empty(hidden_units))
        self.register_buffer("sign_template", torch.empty(hidden_units))

        self.reset_parameters(target_fn=target_fn, init_points=init_points)

    def reset_parameters(self, *, target_fn: TensorFn, init_points: int) -> None:
        width = max(self.domain_hi - self.domain_lo, 1e-6)
        breakpoints = torch.linspace(
            self.domain_lo, self.domain_hi, self.hidden_units + 2, dtype=torch.float32
        )[1:-1]
        base_mag = self.hidden_units / width
        magnitude = base_mag * (0.75 + 0.5 * torch.rand(self.hidden_units))

        if self.n_sign == "positive":
            sign = torch.ones(self.hidden_units, dtype=torch.float32)
        elif self.n_sign == "negative":
            sign = -torch.ones(self.hidden_units, dtype=torch.float32)
        else:
            sign = torch.where(
                torch.rand(self.hidden_units) >= 0.5,
                torch.ones(self.hidden_units),
                -torch.ones(self.hidden_units),
            ).to(torch.float32)
        self.sign_template.copy_(sign)

        n0 = sign * magnitude
        b0 = -n0 * breakpoints
        if self.b_sign == "positive":
            b0 = b0.abs().clamp_min(1e-5)

        if self.n_sign == "random":
            self.raw_n.data.copy_(n0)
        else:
            self.raw_n.data.copy_(inverse_softplus(n0.abs().clamp_min(1e-5)))
        if self.b_sign == "positive":
            self.raw_b.data.copy_(inverse_softplus(b0.clamp_min(1e-5)))
        else:
            self.raw_b.data.copy_(b0)

        x_init = torch.linspace(
            self.domain_lo,
            self.domain_hi,
            max(init_points, self.hidden_units * 64),
            dtype=torch.float32,
        )
        with torch.inference_mode():
            features = torch.relu(x_init[:, None] * n0[None, :] + b0[None, :])
            y_init = target_fn(x_init)
            solution = torch.linalg.lstsq(features, y_init[:, None]).solution.squeeze(-1)
        self.m.data.copy_(solution.to(dtype=torch.float32))

    def effective_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.n_sign == "random":
            n = self.raw_n
        else:
            n_mag = F.softplus(self.raw_n) + 1e-6
            n = self.sign_template * n_mag
        if self.b_sign == "positive":
            b = F.softplus(self.raw_b) + 1e-6
        else:
            b = self.raw_b
        return self.m, n, b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m, n, b = self.effective_parameters()
        return torch.relu(x[..., None] * n + b).matmul(m)


def get_scaling(spec: TargetSpec, args: argparse.Namespace) -> ScalingSpec:
    if spec.name == "rsqrt" and args.disable_rsqrt_scaling:
        return ScalingSpec(enabled=False)
    return spec.scaling


def get_training_range(spec: TargetSpec, scaling: ScalingSpec) -> tuple[float, float]:
    if scaling.enabled and scaling.train_lo is not None and scaling.train_hi is not None:
        return scaling.train_lo, scaling.train_hi
    return spec.eval_lo, spec.eval_hi


def make_uniform_samples(lo: float, hi: float, count: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(lo, hi, count, dtype=torch.float32, device=device)


def apply_runtime_scaling(x: torch.Tensor, scaling: ScalingSpec) -> tuple[torch.Tensor, torch.Tensor]:
    if not scaling.enabled:
        return x, torch.ones_like(x)

    scaled_x = x.clone()
    output_scale = torch.ones_like(x)
    mask = x < scaling.threshold
    if mask.any():
        scaled_x[mask] = x[mask] * scaling.factor
        output_scale[mask] = math.sqrt(scaling.factor)
    return scaled_x, output_scale


def choose_probe(
    left: float,
    right: float,
    domain_lo: float,
    domain_hi: float,
) -> float:
    if math.isfinite(left) and math.isfinite(right):
        return 0.5 * (left + right)
    if not math.isfinite(left):
        return min(domain_lo, right - max(1.0, abs(right) * 0.01 + 1.0))
    return max(domain_hi, left + max(1.0, abs(left) * 0.01 + 1.0))


def convert_network_to_lut(
    model: ScalarNnLut,
    *,
    domain_lo: float,
    domain_hi: float,
) -> LutTable:
    with torch.inference_mode():
        m, n, b = model.effective_parameters()
        n_sign = torch.where(n >= 0, torch.ones_like(n), -torch.ones_like(n))
        safe_n = torch.where(n.abs() < 1e-6, n_sign * 1e-6, n)
        breakpoints = -b / safe_n
        order = torch.argsort(breakpoints)
        sorted_breakpoints = breakpoints[order].detach().cpu()
        edges = [-math.inf, *sorted_breakpoints.tolist(), math.inf]

        slopes: list[float] = []
        intercepts: list[float] = []
        for index in range(len(edges) - 1):
            probe = choose_probe(edges[index], edges[index + 1], domain_lo, domain_hi)
            probe_tensor = torch.tensor(probe, dtype=m.dtype, device=m.device)
            active = (n * probe_tensor + b) > 0
            slope = torch.sum(m[active] * n[active]).item()
            intercept = torch.sum(m[active] * b[active]).item()
            slopes.append(slope)
            intercepts.append(intercept)

    return LutTable(
        breakpoints=sorted_breakpoints.to(torch.float32),
        slopes=torch.tensor(slopes, dtype=torch.float32),
        intercepts=torch.tensor(intercepts, dtype=torch.float32),
    )


def evaluate_lut(x: torch.Tensor, lut: LutTable) -> torch.Tensor:
    device = x.device
    breakpoints = lut.breakpoints.to(device=device, dtype=x.dtype)
    slopes = lut.slopes.to(device=device, dtype=x.dtype)
    intercepts = lut.intercepts.to(device=device, dtype=x.dtype)
    indices = torch.bucketize(x, breakpoints)
    return slopes[indices] * x + intercepts[indices]


def evaluate_runtime_lut(
    x: torch.Tensor,
    *,
    lut: LutTable,
    scaling: ScalingSpec,
) -> torch.Tensor:
    scaled_x, output_scale = apply_runtime_scaling(x, scaling)
    return evaluate_lut(scaled_x, lut) * output_scale


def fit_uniform_linear_lut(
    spec: TargetSpec,
    *,
    train_lo: float,
    train_hi: float,
    lut_entries: int,
    fit_points: int,
    device: torch.device,
) -> LutTable:
    endpoints = torch.linspace(train_lo, train_hi, lut_entries + 1, dtype=torch.float32)
    breakpoints = endpoints[1:-1].clone()
    slopes: list[float] = []
    intercepts: list[float] = []
    points_per_segment = max(16, fit_points // lut_entries)

    for index in range(lut_entries):
        left = float(endpoints[index].item())
        right = float(endpoints[index + 1].item())
        x_seg = torch.linspace(left, right, points_per_segment, dtype=torch.float32, device=device)
        y_seg = spec.target_fn(x_seg)
        design = torch.stack([x_seg, torch.ones_like(x_seg)], dim=1)
        coeffs = torch.linalg.lstsq(design, y_seg[:, None]).solution.squeeze(-1)
        slopes.append(float(coeffs[0].item()))
        intercepts.append(float(coeffs[1].item()))

    return LutTable(
        breakpoints=breakpoints,
        slopes=torch.tensor(slopes, dtype=torch.float32),
        intercepts=torch.tensor(intercepts, dtype=torch.float32),
    )


def compute_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    abs_error = error.abs()
    mse = torch.mean(error.square()).item()
    mae = torch.mean(abs_error).item()
    rmse = math.sqrt(mse)
    p95 = torch.quantile(abs_error, 0.95).item()
    p99 = torch.quantile(abs_error, 0.99).item()
    max_abs = torch.max(abs_error).item()
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "p95_abs_error": float(p95),
        "p99_abs_error": float(p99),
        "max_abs_error": float(max_abs),
    }


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_lut_csv(
    path: Path,
    *,
    lut: LutTable,
) -> None:
    breakpoints = lut.breakpoints.tolist()
    slopes = lut.slopes.tolist()
    intercepts = lut.intercepts.tolist()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment", "left", "right", "slope", "intercept"])
        for index in range(len(slopes)):
            left = "-inf" if index == 0 else f"{breakpoints[index - 1]:.12g}"
            right = "inf" if index == len(slopes) - 1 else f"{breakpoints[index]:.12g}"
            writer.writerow([index, left, right, f"{slopes[index]:.12g}", f"{intercepts[index]:.12g}"])


def plot_fit(
    path: Path,
    *,
    x: np.ndarray,
    target: np.ndarray,
    nnlut: np.ndarray,
    linear: np.ndarray,
    title: str,
) -> None:
    abs_err_nnlut = np.abs(nnlut - target)
    abs_err_linear = np.abs(linear - target)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    axes[0].plot(x, target, label="Target", color="#111827", linewidth=2.0)
    axes[0].plot(x, nnlut, label="NN-LUT", color="#2563eb", linewidth=1.5)
    axes[0].plot(x, linear, label="Linear-LUT", color="#dc2626", linewidth=1.3, alpha=0.9)
    axes[0].set_ylabel("Output")
    axes[0].set_title(title)
    axes[0].legend(loc="best")

    axes[1].plot(x, abs_err_nnlut, label="NN-LUT abs error", color="#2563eb", linewidth=1.5)
    axes[1].plot(x, abs_err_linear, label="Linear-LUT abs error", color="#dc2626", linewidth=1.3, alpha=0.9)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Input")
    axes[1].set_ylabel("Absolute error")
    axes[1].legend(loc="best")

    fig.savefig(path, dpi=220)
    plt.close(fig)


def train_one_target(
    spec: TargetSpec,
    *,
    args: argparse.Namespace,
    output_root: Path,
    device: torch.device,
) -> dict:
    scaling = get_scaling(spec, args)
    train_lo, train_hi = get_training_range(spec, scaling)
    hidden_units = args.lut_entries - 1
    target_dir = output_root / spec.name
    target_dir.mkdir(parents=True, exist_ok=True)

    x_train = make_uniform_samples(train_lo, train_hi, args.num_samples, device)
    y_train = spec.target_fn(x_train)
    x_eval = make_uniform_samples(spec.eval_lo, spec.eval_hi, args.eval_points, device)
    y_eval = spec.target_fn(x_eval)
    x_conversion = make_uniform_samples(train_lo, train_hi, max(8192, args.eval_points // 2), device)

    model = ScalarNnLut(
        hidden_units=hidden_units,
        domain_lo=train_lo,
        domain_hi=train_hi,
        n_sign=spec.n_sign,
        b_sign=spec.b_sign,
        target_fn=spec.target_fn,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestones = parse_milestones(args.milestones, args.epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.gamma
    )

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = math.inf
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = F.l1_loss(prediction, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            lut = convert_network_to_lut(model, domain_lo=train_lo, domain_hi=train_hi)
            nnlut_eval = evaluate_runtime_lut(x_eval, lut=lut, scaling=scaling)
            metrics = compute_metrics(y_eval, nnlut_eval)
            conversion_network = model(x_conversion)
            conversion_lut = evaluate_lut(x_conversion, lut)
            conversion_error = torch.max((conversion_network - conversion_lut).abs()).item()

        record = {
            "epoch": epoch,
            "train_l1": float(loss.item()),
            "eval_mae": metrics["mae"],
            "eval_rmse": metrics["rmse"],
            "eval_max_abs": metrics["max_abs_error"],
            "conversion_max_abs": float(conversion_error),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        if metrics["mae"] + args.early_stop_min_delta < best_mae:
            best_mae = metrics["mae"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            stale_epochs += 1

        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.inference_mode():
        best_lut = convert_network_to_lut(model, domain_lo=train_lo, domain_hi=train_hi)
        nnlut_eval = evaluate_runtime_lut(x_eval, lut=best_lut, scaling=scaling)
        linear_lut = fit_uniform_linear_lut(
            spec,
            train_lo=train_lo,
            train_hi=train_hi,
            lut_entries=args.lut_entries,
            fit_points=args.linear_fit_points,
            device=device,
        )
        linear_eval = evaluate_runtime_lut(x_eval, lut=linear_lut, scaling=scaling)
        final_metrics = compute_metrics(y_eval, nnlut_eval)
        linear_metrics = compute_metrics(y_eval, linear_eval)

        conversion_network = model(x_conversion)
        conversion_lut = evaluate_lut(x_conversion, best_lut)
        conversion_metrics = compute_metrics(conversion_network, conversion_lut)
        m, n, b = model.effective_parameters()

    np.savez(
        target_dir / "lut_fp32.npz",
        breakpoints=best_lut.breakpoints.cpu().numpy(),
        slopes=best_lut.slopes.cpu().numpy(),
        intercepts=best_lut.intercepts.cpu().numpy(),
    )
    np.savez(
        target_dir / "lut_fp16.npz",
        breakpoints=best_lut.breakpoints.cpu().numpy().astype(np.float16),
        slopes=best_lut.slopes.cpu().numpy().astype(np.float16),
        intercepts=best_lut.intercepts.cpu().numpy().astype(np.float16),
    )
    save_lut_csv(target_dir / "lut_fp32.csv", lut=best_lut)

    save_json(
        target_dir / "lut_fp32.json",
        {
            "target": spec.name,
            "description": spec.description,
            "lut_entries": args.lut_entries,
            "hidden_units": hidden_units,
            "eval_range": [spec.eval_lo, spec.eval_hi],
            "train_range": [train_lo, train_hi],
            "scaling": asdict(scaling),
            "breakpoints": best_lut.breakpoints.tolist(),
            "slopes": best_lut.slopes.tolist(),
            "intercepts": best_lut.intercepts.tolist(),
            "network_parameters": {
                "m": m.detach().cpu().tolist(),
                "n": n.detach().cpu().tolist(),
                "b": b.detach().cpu().tolist(),
            },
        },
    )

    with (target_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plot_fit(
        target_dir / "fit.png",
        x=x_eval.detach().cpu().numpy(),
        target=y_eval.detach().cpu().numpy(),
        nnlut=nnlut_eval.detach().cpu().numpy(),
        linear=linear_eval.detach().cpu().numpy(),
        title=f"{spec.name} approximation ({args.lut_entries} entries)",
    )

    result = {
        "target": spec.name,
        "description": spec.description,
        "lut_entries": args.lut_entries,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "train_range": [train_lo, train_hi],
        "eval_range": [spec.eval_lo, spec.eval_hi],
        "scaling": asdict(scaling),
        "nnlut_metrics": final_metrics,
        "linear_lut_metrics": linear_metrics,
        "conversion_metrics": conversion_metrics,
        "artifacts": {
            "lut_json": str(target_dir / "lut_fp32.json"),
            "lut_csv": str(target_dir / "lut_fp32.csv"),
            "lut_fp32_npz": str(target_dir / "lut_fp32.npz"),
            "lut_fp16_npz": str(target_dir / "lut_fp16.npz"),
            "fit_plot": str(target_dir / "fit.png"),
            "history_csv": str(target_dir / "history.csv"),
        },
    }
    save_json(target_dir / "summary.json", result)
    return result


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = select_device(args.device)
    output_dir = build_output_dir(args)

    targets = list(TARGETS.values()) if args.target == "all" else [TARGETS[args.target]]
    config = {
        "target": args.target,
        "lut_entries": args.lut_entries,
        "num_samples": args.num_samples,
        "eval_points": args.eval_points,
        "epochs": args.epochs,
        "lr": args.lr,
        "milestones": parse_milestones(args.milestones, args.epochs),
        "gamma": args.gamma,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "seed": args.seed,
        "device": str(device),
        "disable_rsqrt_scaling": args.disable_rsqrt_scaling,
        "linear_fit_points": args.linear_fit_points,
        "targets": [spec.name for spec in targets],
    }
    save_json(output_dir / "config.json", config)

    results = [
        train_one_target(spec, args=args, output_root=output_dir, device=device)
        for spec in targets
    ]
    save_json(output_dir / "summary.json", {"results": results})
    print(json.dumps({"output_dir": str(output_dir), "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
