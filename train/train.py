from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import FuncFormatter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    target_fn: Callable[[torch.Tensor], torch.Tensor]
    default_l_range: float
    default_r_range: float
    description: str


def target_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


def target_exp2(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(x)


def target_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def target_erf(x: torch.Tensor) -> torch.Tensor:
    return torch.erf(x) + 1


def target_rsqrt(x: torch.Tensor) -> torch.Tensor:
    # Rsqrt is trained on a negative-domain parameterization, so the real-valued
    # remap is 1 / sqrt(-x), corresponding to reciprocal square root on -x.
    return torch.rsqrt(-x)


def target_recip(x: torch.Tensor) -> torch.Tensor:
    return -torch.reciprocal(x)


def target_sin(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + 1


def target_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x) + 1


def target_softsign(x: torch.Tensor) -> torch.Tensor:
    return F.softsign(x) + 1


def target_arctan(x: torch.Tensor) -> torch.Tensor:
    return torch.atan(x) + x.new_tensor(math.pi / 2)


def target_softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def target_gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.erf(x / math.sqrt(2.0)) + 1


FUNCTION_SPECS: dict[str, FunctionSpec] = {
    "exp": FunctionSpec(
        name="exp",
        target_fn=target_exp,
        default_l_range=-256.0,
        default_r_range=0.0,
        description="exp(x)",
    ),
    "exp2": FunctionSpec(
        name="exp2",
        target_fn=target_exp2,
        default_l_range=-256.0,
        default_r_range=0.0,
        description="2^x",
    ),
    "sigmoid": FunctionSpec(
        name="sigmoid",
        target_fn=target_sigmoid,
        default_l_range=-6.234,
        default_r_range=0.0,
        description="sigmoid(x)",
    ),
    "erf": FunctionSpec(
        name="erf",
        target_fn=target_erf,
        default_l_range=-2.469,
        default_r_range=0.0,
        description="erf(x) + 1 remapping from functions.md",
    ),
    "rsqrt": FunctionSpec(
        name="rsqrt",
        target_fn=target_rsqrt,
        default_l_range=-256.0,
        default_r_range=-1.0,
        description="1 / sqrt(-x) remapping from the negative-domain rsqrt parameterization",
    ),
    "recip": FunctionSpec(
        name="recip",
        target_fn=target_recip,
        default_l_range=-16.0,
        default_r_range=-1.0,
        description="-(1 / x) remapping from the negative-domain reciprocal parameterization",
    ),
    "sin": FunctionSpec(
        name="sin",
        target_fn=target_sin,
        default_l_range=-math.pi / 2,
        default_r_range=0.0,
        description="sin(x) + 1 remapping from functions.md",
    ),
    "tanh": FunctionSpec(
        name="tanh",
        target_fn=target_tanh,
        default_l_range=-3.465,
        default_r_range=0.0,
        description="tanh(x) + 1 remapping from functions.md",
    ),
    "softsign": FunctionSpec(
        name="softsign",
        target_fn=target_softsign,
        default_l_range=-128.0,
        default_r_range=0.0,
        description="softsign(x) + 1 remapping from functions.md",
    ),
    "arctan": FunctionSpec(
        name="arctan",
        target_fn=target_arctan,
        default_l_range=-128.0,
        default_r_range=0.0,
        description="arctan(x) + pi/2 remapping from functions.md",
    ),
    "softplus": FunctionSpec(
        name="softplus",
        target_fn=target_softplus,
        default_l_range=-16.0,
        default_r_range=0.0,
        description="softplus(x)",
    ),
    "gelu": FunctionSpec(
        name="gelu",
        target_fn=target_gelu,
        default_l_range=-8.0,
        default_r_range=0.0,
        description="erf(x / sqrt(2)) + 1 branch used by GeLU",
    ),
}


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().tolist()


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))


def geomspace(
    start: float,
    stop: float,
    steps: int,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.exp(
        torch.linspace(math.log(start), math.log(stop), steps, dtype=dtype, device=device)
    )


class Approx(nn.Module):
    def __init__(self, num_units: int, reparam: str | None):
        super().__init__()
        self.num_units = num_units
        self.reparam = reparam
        self.w = nn.Parameter(torch.randn(num_units))
        self.k = nn.Parameter(torch.randn(num_units))
        self.b = nn.Parameter(torch.randn(num_units))

    def __str__(self) -> str:
        return f"Approx(num_units={self.num_units}, reparam={self.reparam or 'none'})"

    def effective_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reparam == "exp":
            w = torch.exp(self.w)
            k = torch.exp(self.k)
            b = torch.exp(self.b)
        elif self.reparam == "softplus":
            w = F.softplus(self.w)
            k = F.softplus(self.k)
            b = F.softplus(self.b)
        else:
            w = self.w
            k = self.k
            b = self.b

        return w, k, b

    def export_parameters(self) -> dict[str, Any]:
        w, k, b = self.effective_parameters()
        k_positive = bool(torch.all(k > 0).item())

        payload: dict[str, Any] = {
            "raw": {
                "w": tensor_to_list(self.w),
                "k": tensor_to_list(self.k),
                "b": tensor_to_list(self.b),
            },
            "effective": {
                "w": tensor_to_list(w),
                "k": tensor_to_list(k),
                "b": tensor_to_list(b),
            },
            "properties": {
                "w_positive": bool(torch.all(w > 0).item()),
                "k_positive": k_positive,
                "b_positive": bool(torch.all(b > 0).item()),
                "relu_scale_fusable": k_positive,
            },
        }
        if k_positive:
            payload["fused"] = {
                "wk": tensor_to_list(w * k),
                "bk": tensor_to_list(b * k),
            }
        else:
            payload["fused"] = None

        return payload

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, k, b = self.effective_parameters()
        return (torch.relu(x[..., None] * w + b) * k).sum(dim=-1)


def maybe_initialize_model_from_function(
    model: Approx,
    *,
    func_name: str,
    reparam: str | None,
    l_range: float,
    r_range: float,
) -> dict[str, Any] | None:
    # This only seeds positive parameter scales and hinge locations from the
    # input domain; it does not solve coefficients from target samples.
    if reparam not in {"exp", "softplus"}:
        return None
    if func_name not in {"rsqrt", "recip"}:
        return None
    if not (l_range < r_range < 0.0):
        return None

    raw_seed_w = torch.empty_like(model.w).uniform_(-5.0, 0.0)
    raw_seed_k = torch.empty_like(model.k).uniform_(-5.0, 0.0)
    raw_seed_b = torch.empty_like(model.b).uniform_(-5.0, 0.0)

    effective_w = torch.exp(raw_seed_w)
    effective_k = torch.exp(raw_seed_k)
    effective_b = torch.exp(raw_seed_b)
    metadata: dict[str, Any] = {
        "strategy": "bounded effective-parameter init",
        "raw_init_range": [-5.0, 0.0],
    }

    if func_name == "recip":
        left_magnitude = abs(l_range) * 2.0
        right_magnitude = max(abs(r_range) * 0.5, 1e-6)
        magnitudes = geomspace(
            right_magnitude,
            left_magnitude,
            model.num_units,
            dtype=model.w.dtype,
            device=model.w.device,
        )
        breakpoints = -torch.flip(magnitudes, dims=[0])
        effective_b = (-breakpoints * effective_w).clamp_min(1e-12)
        metadata.update(
            {
                "strategy": "bounded effective-parameter init + geometric breakpoint seeding",
                "breakpoints": tensor_to_list(breakpoints),
                "expanded_domain": [float(-left_magnitude), float(-right_magnitude)],
            }
        )

    if reparam == "exp":
        raw_w = torch.log(effective_w)
        raw_k = torch.log(effective_k)
        raw_b = torch.log(effective_b)
    elif reparam == "softplus":
        raw_w = inverse_softplus(effective_w)
        raw_k = inverse_softplus(effective_k)
        raw_b = inverse_softplus(effective_b)
    else:
        return None

    with torch.no_grad():
        model.w.copy_(raw_w.to(device=model.w.device, dtype=model.w.dtype))
        model.k.copy_(raw_k.to(device=model.k.device, dtype=model.k.dtype))
        model.b.copy_(raw_b.to(device=model.b.device, dtype=model.b.dtype))

    return metadata


class Loss(nn.Module):
    def __init__(self, y_min: float, y_max: float, l_bound: float):
        super().__init__()
        self.register_buffer("y_min", torch.tensor(y_min, dtype=torch.float32))
        self.register_buffer("y_max", torch.tensor(y_max, dtype=torch.float32))
        self.l_bound = l_bound

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(predictions, targets)
        out_of_bounds = torch.relu(self.y_min - predictions) + torch.relu(
            predictions - self.y_max
        )
        bound_loss = F.mse_loss(out_of_bounds, torch.zeros_like(out_of_bounds))
        return (1 - 2 * self.l_bound) * mse_loss + self.l_bound * bound_loss


@dataclass
class EarlyStopper:
    patience: int
    min_delta: float = 0.0
    best_value: float = math.inf
    best_epoch: int = 0
    stale_epochs: int = 0

    def step(self, value: float, epoch: int) -> bool:
        if self.best_value - value > self.min_delta:
            self.best_value = value
            self.best_epoch = epoch
            self.stale_epochs = 0
            return False

        self.stale_epochs += 1
        return self.stale_epochs >= self.patience


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the piecewise-ReLU approximator from train.ipynb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--func",
        choices=[
            "exp",
            "exp2",
            "sigmoid",
            "erf",
            "rsqrt",
            "recip",
            "sin",
            "tanh",
            "softsign",
            "arctan",
            "softplus",
            "gelu",
        ],
        default="exp",
        help="Target function to approximate.",
    )
    parser.add_argument(
        "--l-range",
        type=float,
        default=None,
        help="Left edge of the training interval. Defaults from --func.",
    )
    parser.add_argument(
        "--r-range",
        type=float,
        default=None,
        help="Right edge of the training interval. Defaults from --func.",
    )
    parser.add_argument(
        "--num-units",
        type=int,
        default=16,
        help="Number of ReLU units in the approximator.",
    )
    parser.add_argument(
        "--reparam",
        choices=["none", "exp", "softplus"],
        default="exp",
        help="Parameter reparameterization used to constrain model weights.",
    )
    parser.add_argument(
        "--l-bound",
        type=float,
        default=0.15,
        help="Boundary loss weight. Must be in [0, 0.5) so the MSE term stays non-negative.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Lower output bound. Defaults from --func.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Upper output bound. Defaults from --func.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Training batch size."
    )
    parser.add_argument(
        "--optim", choices=["Adam", "AdamW", "SGD"], default="Adam", help="Optimizer."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "ReduceLROnPlateau"],
        default="none",
        help="Per-epoch learning-rate scheduler.",
    )
    parser.add_argument(
        "--lr-scheduler-metric",
        choices=["avg_loss", "mse"],
        default="mse",
        help="Metric monitored by --lr-scheduler.",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=1000,
        help="Number of bad epochs tolerated by --lr-scheduler before reducing the learning rate.",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="Multiplicative decay factor used by --lr-scheduler.",
    )
    parser.add_argument(
        "--lr-scheduler-threshold",
        type=float,
        default=1e-6,
        help="Minimum change in --lr-scheduler-metric to qualify as an improvement.",
    )
    parser.add_argument(
        "--lr-scheduler-threshold-mode",
        choices=["rel", "abs"],
        default="rel",
        help="Interpretation mode for --lr-scheduler-threshold.",
    )
    parser.add_argument(
        "--lr-scheduler-cooldown",
        type=int,
        default=500,
        help="Cooldown epochs after a learning-rate reduction before bad epochs start accumulating again.",
    )
    parser.add_argument(
        "--lr-scheduler-min-lr",
        type=float,
        default=1e-4,
        help="Lower bound enforced by --lr-scheduler.",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1000, help="Number of training epochs."
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=["avg_loss", "mse"],
        default="mse",
        help="Metric monitored by early stopping.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=100,
        help="Stop after this many epochs without --early-stop-metric improving by at least --early-stop-min-delta. Set to 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-8,
        help="Minimum monitored-metric improvement required to reset the early stopper.",
    )
    parser.add_argument(
        "--train-data-shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        default=(2048, 1024),
        help="Shape of the randomly generated training tensor.",
    )
    parser.add_argument(
        "--eval-points",
        type=int,
        default=1000,
        help="Number of points in the evaluation grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization and data generation.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Execution device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--pin-memory", action="store_true", help="Enable DataLoader pin_memory."
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional human-readable suffix for the output folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints and plots. Defaults to train/runs/<timestamp>-<config>.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard event files under the run directory.",
    )
    args = parser.parse_args()
    apply_function_defaults(args)
    validate_args(args)
    validate_reparam_compatibility(args)
    return args


def get_function_spec(name: str) -> FunctionSpec:
    try:
        return FUNCTION_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported target function: {name}") from exc


def apply_function_defaults(args: argparse.Namespace) -> None:
    spec = get_function_spec(args.func)
    if args.l_range is None:
        args.l_range = spec.default_l_range
    if args.r_range is None:
        args.r_range = spec.default_r_range


def validate_args(args: argparse.Namespace) -> None:
    if args.num_units <= 0:
        raise ValueError("--num-units must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.lr_scheduler_patience < 0:
        raise ValueError("--lr-scheduler-patience must be non-negative.")
    if not 0.0 < args.lr_scheduler_factor < 1.0:
        raise ValueError("--lr-scheduler-factor must be in (0, 1).")
    if args.lr_scheduler_threshold < 0:
        raise ValueError("--lr-scheduler-threshold must be non-negative.")
    if args.lr_scheduler_cooldown < 0:
        raise ValueError("--lr-scheduler-cooldown must be non-negative.")
    if args.lr_scheduler_min_lr < 0:
        raise ValueError("--lr-scheduler-min-lr must be non-negative.")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be non-negative.")
    if args.eval_points <= 1:
        raise ValueError("--eval-points must be greater than 1.")
    if args.l_range >= args.r_range:
        raise ValueError("--l-range must be smaller than --r-range.")
    if len(args.train_data_shape) != 2 or any(
        dim <= 0 for dim in args.train_data_shape
    ):
        raise ValueError("--train-data-shape requires two positive integers.")
    if not 0.0 <= args.l_bound < 0.5:
        raise ValueError("--l-bound must be in [0, 0.5).")
    if args.y_min is not None and args.y_max is not None and args.y_min > args.y_max:
        raise ValueError("--y-min must be smaller than or equal to --y-max.")


def validate_reparam_compatibility(args: argparse.Namespace) -> None:
    if args.reparam == "none":
        return

    target_fn = get_target_function(args.func)
    sample_x = torch.linspace(args.l_range, args.r_range, 4097, dtype=torch.float64)
    with torch.inference_mode():
        sample_y = target_fn(sample_x)

    if sample_y.ndim != 1:
        sample_y = sample_y.reshape(-1)
    if not torch.isfinite(sample_y).all():
        raise ValueError(
            "Sampled target values are non-finite. Check the selected function and input range."
        )

    delta_x = sample_x[1:] - sample_x[:-1]
    slopes = (sample_y[1:] - sample_y[:-1]) / delta_x
    curvature = (slopes[1:] - slopes[:-1]) / ((delta_x[1:] + delta_x[:-1]) / 2.0)

    min_value = float(sample_y.min().item())
    min_slope = float(slopes.min().item())
    min_curvature = float(curvature.min().item())
    tolerance = 1e-7

    violations: list[str] = []
    if min_value < -tolerance:
        violations.append(f"non-negative (min value {min_value:.6g})")
    if min_slope < -tolerance:
        violations.append(f"monotone increasing (min slope {min_slope:.6g})")
    if min_curvature < -tolerance:
        violations.append(f"convex (min curvature {min_curvature:.6g})")

    if violations:
        violation_text = ", ".join(violations)
        raise ValueError(
            f"--reparam {args.reparam} constrains the approximator to a non-negative, "
            f"monotone-increasing, convex function on [{args.l_range:g}, {args.r_range:g}], "
            f"but the sampled target violates: {violation_text}. "
            "Use --reparam none or remap the target/range."
        )


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_target_function(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return get_function_spec(name).target_fn


def infer_output_bounds(
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    l_range: float,
    r_range: float,
) -> tuple[float, float]:
    with torch.inference_mode():
        endpoints = torch.tensor([l_range, r_range], dtype=torch.float64)
        outputs = target_fn(endpoints)
    if not torch.isfinite(outputs).all():
        raise ValueError(
            "Resolved output bounds are non-finite. Check the selected function and input range."
        )
    return float(outputs.min().item()), float(outputs.max().item())


def resolve_bounds(args: argparse.Namespace) -> tuple[float, float]:
    default_y_min, default_y_max = infer_output_bounds(
        get_target_function(args.func), args.l_range, args.r_range
    )
    y_min = default_y_min if args.y_min is None else args.y_min
    y_max = default_y_max if args.y_max is None else args.y_max

    if y_min > y_max:
        raise ValueError("Resolved output bounds are invalid.")

    return float(y_min), float(y_max)


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        parts = [
            timestamp,
            args.func,
            f"n{args.num_units}",
            args.reparam,
            args.optim.lower(),
            f"lr{args.lr:g}",
        ]
        if args.lr_scheduler == "ReduceLROnPlateau":
            parts.append("plateau")
        if args.run_name:
            parts.append(args.run_name)
        output_dir = Path(__file__).resolve().parent / "runs" / "-".join(parts)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_optimizer(name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def build_lr_scheduler(
    args: argparse.Namespace, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            threshold=args.lr_scheduler_threshold,
            threshold_mode=args.lr_scheduler_threshold_mode,
            cooldown=args.lr_scheduler_cooldown,
            min_lr=args.lr_scheduler_min_lr,
        )
    raise ValueError(f"Unknown lr scheduler: {args.lr_scheduler}")


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def maybe_create_writer(enabled: bool, log_dir: Path):
    if not enabled:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging requested but tensorboard is not installed."
        ) from exc

    return SummaryWriter(log_dir=str(log_dir))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resolve_metric_value(name: str, *, avg_loss: float, mse: float) -> float:
    if name == "avg_loss":
        return avg_loss
    if name == "mse":
        return mse
    raise ValueError(f"Unknown metric: {name}")


def save_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def infer_plot_smooth_span(num_points: int) -> int:
    return max(25, min(401, num_points // 60))


def ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        smoothed[idx] = alpha * values[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def smooth_metric(values: np.ndarray, span: int) -> np.ndarray:
    if np.all(values > 0.0):
        return np.power(10.0, ema(np.log10(values), span))
    return ema(values, span)


def downsample_series(
    epochs: np.ndarray, values: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(epochs) <= max_points:
        return epochs, values
    step = int(math.ceil(len(epochs) / max_points))
    return epochs[::step], values[::step]


def format_epoch_tick(value: float, _: float) -> str:
    return f"{int(value):,}"


def plot_metric_series(
    axis: plt.Axes,
    epochs: np.ndarray,
    raw_series: list[tuple[str, np.ndarray, str]],
    smooth_series_map: list[tuple[str, np.ndarray, str]],
    *,
    ylabel: str,
    best_epoch: int | None,
    raw_max_points: int,
) -> None:
    for label, values, color in raw_series:
        raw_x, raw_y = downsample_series(epochs, values, raw_max_points)
        axis.plot(
            raw_x,
            raw_y,
            color=color,
            alpha=0.18,
            linewidth=0.8,
            label=f"{label} raw",
        )

    for label, values, color in smooth_series_map:
        axis.plot(epochs, values, color=color, linewidth=2.2, label=f"{label} smooth")

    if best_epoch is not None:
        axis.axvline(
            best_epoch,
            color="#111827",
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
        )

    axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    axis.legend(loc="best", fontsize=9)


def plot_history(path: Path, history: list[dict[str, Any]]) -> None:
    epochs = np.array([int(entry["epoch"]) for entry in history], dtype=np.float64)
    avg_loss = np.array([entry["avg_loss"] for entry in history], dtype=np.float64)
    max_loss = np.array([entry["max_loss"] for entry in history], dtype=np.float64)
    mse = np.array([entry["mse"] for entry in history], dtype=np.float64)
    rmse = np.array([entry["rmse"] for entry in history], dtype=np.float64)

    smooth_span = infer_plot_smooth_span(len(epochs))
    best_epoch = int(epochs[int(np.argmin(mse))])
    raw_max_points = 4000

    smoothed_avg_loss = smooth_metric(avg_loss, smooth_span)
    smoothed_max_loss = smooth_metric(max_loss, smooth_span)
    smoothed_mse = smooth_metric(mse, smooth_span)
    smoothed_rmse = smooth_metric(rmse, smooth_span)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)

    plot_metric_series(
        axes[0],
        epochs,
        [
            ("Avg loss", avg_loss, "#2563eb"),
            ("Max loss", max_loss, "#dc2626"),
        ],
        [
            ("Avg loss", smoothed_avg_loss, "#1d4ed8"),
            ("Max loss", smoothed_max_loss, "#b91c1c"),
        ],
        ylabel="Loss",
        best_epoch=best_epoch,
        raw_max_points=raw_max_points,
    )
    plot_metric_series(
        axes[1],
        epochs,
        [
            ("MSE", mse, "#059669"),
            ("RMSE", rmse, "#7c3aed"),
        ],
        [
            ("MSE", smoothed_mse, "#047857"),
            ("RMSE", smoothed_rmse, "#6d28d9"),
        ],
        ylabel="Error",
        best_epoch=best_epoch,
        raw_max_points=raw_max_points,
    )

    axes[0].set_title("Training History")
    axes[1].set_xlabel("Epoch")
    for axis in axes:
        axis.xaxis.set_major_formatter(FuncFormatter(format_epoch_tick))

    fig.suptitle(
        f"Smoothed history (EMA span={smooth_span})",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    *,
    model: Approx,
    epoch: int,
    metrics: dict[str, float],
    best_mse: float,
    args: argparse.Namespace,
    output_bounds: tuple[float, float],
) -> None:
    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "best_mse": best_mse,
        "model": {
            "num_units": model.num_units,
            "reparam": model.reparam,
        },
        "train_args": serialize_args(args),
        "output_bounds": {
            "y_min": output_bounds[0],
            "y_max": output_bounds[1],
        },
        "parameters": model.export_parameters(),
    }
    save_json(path, payload)


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "func": args.func,
        "l_range": args.l_range,
        "r_range": args.r_range,
        "num_units": args.num_units,
        "reparam": None if args.reparam == "none" else args.reparam,
        "l_bound": args.l_bound,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "batch_size": args.batch_size,
        "optim": args.optim,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_metric": args.lr_scheduler_metric,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "lr_scheduler_threshold": args.lr_scheduler_threshold,
        "lr_scheduler_threshold_mode": args.lr_scheduler_threshold_mode,
        "lr_scheduler_cooldown": args.lr_scheduler_cooldown,
        "lr_scheduler_min_lr": args.lr_scheduler_min_lr,
        "num_epochs": args.num_epochs,
        "early_stop_metric": args.early_stop_metric,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "train_data_shape": list(args.train_data_shape),
        "eval_points": args.eval_points,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "run_name": args.run_name,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "tensorboard": args.tensorboard,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)

    device = select_device(args.device)
    output_dir = build_output_dir(args)
    args.output_dir = output_dir
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    target_fn = get_target_function(args.func)
    output_bounds = resolve_bounds(args)
    reparam = None if args.reparam == "none" else args.reparam

    model = Approx(args.num_units, reparam).to(device)
    init_metadata = maybe_initialize_model_from_function(
        model,
        func_name=args.func,
        reparam=reparam,
        l_range=args.l_range,
        r_range=args.r_range,
    )
    loss_fn = Loss(output_bounds[0], output_bounds[1], args.l_bound).to(device)
    optimizer = build_optimizer(args.optim, model, args.lr)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    train_rows, train_cols = args.train_data_shape
    train_x = (
        torch.rand(train_rows, train_cols) * (args.r_range - args.l_range)
        + args.l_range
    )
    train_y = target_fn(train_x)
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        generator=torch.Generator().manual_seed(args.seed),
    )

    writer = maybe_create_writer(args.tensorboard, output_dir / "tensorboard")
    function_spec = get_function_spec(args.func)

    config_payload = {
        **serialize_args(args),
        "resolved_device": str(device),
        "resolved_output_bounds": {
            "y_min": output_bounds[0],
            "y_max": output_bounds[1],
        },
        "initialization": init_metadata,
        "function_description": function_spec.description,
        "model": str(model),
    }
    save_json(output_dir / "config.json", config_payload)
    if writer is not None:
        writer.add_text("config", json.dumps(config_payload, indent=2))

    history: list[dict[str, Any]] = []
    best_mse = float("inf")
    best_epoch = -1
    early_stopper = (
        EarlyStopper(
            patience=args.early_stop_patience, min_delta=args.early_stop_min_delta
        )
        if args.early_stop_patience > 0
        else None
    )
    stopped_early = False
    stop_reason: str | None = None

    eval_x = torch.linspace(
        args.l_range, args.r_range, args.eval_points, device=device
    ).unsqueeze(1)
    eval_y = target_fn(eval_x)
    with torch.inference_mode():
        initial_predictions = model(eval_x)
        initial_loss = loss_fn(initial_predictions, eval_y).item()
        initial_mse = F.mse_loss(initial_predictions, eval_y).item()
        initial_rmse = math.sqrt(initial_mse)
    initial_metrics = {
        "epoch": 0,
        "avg_loss": initial_loss,
        "max_loss": initial_loss,
        "mse": initial_mse,
        "rmse": initial_rmse,
        "lr": get_current_lr(optimizer),
    }
    if early_stopper is not None:
        initial_metrics["early_stop_stale_epochs"] = 0
    best_mse = initial_mse
    best_epoch = 0
    history.append(initial_metrics)
    save_checkpoint(
        checkpoints_dir / "best.json",
        model=model,
        epoch=0,
        metrics=initial_metrics,
        best_mse=best_mse,
        args=args,
        output_bounds=output_bounds,
    )
    if writer is not None:
        writer.add_scalar("Loss/avg", initial_loss, 0)
        writer.add_scalar("Loss/max", initial_loss, 0)
        writer.add_scalar("Error/MSE", initial_mse, 0)
        writer.add_scalar("Error/RMSE", initial_rmse, 0)

    progress = tqdm(range(1, args.num_epochs + 1), desc="Training")
    for epoch in progress:
        model.train()
        total_loss = 0.0
        total_examples = 0
        max_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=args.pin_memory)
            batch_y = batch_y.to(device, non_blocking=args.pin_memory)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_x.size(0)
            loss_value = loss.item()
            total_loss += loss_value * batch_size
            total_examples += batch_size
            max_loss = max(max_loss, loss_value)

        avg_loss = total_loss / total_examples

        model.eval()
        with torch.inference_mode():
            eval_predictions = model(eval_x)
            mse = F.mse_loss(eval_predictions, eval_y).item()
            rmse = math.sqrt(mse)

        early_stop_monitor = resolve_metric_value(
            args.early_stop_metric, avg_loss=avg_loss, mse=mse
        )
        lr_scheduler_monitor = resolve_metric_value(
            args.lr_scheduler_metric, avg_loss=avg_loss, mse=mse
        )
        should_stop = False
        stale_epochs = 0
        if early_stopper is not None:
            should_stop = early_stopper.step(early_stop_monitor, epoch)
            stale_epochs = early_stopper.stale_epochs

        current_lr = get_current_lr(optimizer)

        metrics = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "max_loss": max_loss,
            "mse": mse,
            "rmse": rmse,
            "lr": current_lr,
        }
        if early_stopper is not None:
            metrics["early_stop_stale_epochs"] = stale_epochs

        if lr_scheduler is not None:
            lr_scheduler.step(lr_scheduler_monitor)
            # metrics["lr_next"] = get_current_lr(optimizer)
        history.append(metrics)

        postfix = {
            "loss": f"{avg_loss:.3e}",
            "mse": f"{mse:.3e}",
            "lr": f"{current_lr:.2e}",
        }
        if early_stopper is not None:
            postfix["stale"] = f"{stale_epochs}/{early_stopper.patience}"
        progress.set_postfix(postfix)

        if writer is not None:
            writer.add_scalar("Loss/avg", avg_loss, epoch)
            writer.add_scalar("Loss/max", max_loss, epoch)
            writer.add_scalar("Error/MSE", mse, epoch)
            writer.add_scalar("Error/RMSE", rmse, epoch)
            # writer.add_scalar("Optimizer/lr", current_lr, epoch)
            # if lr_scheduler is not None:
            #     writer.add_scalar("Optimizer/lr_next", metrics["lr_next"], epoch)

        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            save_checkpoint(
                checkpoints_dir / "best.json",
                model=model,
                epoch=epoch,
                metrics=metrics,
                best_mse=best_mse,
                args=args,
                output_bounds=output_bounds,
            )

        if early_stopper is not None and should_stop:
            stopped_early = True
            stop_reason = (
                f"{args.early_stop_metric} did not improve by at least "
                f"{args.early_stop_min_delta:g} for {args.early_stop_patience} epochs"
            )
            progress.write(f"Early stopping at epoch {epoch}: {stop_reason}.")
            break

        if epoch > 0 and epoch % 1000 == 0:
            save_history_csv(output_dir / "history.csv", history)
            plot_history(output_dir / "loss_curve.png", history)

    last_metrics = history[-1]
    save_checkpoint(
        checkpoints_dir / "last.json",
        model=model,
        epoch=int(last_metrics["epoch"]),
        metrics=last_metrics,
        best_mse=best_mse,
        args=args,
        output_bounds=output_bounds,
    )

    save_history_csv(output_dir / "history.csv", history)
    save_json(
        output_dir / "summary.json",
        {
            "best_epoch": best_epoch,
            "best_mse": best_mse,
            "last_epoch": int(last_metrics["epoch"]),
            "last_mse": last_metrics["mse"],
            "last_rmse": last_metrics["rmse"],
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "early_stop": None
            if early_stopper is None
            else {
                "metric": args.early_stop_metric,
                "patience": early_stopper.patience,
                "min_delta": early_stopper.min_delta,
                "best_value": early_stopper.best_value,
                "best_epoch": early_stopper.best_epoch,
                "stale_epochs": early_stopper.stale_epochs,
            },
            "lr_scheduler": None
            if lr_scheduler is None
            else {
                "name": args.lr_scheduler,
                "metric": args.lr_scheduler_metric,
                "patience": args.lr_scheduler_patience,
                "factor": args.lr_scheduler_factor,
                "threshold": args.lr_scheduler_threshold,
                "threshold_mode": args.lr_scheduler_threshold_mode,
                "cooldown": args.lr_scheduler_cooldown,
                "min_lr": args.lr_scheduler_min_lr,
                "last_lr": get_current_lr(optimizer),
            },
        },
    )
    plot_history(output_dir / "loss_curve.png", history)

    if writer is not None:
        writer.close()

    return {
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_mse": best_mse,
        "last_epoch": int(last_metrics["epoch"]),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "last_mse": last_metrics["mse"],
        "last_rmse": last_metrics["rmse"],
        "last_lr": get_current_lr(optimizer),
    }


def main() -> int:
    args = parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
