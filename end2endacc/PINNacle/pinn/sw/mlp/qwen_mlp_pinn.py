import os

import torch
from torch import nn

from ....quantization.quantizer import ActQuantizer
from .llama_mlp_pinn import _linear_compute_dtype


def _load_sigmoid_net(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.pinn_dim == 8:
        from .d8.pinn_d8 import MlpSigmoidApproximator_d8

        net = MlpSigmoidApproximator_d8()
        path = os.path.join(current_dir, "d8", "best_mlp_sigmoid_approx_8.pth")
    elif args.pinn_dim == 16:
        from .d16.pinn_d16 import MlpSigmoidApproximator_d16

        net = MlpSigmoidApproximator_d16()
        path = os.path.join(current_dir, "d16", "best_mlp_sigmoid_approx_16.pth")
    elif args.pinn_dim == 32:
        from .d32.pinn_d32 import MlpSigmoidApproximator_d32

        net = MlpSigmoidApproximator_d32()
        path = os.path.join(current_dir, "d32", "best_mlp_sigmoid_approx_32.pth")
    else:
        raise ValueError(f"Unsupported pinn_dim={args.pinn_dim}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"ScalarExpoNet weight file not found: {path}")
    print(f"Loading ScalarExpoNet weights from {path}...")
    net.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    print("ScalarExpoNet weights loaded successfully.")
    return net


class QwenMLP_PINN(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.args = args

        self.pinn_gate_quantizer = ActQuantizer(
            n_bits=args.a_bits,
            q_group_size=args.a_group_size,
            per_tensor=args.a_per_tensor,
            fpq=args.fpq,
            mantissa_bit=args.a_mantissa_bit,
            clip=args.a_clip,
        )
        self.pinn_up_quantizer = ActQuantizer(
            n_bits=args.a_bits,
            q_group_size=args.a_group_size,
            per_tensor=args.a_per_tensor,
            fpq=args.fpq,
            mantissa_bit=args.a_mantissa_bit,
            clip=args.a_clip,
        )
        self.custom_sigmoid_net = _load_sigmoid_net(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_dtype = _linear_compute_dtype(self.down_proj, x.dtype)
        gate_output = self.gate_proj(x).to(proj_dtype)
        original_shape = gate_output.shape
        flat_input = gate_output.flatten().reshape(-1, 1)
        positive_mask = flat_input >= 0

        if self.args.quant_act:
            flat_input = self.pinn_gate_quantizer(flat_input)

        output_when_neg = self.custom_sigmoid_net(flat_input, self.args)
        output_when_pos = 1.0 - self.custom_sigmoid_net(-flat_input, self.args)
        sigmoid_flat = torch.where(positive_mask, output_when_pos, output_when_neg)
        sigmoid_gate = sigmoid_flat.reshape(original_shape)
        activated_gate = (sigmoid_gate * gate_output).to(proj_dtype)

        up_output = self.up_proj(x).to(proj_dtype)
        if self.args.quant_act:
            up_output = self.pinn_up_quantizer(up_output)
        return self.down_proj((activated_gate * up_output).to(proj_dtype))
