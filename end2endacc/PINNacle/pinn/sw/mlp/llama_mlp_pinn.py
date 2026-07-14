import math
import warnings
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache, StaticCache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    # is_flash_attn_greater_or_equal_2_10,
    apply_rotary_pos_emb,
    repeat_kv,
)

import time
import os

from ....quantization.clip import pseudo_quantize_tensor_clip
from ....quantization.quantizer import ActQuantizer


def _linear_compute_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    weight = getattr(module, "weight", None)
    if weight is not None:
        return weight.dtype
    weight_fp = getattr(module, "weight_fp", None)
    if weight_fp is not None:
        return weight_fp.dtype
    bias = getattr(module, "bias", None)
    if bias is not None:
        return bias.dtype
    return fallback


class LlamaMLP_PINN(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        # self.act_fn = ACT2FN[config.hidden_act]
        
        self.args = args
        
        self.pinn_gate_quantizer = ActQuantizer(n_bits=args.a_bits, q_group_size=args.a_group_size, per_tensor=args.a_per_tensor,
                                        fpq=args.fpq, mantissa_bit=args.a_mantissa_bit, clip=args.a_clip)
        self.pinn_up_quantizer = ActQuantizer(n_bits=args.a_bits, q_group_size=args.a_group_size, per_tensor=args.a_per_tensor,
                                        fpq=args.fpq, mantissa_bit=args.a_mantissa_bit, clip=args.a_clip)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if args.pinn_dim == 8:
            from .d8.pinn_d8 import MlpSigmoidApproximator_d8
            self.custom_sigmoid_net = MlpSigmoidApproximator_d8()
            mlp_sigmoid_save_path = os.path.join(current_dir, "d8", "best_mlp_sigmoid_approx_8.pth")
            if os.path.exists(mlp_sigmoid_save_path):
                print(f"Loading ScalarExpoNet weights from {mlp_sigmoid_save_path}...")
                self.custom_sigmoid_net.load_state_dict(torch.load(mlp_sigmoid_save_path, map_location='cpu'), strict=False)
                print("ScalarExpoNet weights loaded successfully.")
            else:
                raise Exception(f"ScalarExpoNet weight file not found: {mlp_sigmoid_save_path}")
        elif args.pinn_dim == 16:
            from .d16.pinn_d16 import MlpSigmoidApproximator_d16
            self.custom_sigmoid_net = MlpSigmoidApproximator_d16()
            mlp_sigmoid_save_path = os.path.join(current_dir, "d16", "best_mlp_sigmoid_approx_16.pth")
            if os.path.exists(mlp_sigmoid_save_path):
                print(f"Loading ScalarExpoNet weights from {mlp_sigmoid_save_path}...")
                self.custom_sigmoid_net.load_state_dict(torch.load(mlp_sigmoid_save_path, map_location='cpu'), strict=False)
                print("ScalarExpoNet weights loaded successfully.")
            else:
                raise Exception(f"ScalarExpoNet weight file not found: {mlp_sigmoid_save_path}")
        elif args.pinn_dim == 32:
            from .d32.pinn_d32 import MlpSigmoidApproximator_d32
            self.custom_sigmoid_net = MlpSigmoidApproximator_d32()
            mlp_sigmoid_save_path = os.path.join(current_dir, "d32", "best_mlp_sigmoid_approx_32.pth")
            if os.path.exists(mlp_sigmoid_save_path):
                print(f"Loading ScalarExpoNet weights from {mlp_sigmoid_save_path}...")
                self.custom_sigmoid_net.load_state_dict(torch.load(mlp_sigmoid_save_path, map_location='cpu'), strict=False)
                print("ScalarExpoNet weights loaded successfully.")
            else:
                raise Exception(f"ScalarExpoNet weight file not found: {mlp_sigmoid_save_path}")
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_dtype = _linear_compute_dtype(self.down_proj, x.dtype)
        gate_output = self.gate_proj(x).to(proj_dtype)

        clipped_gate_output = gate_output

        original_shape = clipped_gate_output.shape
        clipped_gate_output=clipped_gate_output.flatten()
        flat_input = clipped_gate_output.reshape(-1, 1)
        positive_mask = flat_input >= 0
        if self.args.quant_act:
            flat_input = self.pinn_gate_quantizer(flat_input)
        output_when_neg = self.custom_sigmoid_net(flat_input, self.args)
        output_when_pos = 1.0 - self.custom_sigmoid_net(-flat_input, self.args)
        activated_gate_imed_flat = torch.where(positive_mask, output_when_pos, output_when_neg)
        #activated_gate_imed = self.custom_sigmoid_net(clipped_gate_output)
        clipped_gate_output=clipped_gate_output.reshape(original_shape)
        activated_gate_imed=activated_gate_imed_flat.reshape(original_shape) 
        activated_gate = (activated_gate_imed * clipped_gate_output).to(proj_dtype)

        up_output = self.up_proj(x).to(proj_dtype)
        if self.args.quant_act:
            up_output = self.pinn_up_quantizer(up_output)
        gated_mechanism_output = (activated_gate * up_output).to(proj_dtype)

        down_proj_output = self.down_proj(gated_mechanism_output)
        
        return down_proj_output
