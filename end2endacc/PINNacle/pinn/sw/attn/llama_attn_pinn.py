"""
PyTorch LLaMA Attention model from llama: 
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
import math
import warnings
from typing import List, Optional, Tuple, Union, Callable, Any

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


def qeager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor], # 4D causal mask
    scaling: float,
    dropout: float = 0.0,
    args: Optional[Any] = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # if args.quant_act:
    #     query = module.q_quantizer(query)
    #     key_states = module.k_quantizer(key_states)
    #     value_states = module.v_quantizer(value_states)

    logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    logits_fp32 = logits.float()
    current_dtype = query.dtype

    if attention_mask is not None:
        key_seq_len = key_states.shape[-2]
        causal_mask = attention_mask[:, :, :query.shape[-2], :key_seq_len]
        valid_positions_mask = causal_mask > -1e4
        if valid_positions_mask.shape[1] == 1 and logits_fp32.shape[1] > 1:
            valid_positions_mask = valid_positions_mask.expand_as(logits_fp32)
    else:
        valid_positions_mask = torch.ones_like(logits_fp32, dtype=torch.bool)

    masked_logits = logits_fp32.masked_fill(~valid_positions_mask, -float("inf"))
    max_logits_per_row = torch.max(masked_logits, dim=-1, keepdim=True)[0]
    max_logits_per_row = torch.where(
        torch.isneginf(max_logits_per_row),
        torch.zeros_like(max_logits_per_row),
        max_logits_per_row,
    )
    shifted_logits = logits_fp32 - max_logits_per_row

    exp_approx_fp32 = torch.zeros_like(logits_fp32, dtype=torch.float32)
    valid_shifted_logits = shifted_logits[valid_positions_mask].unsqueeze(-1)
    if valid_shifted_logits.numel() == 0:
        probabilities_fp32 = exp_approx_fp32
    else:
        if getattr(args, "quant_act", False):
            valid_shifted_logits = module.pinn_input_quantizer(valid_shifted_logits)
        exp_elements_approx = module.custom_scalar_expo_net(valid_shifted_logits, args).float()
        if getattr(args, "quant_act", False):
            exp_elements_approx = module.pinn_output_quantizer(exp_elements_approx).float()
        exp_approx_fp32[valid_positions_mask] = exp_elements_approx.squeeze(-1)
        sum_exp_approx_per_row = exp_approx_fp32.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        probabilities_fp32 = exp_approx_fp32 / sum_exp_approx_per_row

    attn_weights_processed = torch.nan_to_num(probabilities_fp32, nan=0.0, posinf=0.0, neginf=0.0)
    attn_weights = attn_weights_processed.to(current_dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    expected_output_dtype = _linear_compute_dtype(module.o_proj, current_dtype)
    attn_output = attn_output.to(expected_output_dtype)

    return attn_output, attn_weights



class LlamaAttention_PINN(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, args, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        self.args = args

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        
        self.q_quantizer = ActQuantizer(n_bits=args.a_bits, q_group_size=args.a_group_size, per_tensor=args.a_per_tensor,
                                        fpq=args.fpq, mantissa_bit=args.a_mantissa_bit, clip=args.a_clip)
        self.k_quantizer = ActQuantizer(n_bits=args.a_bits, q_group_size=args.a_group_size, per_tensor=args.a_per_tensor,
                                        fpq=args.fpq, mantissa_bit=args.a_mantissa_bit, clip=args.a_clip)
        self.v_quantizer = ActQuantizer(n_bits=args.a_bits, q_group_size=args.a_group_size, per_tensor=args.a_per_tensor,
                                        fpq=args.fpq, mantissa_bit=args.a_mantissa_bit, clip=args.a_clip)
        self.pinn_input_quantizer = ActQuantizer(n_bits=args.a_bits, q_group_size=args.a_group_size, per_tensor=args.a_per_tensor,
                                        fpq=args.fpq, mantissa_bit=args.a_mantissa_bit, clip=args.a_clip, use_8bit_scale=True)
        self.pinn_output_quantizer = ActQuantizer(n_bits=args.a_bits, q_group_size=args.a_group_size, per_tensor=args.a_per_tensor,
                                        fpq=args.fpq, mantissa_bit=args.a_mantissa_bit, clip=args.a_clip, use_8bit_scale=True)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if args.pinn_dim == 8:
            from .d8.pinn_d8 import ScalarExpoNet_d8
            self.custom_scalar_expo_net = ScalarExpoNet_d8()
            mlp1_scalar_expo_save_path = os.path.join(current_dir, "d8", "best_mlp1_scalar_expo_8.pth")
            if os.path.exists(mlp1_scalar_expo_save_path):
                print(f"Loading ScalarExpoNet weights from {mlp1_scalar_expo_save_path}...")
                self.custom_scalar_expo_net.load_state_dict(torch.load(mlp1_scalar_expo_save_path, map_location='cpu'), strict=False)
                print("ScalarExpoNet weights loaded successfully.")
            else:
                raise Exception(f"ScalarExpoNet weight file not found: {mlp1_scalar_expo_save_path}")
        elif args.pinn_dim == 16:
            from .d16.pinn_d16 import ScalarExpoNet_d16
            self.custom_scalar_expo_net = ScalarExpoNet_d16()
            mlp1_scalar_expo_save_path = os.path.join(current_dir, "d16", "best_mlp1_scalar_expo_16.pth")
            if os.path.exists(mlp1_scalar_expo_save_path):
                print(f"Loading ScalarExpoNet weights from {mlp1_scalar_expo_save_path}...")
                self.custom_scalar_expo_net.load_state_dict(torch.load(mlp1_scalar_expo_save_path, map_location='cpu'), strict=False)
                print("ScalarExpoNet weights loaded successfully.")
            else:
                raise Exception(f"ScalarExpoNet weight file not found: {mlp1_scalar_expo_save_path}")
        elif args.pinn_dim == 32:
            from .d32.pinn_d32 import ScalarExpoNet_d32
            self.custom_scalar_expo_net = ScalarExpoNet_d32()
            mlp1_scalar_expo_save_path = os.path.join(current_dir, "d32", "best_mlp1_scalar_expo_32.pth")
            if os.path.exists(mlp1_scalar_expo_save_path):
                print(f"Loading ScalarExpoNet weights from {mlp1_scalar_expo_save_path}...")
                self.custom_scalar_expo_net.load_state_dict(torch.load(mlp1_scalar_expo_save_path, map_location='cpu'), strict=False)
                print("ScalarExpoNet weights loaded successfully.")
            else:
                raise Exception(f"ScalarExpoNet weight file not found: {mlp1_scalar_expo_save_path}")  
        
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if position_embeddings is None:
            raise ValueError("LlamaAttention_PINN requires precomputed position_embeddings.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = qeager_attention_forward
        if self.config._attn_implementation != "eager":
            raise ValueError("Only eager attention is supported for LlamaAttention_PINN")

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            args=self.args,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights
