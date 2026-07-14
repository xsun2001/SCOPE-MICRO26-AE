"""
PyTorch LLaMA Attention model from llama: 
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
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



def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor], # 4D causal mask
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    target_dtype = torch.float16
    
    query = query.to(target_dtype)
    key = key.to(target_dtype)
    value = value.to(target_dtype)


    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    raw_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    attn_logits_base = raw_logits
    current_dtype = query.dtype
    original_shape = attn_logits_base.shape

    if attention_mask is not None:
        key_seq_len = key_states.shape[-2]
        causal_mask = attention_mask[:, :, :query.shape[-2], :key_seq_len]
        valid_positions_mask = (causal_mask > -1e4)
        if valid_positions_mask.shape[1] == 1 and attn_logits_base.shape[1] > 1:
            valid_positions_mask = valid_positions_mask.expand_as(attn_logits_base)
    else:
        valid_positions_mask = torch.ones_like(attn_logits_base, dtype=torch.bool)
    temp_logits_for_max_calc = torch.where(
        valid_positions_mask, 
        attn_logits_base, 
        torch.tensor(-float('inf'), device=attn_logits_base.device, dtype=attn_logits_base.dtype)
    )


    max_logits_per_row = torch.max(temp_logits_for_max_calc, dim=-1, keepdim=True)[0]

    max_logits_per_row = torch.where(
        torch.isneginf(max_logits_per_row), 
        torch.zeros_like(max_logits_per_row), 
        max_logits_per_row
    )

    shifted_logits = attn_logits_base - max_logits_per_row

    exp_approx_fp32 = torch.zeros_like(shifted_logits, dtype=torch.float16)
    valid_base_logits=attn_logits_base[valid_positions_mask]
    valid_shifted_logits = shifted_logits[valid_positions_mask] 
    valid_shifted_logits_reshaped = valid_shifted_logits.to(torch.float16).unsqueeze(-1) # (num_valid, 1) #float32

    v=valid_shifted_logits_reshaped
    NUM_SEGMENTS = 8
    nodes_x = torch.linspace(-1, 0, NUM_SEGMENTS + 1)
    nodes_y = torch.pow(2.0, nodes_x)
    slopes = (nodes_y[1:] - nodes_y[:-1]) / (nodes_x[1:] - nodes_x[:-1])
    intercepts = nodes_y[:-1] - slopes * nodes_x[:-1]
    LOG2_E = math.log2(math.e)
    s = slopes.to(v.device)
    i = intercepts.to(v.device)
    x = v * LOG2_E
    x_i = torch.ceil(x)
    x_f = x - x_i
    two_to_the_x_i = torch.pow(2.0, x_i)
    k = torch.floor((x_f + 1) * NUM_SEGMENTS)
    k = torch.clamp(k, 0, NUM_SEGMENTS - 1).long()
    slope_k = s[k]
    intercept_k = i[k]
    two_to_the_x_f_approx = slope_k * x_f + intercept_k
    exp_elements_approx = (two_to_the_x_i * two_to_the_x_f_approx).to(torch.float16)


    exp_approx_fp32[valid_positions_mask] = exp_elements_approx.squeeze(-1)
    sum_exp_approx_per_row = torch.sum(exp_approx_fp32, dim=-1, keepdim=True)

    probabilities_fp32 =exp_approx_fp32 / (sum_exp_approx_per_row )

    attn_weights_processed = torch.nan_to_num(probabilities_fp32, nan=0.0, posinf=0.0, neginf=0.0)

    attn_weights = attn_weights_processed.to(current_dtype)
    
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_av = time.perf_counter()
    attn_output = torch.matmul(attn_weights, value_states)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    av_time = time.perf_counter() - start_av

    attn_output = attn_output.transpose(1, 2).contiguous()
    expected_output_dtype = module.o_proj.weight.dtype
    attn_output = attn_output.to(expected_output_dtype)

    return attn_output, attn_weights


class LlamaAttention_PLF(nn.Module):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            raise ValueError("Only eager attention is supported for LlamaAttention_PLF")

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
