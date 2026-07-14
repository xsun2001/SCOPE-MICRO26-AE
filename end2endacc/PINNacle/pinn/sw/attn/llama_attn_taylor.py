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



def taylor_exp_approx(v: torch.Tensor, order: int) -> torch.Tensor:
    """
    Efficiently compute the N-th order Taylor expansion approximation of e^x using Horner's method.
    Args:
        v (torch.Tensor): Input tensor.
        order (int): Order of the Taylor expansion.

    Returns:
        torch.Tensor: N-th order Taylor expansion approximation result of e^v.
    """
    if order == 0:
        return torch.ones_like(v)
    
    # Start calculation from the innermost layer of Horner's method: (1 + v / N)
    y = 1.0 + v / order
    
    # Loop from N-1 down to 2
    for i in range(order - 1, 1, -1):
        y = 1.0 + v / i * y
        
    # Final step: 1 + v * (...)
    y = 1.0 + v * y
    
    return y

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

    # 1. Compute the original logits
    raw_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # 2. Pre-prune the original logits
    #    Based on your code snippet, the latest clipping range is min=-20.0, max=15.0

    attn_logits_base = raw_logits
    # attn_logits_base=raw_logits
    current_dtype = query.dtype # LLaMA's main inference precision (e.g., fp16)
    original_shape = attn_logits_base.shape # (B, H, Q, K)

    if attention_mask is not None:
        # attention_mask values close to 0 (or > a certain negative threshold) are valid positions
        key_seq_len = key_states.shape[-2]
        causal_mask = attention_mask[:, :, :query.shape[-2], :key_seq_len]
        valid_positions_mask = (causal_mask > -1e4) # Threshold may need to be adjusted based on your masking strategy
        if valid_positions_mask.shape[1] == 1 and attn_logits_base.shape[1] > 1: # Broadcast head dimension
            valid_positions_mask = valid_positions_mask.expand_as(attn_logits_base)
    else:
        valid_positions_mask = torch.ones_like(attn_logits_base, dtype=torch.bool)

    # b. For numerical stability, subtract the maximum value of each row from the valid logits
    #    Create a temporary tensor where invalid positions are filled with -inf to correctly find the maximum value of the valid part
    temp_logits_for_max_calc = torch.where(
        valid_positions_mask, 
        attn_logits_base, 
        torch.tensor(-float('inf'), device=attn_logits_base.device, dtype=attn_logits_base.dtype)
    )


    max_logits_per_row = torch.max(temp_logits_for_max_calc, dim=-1, keepdim=True)[0]


    # Handle the case where an entire row is masked (in which case max_logits_per_row is -inf), replace with 0 to avoid -(-inf)
    max_logits_per_row = torch.where(
        torch.isneginf(max_logits_per_row), 
        torch.zeros_like(max_logits_per_row), 
        max_logits_per_row
    )
    

    # --- Timing ends ---

    shifted_logits = attn_logits_base - max_logits_per_row# Core problem point

    #    Create a tensor with the same shape as shifted_logits, filled with 0, to store the exponential terms
    exp_approx_fp32 = torch.zeros_like(shifted_logits, dtype=target_dtype)#float32
    # Extract valid logits and reshape to (num_valid_elements, 1)
    valid_base_logits=attn_logits_base[valid_positions_mask]
    valid_shifted_logits = shifted_logits[valid_positions_mask] 
    valid_shifted_logits_reshaped = valid_shifted_logits.to(target_dtype).unsqueeze(-1) # (num_valid, 1) #float32

    # Use a piecewise linear fitting neural network for exp fp16
    v=valid_shifted_logits_reshaped
    ln2 = 0.6931471805599453
    # 2. Decompose the input v into an integer n and a small-range residual r
    #  v = n * ln(e) + r
    n = torch.round(v / ln2)
    two_n = torch.pow(2.0, n.to(target_dtype)) 
    r = v - n * ln2
    
    #exp_elements_approx = taylor_exp_approx(valid_shifted_logits_reshaped/35, 1).to(torch.float16)
    # Here, range reduction is necessary (principle: the reciprocal of each order near 0, expanded near 0, the farther away from 0, the less like; for large numbers, overflow; for large numbers, it cannot keep up with the growth of e^x)
    # # First-order Taylor expansion
    # exp_elements_approx=(1+r).to(torch.float16)
    #exp_elements_approx = taylor_exp_approx(r, 1).to(torch.float16)
    # Second-order Taylor expansion
    # exp_elements_approx = (1 + r * (1 + r / 2.0)).to(torch.float16)
    # exp_elements_approx = taylor_exp_approx(v/35, 2).to(torch.float16)# Directly scale to -0.1, 0
    # Third-order Taylor expansion
    exp_elements_approx = taylor_exp_approx(r, 3).to(target_dtype)
    # Fourth-order Taylor expansion
    #exp_elements_approx = taylor_exp_approx(r, 4).to(torch.float16)
    # Fifth-order Taylor expansion
    # exp_elements_approx = taylor_exp_approx(r, 5).to(torch.float16)
    # # Merge the results
    exp_elements_approx = (two_n * exp_elements_approx).to(target_dtype)

    exp_approx_fp32[valid_positions_mask] = exp_elements_approx.squeeze(-1) # Remove the last dimension 1
    #    Since the invalid positions of exp_approx_fp32 are already 0, directly sum
    sum_exp_approx_per_row = torch.sum(exp_approx_fp32, dim=-1, keepdim=True) # (B,H,Q,1

    probabilities_fp32 =exp_approx_fp32 / (sum_exp_approx_per_row )

    #probabilities_fp32 = torch.where(probabilities_fp32 > 0, probabilities_fp32, 5.96e-8)

    # f. Handle possible NaNs caused by an entire row summing to 0 (e.g., an entire row is masked, causing sum_exp_logits_per_row to be 0)
    attn_weights_processed = torch.nan_to_num(probabilities_fp32, nan=0.0, posinf=0.0, neginf=0.0)

    # g. Convert the final probabilities back to the original query's data type
    attn_weights = attn_weights_processed.to(current_dtype)
    
    # --- Softmax application ends ---

    # 4. Apply Dropout
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # 5. Compute the attention output
    # --- 3. Attn_weights and Value multiplication time statistics ---
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_av = time.perf_counter()
    attn_output = torch.matmul(attn_weights, value_states)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    av_time = time.perf_counter() - start_av

    attn_output = attn_output.transpose(1, 2).contiguous()
    expected_output_dtype = module.o_proj.weight.dtype
    attn_output = attn_output.to(expected_output_dtype)


    return attn_output, attn_weights


class LlamaAttention_Taylor(nn.Module):
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
            raise ValueError("Only eager attention is supported for LlamaAttention_Taylor")

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
