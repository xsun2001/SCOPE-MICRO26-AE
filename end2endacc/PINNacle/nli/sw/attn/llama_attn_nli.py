"""
PyTorch LLaMA attention with attention-exp replaced by the NLI LUT baseline.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from ....quantization.quantizer import ActQuantizer
from ...common import ScalarNliLut


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
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    args: Optional[Any] = None,
    **kwargs,
):
    del kwargs
    target_dtype = query.dtype
    query = query * scaling

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    raw_logits = torch.matmul(query, key_states.transpose(2, 3))

    attn_logits_base = raw_logits
    current_dtype = query.dtype

    if attention_mask is not None:
        key_seq_len = key_states.shape[-2]
        causal_mask = attention_mask[:, :, : query.shape[-2], :key_seq_len]
        valid_positions_mask = causal_mask > -1e4
        if valid_positions_mask.shape[1] == 1 and attn_logits_base.shape[1] > 1:
            valid_positions_mask = valid_positions_mask.expand_as(attn_logits_base)
    else:
        valid_positions_mask = torch.ones_like(attn_logits_base, dtype=torch.bool)

    temp_logits_for_max_calc = torch.where(
        valid_positions_mask,
        attn_logits_base,
        torch.tensor(-float("inf"), device=attn_logits_base.device, dtype=attn_logits_base.dtype),
    )
    max_logits_per_row = torch.max(temp_logits_for_max_calc, dim=-1, keepdim=True)[0]
    max_logits_per_row = torch.where(
        torch.isneginf(max_logits_per_row),
        torch.zeros_like(max_logits_per_row),
        max_logits_per_row,
    )

    shifted_logits = attn_logits_base - max_logits_per_row
    exp_approx = torch.zeros_like(shifted_logits, dtype=target_dtype)
    valid_shifted_logits = shifted_logits[valid_positions_mask]
    valid_shifted_logits_reshaped = valid_shifted_logits.to(target_dtype).unsqueeze(-1)

    if args.quant_approx_activations:
        valid_shifted_logits_reshaped = module.approx_input_quantizer(valid_shifted_logits_reshaped)

    exp_elements_approx = module.exp_lut(
        valid_shifted_logits_reshaped,
        input_scale=module.approx_input_quantizer.scale if args.quant_approx_activations else None,
    )

    if args.quant_approx_activations:
        exp_elements_approx = module.approx_output_quantizer(exp_elements_approx)

    exp_approx[valid_positions_mask] = exp_elements_approx.squeeze(-1).to(exp_approx.dtype)
    sum_exp_approx_per_row = torch.sum(exp_approx, dim=-1, keepdim=True)
    probabilities = exp_approx / sum_exp_approx_per_row
    attn_weights_processed = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    attn_weights = attn_weights_processed.to(current_dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    expected_output_dtype = _linear_compute_dtype(module.o_proj, current_dtype)
    attn_output = attn_output.to(expected_output_dtype)
    return attn_output, attn_weights


class LlamaAttention_NLI(nn.Module):
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

        self.approx_input_quantizer = ActQuantizer(
            n_bits=args.a_bits,
            q_group_size=args.a_group_size,
            per_tensor=args.a_per_tensor,
            fpq=args.fpq,
            mantissa_bit=args.a_mantissa_bit,
            clip=args.a_clip,
            use_8bit_scale=True,
            emit_int_codes=True,
        )
        self.approx_output_quantizer = ActQuantizer(
            n_bits=args.a_bits,
            q_group_size=args.a_group_size,
            per_tensor=args.a_per_tensor,
            fpq=args.fpq,
            mantissa_bit=args.a_mantissa_bit,
            clip=args.a_clip,
            use_8bit_scale=True,
        )
        self.exp_lut = ScalarNliLut(
            args.approx_exp_lut_path,
            quantize_weights=args.quant_approx_weights,
            args=args,
            use_integer_control=args.quant_approx_activations,
        )

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        del output_attentions, use_cache, position_ids
        if position_embeddings is None:
            raise ValueError("LlamaAttention_NLI requires precomputed position_embeddings.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = qeager_attention_forward
        if self.config._attn_implementation != "eager":
            raise ValueError("Only eager attention is supported for LlamaAttention_NLI")

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
        return attn_output, attn_weights, past_key_value
