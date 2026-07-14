"""
PyTorch OPT attention with attention-exp replaced by the NLI LUT baseline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.opt.configuration_opt import OPTConfig

from ....quantization.quantizer import ActQuantizer
from ...common import ScalarNliLut


class OPTAttention_NLI(nn.Module):
    def __init__(
        self,
        args,
        config: OPTConfig,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias
        self.layer_idx = layer_idx
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True
        self.args = args

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        del output_attentions, position_ids
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        raw_logits = torch.matmul(query_states, key_states.transpose(3, 2))
        attn_logits_base = raw_logits
        current_dtype = query_states.dtype

        if attention_mask is not None:
            key_seq_len = key_states.shape[-2]
            causal_mask = attention_mask[:, :, : query_states.shape[-2], :key_seq_len]
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
        exp_approx = torch.zeros_like(shifted_logits, dtype=current_dtype)
        valid_shifted_logits = shifted_logits[valid_positions_mask]
        valid_shifted_logits_reshaped = valid_shifted_logits.to(current_dtype).unsqueeze(-1)

        if self.args.quant_approx_activations:
            valid_shifted_logits_reshaped = self.approx_input_quantizer(valid_shifted_logits_reshaped)
        exp_elements_approx = self.exp_lut(
            valid_shifted_logits_reshaped,
            input_scale=self.approx_input_quantizer.scale if self.args.quant_approx_activations else None,
        )
        if self.args.quant_approx_activations:
            exp_elements_approx = self.approx_output_quantizer(exp_elements_approx)

        exp_approx[valid_positions_mask] = exp_elements_approx.squeeze(-1).to(exp_approx.dtype)
        sum_exp_approx_per_row = torch.sum(exp_approx, dim=-1, keepdim=True)
        probabilities = exp_approx / sum_exp_approx_per_row
        attn_weights_processed = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        attn_weights = attn_weights_processed.to(current_dtype)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_probs, past_key_value
