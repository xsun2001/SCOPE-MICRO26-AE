from typing import Optional, Tuple

import os
import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as apply_qwen2_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as apply_qwen3_rotary_pos_emb

from ....quantization.quantizer import ActQuantizer


def _load_expo_net(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.pinn_dim == 8:
        from .d8.pinn_d8 import ScalarExpoNet_d8

        net = ScalarExpoNet_d8()
        path = os.path.join(current_dir, "d8", "best_mlp1_scalar_expo_8.pth")
    elif args.pinn_dim == 16:
        from .d16.pinn_d16 import ScalarExpoNet_d16

        net = ScalarExpoNet_d16()
        path = os.path.join(current_dir, "d16", "best_mlp1_scalar_expo_16.pth")
    elif args.pinn_dim == 32:
        from .d32.pinn_d32 import ScalarExpoNet_d32

        net = ScalarExpoNet_d32()
        path = os.path.join(current_dir, "d32", "best_mlp1_scalar_expo_32.pth")
    else:
        raise ValueError(f"Unsupported pinn_dim={args.pinn_dim}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"ScalarExpoNet weight file not found: {path}")
    print(f"Loading ScalarExpoNet weights from {path}...")
    net.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    print("ScalarExpoNet weights loaded successfully.")
    return net


def qwen_pinn_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    args=None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    logits_fp32 = logits.float()

    if attention_mask is not None:
        key_seq_len = key_states.shape[-2]
        causal_mask = attention_mask[:, :, : query.shape[-2], :key_seq_len]
        valid_positions_mask = causal_mask > -1e4
        if valid_positions_mask.shape[1] == 1 and logits_fp32.shape[1] > 1:
            valid_positions_mask = valid_positions_mask.expand_as(logits_fp32)
    else:
        valid_positions_mask = torch.ones_like(logits_fp32, dtype=torch.bool)

    masked_logits = logits_fp32.masked_fill(~valid_positions_mask, -float("inf"))
    max_logits = torch.max(masked_logits, dim=-1, keepdim=True)[0]
    max_logits = torch.where(torch.isneginf(max_logits), torch.zeros_like(max_logits), max_logits)
    shifted_logits = logits_fp32 - max_logits

    exp_values = torch.zeros_like(logits_fp32, dtype=torch.float32)
    valid_shifted = shifted_logits[valid_positions_mask].unsqueeze(-1)
    if valid_shifted.numel() == 0:
        attn_weights_fp32 = exp_values
    else:
        if args.quant_act:
            valid_shifted = module.pinn_input_quantizer(valid_shifted)
        exp_approx = module.custom_scalar_expo_net(valid_shifted, args).float()
        if args.quant_act:
            exp_approx = module.pinn_output_quantizer(exp_approx).float()
        exp_values[valid_positions_mask] = exp_approx.squeeze(-1)
        denom = exp_values.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        attn_weights_fp32 = exp_values / denom

    attn_weights = torch.nan_to_num(attn_weights_fp32, nan=0.0, posinf=0.0, neginf=0.0).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class _QwenAttentionPinnBase(nn.Module):
    def _init_common(self, args, config, layer_idx: int) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.args = args

        self.q_quantizer = ActQuantizer(
            n_bits=args.a_bits,
            q_group_size=args.a_group_size,
            per_tensor=args.a_per_tensor,
            fpq=args.fpq,
            mantissa_bit=args.a_mantissa_bit,
            clip=args.a_clip,
        )
        self.k_quantizer = ActQuantizer(
            n_bits=args.a_bits,
            q_group_size=args.a_group_size,
            per_tensor=args.a_per_tensor,
            fpq=args.fpq,
            mantissa_bit=args.a_mantissa_bit,
            clip=args.a_clip,
        )
        self.v_quantizer = ActQuantizer(
            n_bits=args.a_bits,
            q_group_size=args.a_group_size,
            per_tensor=args.a_per_tensor,
            fpq=args.fpq,
            mantissa_bit=args.a_mantissa_bit,
            clip=args.a_clip,
        )
        self.pinn_input_quantizer = ActQuantizer(
            n_bits=args.a_bits,
            q_group_size=args.a_group_size,
            per_tensor=args.a_per_tensor,
            fpq=args.fpq,
            mantissa_bit=args.a_mantissa_bit,
            clip=args.a_clip,
            use_8bit_scale=True,
        )
        self.pinn_output_quantizer = ActQuantizer(
            n_bits=args.a_bits,
            q_group_size=args.a_group_size,
            per_tensor=args.a_per_tensor,
            fpq=args.fpq,
            mantissa_bit=args.a_mantissa_bit,
            clip=args.a_clip,
            use_8bit_scale=True,
        )
        self.custom_scalar_expo_net = _load_expo_net(args)

    def _project_qkv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    def _normalize_qk(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return query_states, key_states

    def _apply_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

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
            raise ValueError(f"{self.__class__.__name__} requires precomputed position_embeddings.")

        input_shape = hidden_states.shape[:-1]
        query_states, key_states, value_states = self._project_qkv(hidden_states)
        query_states, key_states = self._normalize_qk(query_states, key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation != "eager":
            raise ValueError(f"Only eager attention is supported for {self.__class__.__name__}")

        attn_output, attn_weights = qwen_pinn_attention_forward(
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


class Qwen2Attention_PINN(_QwenAttentionPinnBase):
    def __init__(self, args, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self._init_common(args, config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def _apply_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return apply_qwen2_rotary_pos_emb(query_states, key_states, cos, sin)


class Qwen3Attention_PINN(_QwenAttentionPinnBase):
    def __init__(self, args, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self._init_common(args, config, layer_idx)
        attention_bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _normalize_qk(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(query_states), self.k_norm(key_states)

    def _apply_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return apply_qwen3_rotary_pos_emb(query_states, key_states, cos, sin)
