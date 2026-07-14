"""
PyTorch OPT Attention model from opt: 
https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py
"""
import math
import warnings
from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache, StaticCache
from transformers.models.opt.configuration_opt import OPTConfig

from ....quantization.clip import pseudo_quantize_tensor_clip
from ....quantization.quantizer import ActQuantizer

import os

class OPTAttention_PINN(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        args,
        config: OPTConfig,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # isn't needed in normal attention, but needed in flash attention so to keep the signature same
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Keep q/k/v exact here to match the LLaMA FPQ path. FP8 PINN is intended
        # to quantize the approximator path, not to add another attention-path change.
        
        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        raw_logits = torch.matmul(query_states, key_states.transpose(3, 2))

        #------PINN softmax------#
        target_dtype = torch.float16
        attn_logits_base = raw_logits
        current_dtype = query_states.dtype 

        if attention_mask is not None:
            key_seq_len = key_states.shape[-2]
            causal_mask = attention_mask[:, :, :query_states.shape[-2], :key_seq_len]
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

        exp_approx_fp32 = torch.zeros_like(shifted_logits, dtype=target_dtype)
        valid_base_logits=attn_logits_base[valid_positions_mask]
        valid_shifted_logits = shifted_logits[valid_positions_mask] 
        valid_shifted_logits_reshaped = valid_shifted_logits.to(target_dtype).unsqueeze(-1) # (num_valid, 1) #float32

        if torch.all(valid_shifted_logits_reshaped == 0):
            print("valid_shifted_logits_reshaped is all zeros:", valid_shifted_logitss_reshaped)
            raise ValueError("valid_shifted_logits_reshaped is all zeros")
        if self.args.quant_act:
            valid_shifted_logits_reshaped = self.pinn_input_quantizer(valid_shifted_logits_reshaped)
        exp_elements_approx = self.custom_scalar_expo_net(valid_shifted_logits_reshaped, self.args)  # (num_valid, 1)
        if self.args.quant_act:
            exp_elements_approx = self.pinn_output_quantizer(exp_elements_approx)
        exp_approx_fp32[valid_positions_mask] = exp_elements_approx.squeeze(-1)
        sum_exp_approx_per_row = torch.sum(exp_approx_fp32, dim=-1, keepdim=True) # (B,H,Q,1

        probabilities_fp32 =exp_approx_fp32 / (sum_exp_approx_per_row )

        attn_weights_processed = torch.nan_to_num(probabilities_fp32, nan=0.0, posinf=0.0, neginf=0.0)

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

        attn_output = attn_output.transpose(1, 2).contiguous()

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs, past_key_value
