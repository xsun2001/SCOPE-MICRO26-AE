import torch, torch.nn as nn, torch.nn.functional as F, math, fast_hadamard_transform
import utils.hadamard_utils as hadamard_utils
from utils import Rdtype
from einops import rearrange
from fast_hadamard_transform import hadamard_transform
from .quantizer import Quantizer
from .scna import SCNAExpApproximator


class H(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_hadamard_transform = False  
        self.online_full_had = False  
        self.online_partial_had = False  
        self.had_K = None
        self.K = 1
        self.down_dim = 1
        self.fp32_had = False

    def free_temporary(self):
        if hasattr(self, "temporary"):
            self.temporary = False
        if hasattr(self, "temp_weight"):
            self.temp_weight = None
        if hasattr(self, "temp_bias"):
            self.temp_bias = None

    def may_hadamard_transform(self, out):
        i_dtype = out.dtype
        if self.use_hadamard_transform:  
            out = hadamard_transform(
                out.float() if self.fp32_had else out, scale=1 / (math.sqrt(out.size(-1)))
            ).to(i_dtype)
        if self.online_full_had:  
            if self.down_dim != 1:
                shape = out.shape  
                out = out.reshape(shape[0], shape[1], -1, self.down_dim).transpose(2, 3)
                if self.fp32_had:
                    out = (
                        hadamard_utils.matmul_hadU_cuda(
                            (out.float() if self.fp32_had else out).contiguous(), self.had_K, self.K
                        )
                        .transpose(2, 3)
                        .reshape(shape)
                        .to(i_dtype)
                    )
                else:
                    out = (
                        hadamard_utils.matmul_hadU_cuda(
                            (out.float() if self.fp32_had else out).contiguous(), self.had_K, self.K
                        )
                        .transpose(2, 3)
                        .reshape(shape)
                    )
            else:
                if self.fp32_had:
                    out = hadamard_utils.matmul_hadU_cuda(
                        out.float(), self.had_K, self.K
                    ).to(i_dtype)
                else:
                    out = hadamard_utils.matmul_hadU_cuda(out, self.had_K, self.K).to(
                        i_dtype
                    )
        elif self.online_partial_had:
            init_shape = out.shape  
            if (
                self.K == 1
            ):
                if self.fp32_had:
                    out = (
                        fast_hadamard_transform.hadamard_transform(
                            out.float().permute(0, 2, 3, 1),
                            scale=1 / math.sqrt(out.shape[1]),
                        )
                        .permute(0, 3, 1, 2)
                        .to(i_dtype)
                    )
                else:
                    out = fast_hadamard_transform.hadamard_transform(
                        out.permute(0, 2, 3, 1), scale=1 / math.sqrt(out.shape[1])
                    ).permute(
                        0, 3, 1, 2
                    )  
            else:
                if self.fp32_had:
                    out = out.float()
                
                out = (
                    (
                        out.permute(0, 2, 3, 1)
                        @ self.had_K.to(dtype=out.dtype, device=out.device)
                    ).permute(0, 3, 1, 2)
                    / (math.sqrt(self.K))
                ).to(i_dtype)
                
                
                
        return out


class QuantLinear(nn.Linear, H):
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params=dict(bits=8, sym=True, dynamic_method="pertoken"),
        act_quant_params=dict(bits=16, sym=True, dynamic_method="pertoken"),
        name=None,
        attn_instance=None,
    ):
        nn.Module.__init__(self)
        
        self.weight = org_module.weight.requires_grad_(False)
        if org_module.bias is not None:
            self.bias = org_module.bias.requires_grad_(False)
            
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        self.temporary = False

        self.weight_quantizer = Quantizer(**weight_quant_params)
        self.act_quantizer = Quantizer(**act_quant_params)
        self.use_act_quant = False
        self.use_weight_quant = False

        self.name = name

        self.num_key_value_heads = (
            attn_instance.num_key_value_heads if attn_instance is not None else None
        )
        self.num_key_value_groups = (
            attn_instance.num_key_value_groups if attn_instance is not None else None
        )
        self.fast_recoder = False
    def forward(
        self,
        x,
        R_res=None,
        
        R_ov=None,
        S_up_down=None,
        S_qk=None,
        S_ov=None,
        S_up_gate=None,
        S_norm_qkv=None,
        S_norm_upgate=None,
    ):
        if self.temporary:
            ori_dtype = self.weight.dtype
            temp_weight = self.weight
            if self.bias is not None:
                temp_bias = self.bias

            
            if S_up_down is not None and self.name == "up":
                temp_weight = temp_weight.to(Rdtype) / (S_up_down.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) / (S_up_down.view(-1))
            if S_up_down is not None and self.name == "down":
                temp_weight = temp_weight.to(Rdtype) * (S_up_down.view(1, -1))

            
            if S_qk is not None and self.name == "k":
                temp_weight = temp_weight.to(Rdtype) * (S_qk.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) * (S_qk.view(-1))
            if S_qk is not None and self.name == "q":  
                if self.weight.shape[0] > S_qk.numel():
                    S_qk = S_qk.reshape(self.num_key_value_heads, -1)
                    n_head, d = S_qk.shape
                    S_qk = (
                        S_qk[:, None, :]
                        .expand(n_head, self.num_key_value_groups, d)
                        .reshape(-1)
                    )
                temp_weight = temp_weight.to(Rdtype) / (S_qk.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) / (S_qk.view(-1))

            
            if S_ov is not None and self.name == "v":
                temp_weight = temp_weight.to(Rdtype) / (S_ov.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) / (S_ov.view(-1))
            if S_ov is not None and self.name == "o":  
                if self.weight.shape[0] > S_ov.numel():
                    S_ov = S_ov.reshape(self.num_key_value_heads, -1)
                    n_head, d = S_ov.shape
                    S_ov = (
                        S_ov[:, None, :]
                        .expand(n_head, self.num_key_value_groups, d)
                        .reshape(-1)
                    )
                temp_weight = temp_weight.to(Rdtype) * (S_ov.view(-1, 1))

            
            if (
                S_up_gate is not None and self.name == "up"
            ):  
                temp_weight = temp_weight.to(Rdtype) / (S_up_gate.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) / (S_up_gate.view(-1))
            if S_up_gate is not None and self.name == "gate":
                temp_weight = temp_weight.to(Rdtype) * (S_up_gate.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) * (S_up_gate.view(-1))

            
            if R_res is not None and self.name in ["q", "k", "v", "up", "gate"]:
                temp_weight = temp_weight.to(Rdtype) @ R_res.to(Rdtype)
            elif R_res is not None and self.name in ["down", "o"]:
                temp_weight = R_res.T.to(Rdtype) @ temp_weight.to(Rdtype)
                if self.bias is not None:
                    temp_bias = R_res.T.to(Rdtype) @ temp_bias.to(Rdtype)

            
            if R_ov is not None and self.name == "v":
                R_ov = torch.stack(list(R_ov), dim=0)
                had_dim = R_ov.shape[-1]
                W_ = temp_weight.t()
                transposed_shape = W_.shape
                temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                
                temp = ((temp.unsqueeze(2).to(Rdtype)) @ (R_ov.to(Rdtype))).squeeze(2)
                temp_weight = temp.reshape(transposed_shape).t()
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) @ R_ov.to(Rdtype)
            if R_ov is not None and self.name == "o":
                R_ov = torch.stack(list(R_ov), dim=0)
                had_dim = R_ov.shape[-1]
                init_shape = temp_weight.shape
                temp = temp_weight.reshape(-1, init_shape[-1] // had_dim, had_dim)
                if self.num_key_value_groups != 1:
                    h, d1, d2 = R_ov.shape
                    repeated_R_ov = (
                        R_ov[:, None, :, :]
                        .expand(h, self.num_key_value_groups, d1, d2)
                        .reshape(-1, d1, d2)
                    )
                    
                    temp = (
                        (temp.unsqueeze(2).to(Rdtype)) @ (repeated_R_ov.to(Rdtype))
                    ).squeeze(2)
                else:
                    temp = ((temp.unsqueeze(2).to(Rdtype)) @ (R_ov.to(Rdtype))).squeeze(
                        2
                    )
                temp_weight = temp.reshape(init_shape)

            
            if S_norm_qkv is not None and self.name in ["q", "k", "v"]:
                temp_weight = temp_weight.to(Rdtype) * (
                    S_norm_qkv.view(1, -1)
                )  

            
            if S_norm_upgate is not None and self.name in ["up", "gate"]:
                temp_weight = temp_weight.to(Rdtype) * (S_norm_upgate.view(1, -1))

            
            if self.name == "head" and R_res is not None:
                temp_weight = temp_weight.to(Rdtype) @ R_res.to(Rdtype)
            weight = temp_weight.to(ori_dtype)
            if self.bias is not None:
                bias = temp_bias.to(ori_dtype)
            else:
                bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_weight_quant:
            weight = self.weight_quantizer(weight)

        out = F.linear(x, weight, bias)
        if self.use_act_quant:
            out = self.act_quantizer(out)

        return out


class QuantMatmul(H):
    def __init__(
        self,
        act_quant_parmas=dict(bits=8, sym=True, dynamic_method="pertoken"),
        matmul_func=torch.matmul,
        is_pvmat=False,  
        is_qkmat=False,
    ):
        super().__init__()
        self.matmul_func = matmul_func
        self.is_pvmat = is_pvmat
        self.is_qkmat = is_qkmat
        self.act_quantizer = Quantizer(**act_quant_parmas)
        self.use_act_quant = False

    def forward(self, x1, x2):
        if self.is_qkmat:
            out = torch.matmul(x1.float(), x2.float())
        else:
            out = torch.matmul(x1, x2)
        out = self.may_hadamard_transform(out)
        if self.is_pvmat:
            b, h, l, c = out.shape
            out = out.transpose(1, 2).contiguous().reshape(b, l, h * c)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantROPE(H):
    def __init__(
        self,
        rope_quant_parmas=dict(bits=8, sym=True, dynamic_method="pertensor"),
        act_quant_params=dict(bits=8, sym=True, dynamic_method="pertensor"),
    ):
        super().__init__()
        self.rope_quantizer = Quantizer(**rope_quant_parmas)
        self.act_quantizer = Quantizer(**act_quant_params)

        self.use_act_quant = False
        self.use_weight_quant = False

    def forward(self, x, Wrope, pre_rope_Q=None, post_rope_Q=None):
        if pre_rope_Q is not None:
            Wrope = (pre_rope_Q.T.to(Rdtype)) @ Wrope.to(Rdtype)
        if post_rope_Q is not None:
            Wrope = Wrope.to(Rdtype) @ post_rope_Q.to(Rdtype)
        if self.use_weight_quant:
            Wrope = self.rope_quantizer(Wrope)
        out = torch.matmul(x, Wrope.to(x.dtype))
        out = self.may_hadamard_transform(out)
        if self.use_act_quant:
            b, l, h, d = out.shape
            out = out.reshape(b, l, -1)
            out = self.act_quantizer(out)
            out = out.reshape(b, l, h, d)

        return out


class QuantRMSNorm(H):
    weight: torch.Tensor

    def __init__(
        self,
        ori_norm,
        act_quant_params=dict(bits=8, symmetric=True, dynamic_method="perchannel"),
    ):
        super().__init__()
        self.eps = ori_norm.variance_epsilon
        self.act_quantizer = Quantizer(**act_quant_params)
        self.weight = ori_norm.weight.requires_grad_(False)
        self.bias = None

        self.temporary = False
        self.temp_weight = self.temp_bias = None
        self.use_act_quant = False

    def forward(self, hidden_states, S=None):
        i_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        weight = self.weight
        if self.temporary and S is not None:  
            weight = weight / S

        out = (weight * hidden_states).to(i_dtype)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantAdd(H):
    def __init__(
        self, act_quant_parmas=dict(bits=8, sym=True, dynamic_method="pertoken")
    ):
        super().__init__()
        self.act_quantizer = Quantizer(**act_quant_parmas)
        self.use_act_quant = False

    def forward(self, x1, x2):
        out = x1 + x2
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantSoftmax(H):
    def __init__(
        self,
        act_quant_params: dict = dict(),
        dim=-1,
        scna_dim: int | None = None,
        scna_artifact_root: str | None = None,
        scna_input_quant_bits: int = 0,
        scna_input_clip_min: float | None = None,
        scna_input_scale: float = 1.0,
        scna_output_floor_log: float | None = None,
        intattention: bool = False,
        intattention_exp_bits: int = 5,
        intattention_zero_thr: float = 6.6,
        intattention_output_bits: int = 8,
    ):
        super().__init__()
        self.act_quantizer = Quantizer(**act_quant_params)
        self.dim = dim
        self.intattention = intattention
        self.intattention_exp_bits = intattention_exp_bits
        self.intattention_zero_thr = intattention_zero_thr
        self.intattention_output_bits = intattention_output_bits
        self.scna_exp = (
            SCNAExpApproximator(scna_dim, scna_artifact_root)
            if scna_dim is not None and scna_dim > 0
            else None
        )
        self.scna_input_quant_bits = scna_input_quant_bits
        self.scna_input_clip_min = scna_input_clip_min
        self.scna_input_scale = scna_input_scale
        self.scna_output_floor_log = scna_output_floor_log
        self.scna_query_chunk_size = 256

        self.use_act_quant = False

    def forward(self, attn_weights, attn_mask=None):
        i_dtype = attn_weights.dtype
        attn_mask = self._align_attn_mask(attn_mask, attn_weights)
        if self.intattention:
            out_dtype = None if self.use_act_quant else i_dtype
            out = self._forward_intattention(attn_weights, attn_mask, out_dtype=out_dtype)
            if self.use_act_quant:
                out = self.act_quantizer(out)
            return out.to(i_dtype)

        if self.scna_exp is not None:
            out_dtype = None if self.use_act_quant else i_dtype
            out = self._forward_scna(attn_weights, attn_mask, out_dtype=out_dtype)
            if self.use_act_quant:
                out = self.act_quantizer(out)
            return out.to(i_dtype)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        out = F.softmax(attn_weights, dim=self.dim, dtype=torch.float32)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out.to(i_dtype)

    def _align_attn_mask(self, attn_mask, attn_weights):
        if attn_mask is None or attn_mask.dim() != attn_weights.dim():
            return attn_mask

        mask_slice = []
        changed = False
        softmax_dim = self.dim if self.dim >= 0 else attn_weights.dim() + self.dim
        for mask_size, target_size in zip(attn_mask.shape, attn_weights.shape):
            if mask_size == target_size or mask_size == 1:
                mask_slice.append(slice(None))
            elif mask_size > target_size:
                dim = len(mask_slice)
                if dim == softmax_dim:
                    mask_slice.append(slice(0, target_size))
                else:
                    mask_slice.append(slice(mask_size - target_size, mask_size))
                changed = True
            else:
                mask_slice.append(slice(None))
        return attn_mask[tuple(mask_slice)] if changed else attn_mask

    def _valid_mask(self, attn_weights, attn_mask):
        if attn_mask is None:
            return torch.isfinite(attn_weights)

        valid_mask = attn_mask.to(device=attn_weights.device, dtype=torch.float32) > -1e4
        if valid_mask.shape != attn_weights.shape:
            valid_mask = valid_mask.expand_as(attn_weights)
        return valid_mask & torch.isfinite(attn_weights)

    def _slice_attn_mask(self, attn_mask, query_dim, start, end):
        if attn_mask is None or attn_mask.dim() <= query_dim:
            return attn_mask
        if attn_mask.shape[query_dim] == 1:
            return attn_mask
        mask_slice = [slice(None)] * attn_mask.dim()
        mask_slice[query_dim] = slice(start, end)
        return attn_mask[tuple(mask_slice)]

    def _intattention_exp_table(self, device):
        num_bins = 1 << self.intattention_exp_bits
        qmax = float((1 << self.intattention_output_bits) - 1)
        steps = torch.arange(num_bins, device=device, dtype=torch.float32)
        table = torch.round(
            torch.exp(-steps * float(self.intattention_zero_thr) / float(num_bins - 1)) * qmax
        )
        table[0] = qmax
        table[-1] = 0.0
        return table

    def _forward_intattention_tensor(self, scores, attn_mask=None):
        scores = scores.float()
        if attn_mask is not None:
            scores = scores + attn_mask.to(device=scores.device, dtype=torch.float32)

        valid_mask = self._valid_mask(scores, attn_mask)
        masked_scores = scores.masked_fill(~valid_mask, -float("inf"))
        max_per_row = torch.max(masked_scores, dim=self.dim, keepdim=True)[0]
        max_per_row = torch.where(
            torch.isfinite(max_per_row),
            max_per_row,
            torch.zeros_like(max_per_row),
        )

        distance = (max_per_row - scores).masked_fill(~valid_mask, 0.0)
        distance = distance.clamp(min=0.0, max=float(self.intattention_zero_thr))
        num_bins = (1 << self.intattention_exp_bits) - 1
        indices = torch.round(
            distance * float(num_bins) / float(self.intattention_zero_thr)
        ).to(torch.long).clamp_(0, num_bins)
        exp_q = self._intattention_exp_table(scores.device)[indices].masked_fill(~valid_mask, 0.0)

        out_qmax = float((1 << self.intattention_output_bits) - 1)
        denom = exp_q.sum(dim=self.dim, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        probs_q = torch.round(exp_q * out_qmax / denom).clamp_(0.0, out_qmax)
        probs = probs_q / out_qmax
        return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    def _forward_intattention(self, attn_weights, attn_mask=None, out_dtype=None):
        softmax_dim = self.dim if self.dim >= 0 else attn_weights.dim() + self.dim
        if softmax_dim != attn_weights.dim() - 1 or attn_weights.dim() < 2:
            out = self._forward_intattention_tensor(attn_weights, attn_mask)
            return out.to(out_dtype) if out_dtype is not None else out

        query_dim = attn_weights.dim() - 2
        query_len = attn_weights.shape[query_dim]
        chunk_size = self.scna_query_chunk_size
        if query_len <= chunk_size:
            out = self._forward_intattention_tensor(attn_weights, attn_mask)
            return out.to(out_dtype) if out_dtype is not None else out

        chunks = []
        for start in range(0, query_len, chunk_size):
            end = min(start + chunk_size, query_len)
            attn_slice = [slice(None)] * attn_weights.dim()
            attn_slice[query_dim] = slice(start, end)
            mask_chunk = self._slice_attn_mask(attn_mask, query_dim, start, end)
            out_chunk = self._forward_intattention_tensor(attn_weights[tuple(attn_slice)], mask_chunk)
            if out_dtype is not None:
                out_chunk = out_chunk.to(out_dtype)
            chunks.append(out_chunk)
        return torch.cat(chunks, dim=query_dim)

    def _scna_dynamic_quant_dequant(self, x, valid_mask):
        bits = self.scna_input_quant_bits
        if bits <= 0:
            return x

        qmax = float((1 << bits) - 1)
        finite = x.masked_fill(~valid_mask, 0.0)
        xmin = finite.masked_fill(~valid_mask, float("inf")).amin(dim=self.dim, keepdim=True)
        xmax = finite.masked_fill(~valid_mask, -float("inf")).amax(dim=self.dim, keepdim=True)
        zeros = torch.zeros_like(xmin)
        xmin = torch.where(torch.isfinite(xmin), xmin, zeros).clamp(max=0.0)
        xmax = torch.where(torch.isfinite(xmax), xmax, zeros).clamp(min=0.0)
        scale = ((xmax - xmin).clamp_min(1e-6)) / qmax
        zero_point = torch.round(-xmin / scale).clamp(0, qmax)
        safe_x = x.masked_fill(~valid_mask, 0.0)
        q = torch.round(safe_x / scale + zero_point).clamp(0, qmax)
        dequant = (q - zero_point) * scale
        return dequant.masked_fill(~valid_mask, -float("inf"))

    def _condition_scna_input(self, shifted, valid_mask):
        scna_input = shifted
        if self.scna_input_clip_min is not None:
            clipped = scna_input.clamp_min(float(self.scna_input_clip_min))
            scna_input = torch.where(valid_mask, clipped, scna_input)
        if self.scna_input_scale != 1.0:
            scaled = scna_input * float(self.scna_input_scale)
            scna_input = torch.where(valid_mask, scaled, scna_input)
        return self._scna_dynamic_quant_dequant(scna_input, valid_mask)

    def _forward_scna_tensor(self, scores, attn_mask=None):
        scores = scores.float()
        if attn_mask is not None:
            scores = scores + attn_mask.to(device=scores.device, dtype=torch.float32)

        valid_mask = self._valid_mask(scores, attn_mask)
        masked_scores = scores.masked_fill(~valid_mask, -float("inf"))
        max_per_row = torch.max(masked_scores, dim=self.dim, keepdim=True)[0]
        max_per_row = torch.where(
            torch.isfinite(max_per_row),
            max_per_row,
            torch.zeros_like(max_per_row),
        )
        shifted = (scores - max_per_row).masked_fill(~valid_mask, -float("inf"))
        scna_input = self._condition_scna_input(shifted, valid_mask)
        exp_approx = self.scna_exp(scna_input).masked_fill(~valid_mask, 0.0).clamp_min(0.0)
        if self.scna_output_floor_log is not None:
            floor = math.exp(float(self.scna_output_floor_log))
            exp_approx = torch.where(valid_mask, exp_approx.clamp_min(floor), exp_approx)
        denom = exp_approx.sum(dim=self.dim, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        return torch.nan_to_num(exp_approx / denom, nan=0.0, posinf=0.0, neginf=0.0)

    def _forward_scna(self, attn_weights, attn_mask=None, out_dtype=None):
        softmax_dim = self.dim if self.dim >= 0 else attn_weights.dim() + self.dim
        if softmax_dim != attn_weights.dim() - 1 or attn_weights.dim() < 2:
            out = self._forward_scna_tensor(attn_weights, attn_mask)
            return out.to(out_dtype) if out_dtype is not None else out

        query_dim = attn_weights.dim() - 2
        query_len = attn_weights.shape[query_dim]
        chunk_size = self.scna_query_chunk_size
        if query_len <= chunk_size:
            out = self._forward_scna_tensor(attn_weights, attn_mask)
            return out.to(out_dtype) if out_dtype is not None else out

        chunks = []
        for start in range(0, query_len, chunk_size):
            end = min(start + chunk_size, query_len)
            attn_slice = [slice(None)] * attn_weights.dim()
            attn_slice[query_dim] = slice(start, end)
            mask_chunk = self._slice_attn_mask(attn_mask, query_dim, start, end)
            out_chunk = self._forward_scna_tensor(attn_weights[tuple(attn_slice)], mask_chunk)
            if out_dtype is not None:
                out_chunk = out_chunk.to(out_dtype)
            chunks.append(out_chunk)
        return torch.cat(chunks, dim=query_dim)


class QuantMul(H):
    def __init__(
        self, act_quant_param=dict(bits=8, sym=True, dynamic_method="pertoken")
    ):
        super().__init__()
        self.act_quantizer = Quantizer(**act_quant_param)
        self.use_act_quant = False

    def forward(self, x1, x2):
        out = x1 * x2
        
        
        out = self.may_hadamard_transform(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantSiLU(H):
    def __init__(
        self, act_quant_params=dict(bits=8, sym=True, dynamic_method="pertoken")
    ):
        super().__init__()
        self.act_func = F.silu
        self.act_quantizer = Quantizer(**act_quant_params)
        self.use_act_quant = False
        self.smooth = None
        self.temporary = False

    def forward(self, x, S_up_gate=None, **kwargs):
        if self.temporary:
            self.smooth = S_up_gate
        if self.smooth is None:
            out = F.silu(x)
        else:
            out = x * F.sigmoid(x / self.smooth.to(x.device))
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantEmbedding(nn.Embedding):
    def __init__(self, ori: nn.Embedding, act_quant_params=dict(bits=16)):
        super().__init__(
            num_embeddings=ori.num_embeddings,
            embedding_dim=ori.embedding_dim,
            padding_idx=ori.padding_idx,
            max_norm=ori.max_norm,
            norm_type=ori.norm_type,
            scale_grad_by_freq=ori.scale_grad_by_freq,
            sparse=ori.sparse,
            _weight=ori.weight,
            _freeze=True,
            device=ori.weight.device,
            dtype=ori.weight.dtype,
        )
        self.temporary = False
        del self.weight
        self.register_buffer("weight", ori.weight.data)
        self.act_quantizer = Quantizer(**act_quant_params)
        self.use_act_quant = False

    def forward(self, input: torch.Tensor, R_res=None) -> torch.Tensor:
        out = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        if self.temporary and R_res is not None:
            ori_dtype = out.dtype
            out = (
                (out.to(Rdtype))
                @ (R_res.to(dtype=Rdtype, device=out.device))
            ).to(ori_dtype)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out
