import torch
import torch.nn as nn

from .clip import fp_scale, pseudo_quantize_tensor_clip


class ActQuantizer(nn.Module):
    """
    Activation Quantizer that supports both dynamic and static quantization.

    - In 'dynamic' mode, it computes scales on-the-fly for each input.
    - In 'calibrate' mode, it observes the running maximum of the input tensor.
    - In 'static' mode, it uses a fixed scale calculated from the calibration phase.
    - In 'none' mode, it does nothing.
    """
    def __init__(self, n_bits=8, q_group_size=-1, per_tensor=False,
                 fpq=False, mantissa_bit=-1, clip=False,
                 clip_ratio=0.9,
                 mode='none', use_8bit_scale=False, emit_int_codes=False):
        super(ActQuantizer, self).__init__()
        self.n_bits = n_bits
        self.per_tensor = per_tensor
        self.q_group_size = q_group_size
        self.fpq = fpq
        self.M = mantissa_bit
        self.clip = clip
        self.clip_ratio = clip_ratio if clip else 1.0
        self.mode = mode
        self.use_8bit_scale = use_8bit_scale
        self.emit_int_codes = emit_int_codes

        self.register_buffer('scale', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('amax', torch.zeros(1, dtype=torch.float32))

        if self.fpq:
            E = float(self.n_bits - 1 - self.M)
            bias = 2 ** (E - 1) - 1
            max_float = (2 - 2 ** (-self.M)) * 2 ** (2**E - 1 - bias)
            self.register_buffer('bias', torch.tensor(bias, dtype=torch.float32))
            self.register_buffer('max_float', torch.tensor(max_float, dtype=torch.float32))
            self.register_buffer('min_float', torch.tensor(-max_float, dtype=torch.float32))
        else:
            self.register_buffer('bias', None)
            self.register_buffer('max_float', None)
            self.register_buffer('min_float', None)
            self.q_max = 2**(self.n_bits - 1) - 1
            self.q_min = -(2**(self.n_bits - 1))

    def extra_repr(self) -> str:
        return f"n_bits={self.n_bits}, per_tensor={self.per_tensor}, fpq={self.fpq}, \
            mantissa_bit={self.M}, clip={self.clip}, clip_ratio={self.clip_ratio}, mode={self.mode}, \
            emit_int_codes={self.emit_int_codes}"

    @torch.no_grad()
    def update_amax(self, x):
        """Update the running maximum absolute value of the input tensor."""
        if self.per_tensor:
            amax = x.abs().max()
            if amax <= 0:
                print("amax is zero:", amax)
                raise ValueError("amax is zero")
        else:
            raise NotImplementedError
            # Assuming channel is the last dimension
            reduce_dims = list(range(x.dim() - 1))
            amax = torch.amax(x.abs(), dim=reduce_dims)
            
        
        # Update the buffer with the maximum value seen so far
        if amax > self.amax:
            self.amax.copy_(amax)
            if self.amax.is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

    def update_scale(self, name):
        """
        Update the quantization scale based on the collected amax.
        This should be called after the calibration phase.
        """
        if self.amax <= 0.0:
            
            raise ValueError(f"In quantizer '{name}', self.amax is less than or equal to zero. Value: {self.amax.item()}")
        else:
            if self.fpq:
                self.scale = self.clip_ratio * self.amax / self.max_float
            else:
                self.scale = self.clip_ratio * self.amax / self.q_max
        print(f"Updated scale to: {self.scale.mean().item():.4f} from amax: {self.amax.mean().item():.4f}")
        if self.use_8bit_scale:
            self.apply_8bit_scale()
            print(f"Updated 8bit scale to: {self.scale.mean().item():.4f}")
        

    def apply_8bit_scale(self):
        scale = self.scale.detach().float().clamp(min=1e-8)
        reciprocal_code = torch.round(1 / scale).clamp(1, 127)
        reciprocal_scale = 1 / reciprocal_code

        direct_code = torch.round(scale).clamp(1, 127)
        direct_scale = direct_code

        self.scale = torch.where(scale <= 1.0, reciprocal_scale, direct_scale).to(self.scale.device)

    def forward(self, x):
        dtype = x.dtype
        # x = x.float()
        if self.mode == 'calibrate':
            self.update_amax(x)
            x = x.to(dtype)
            return x
        
        if self.mode == 'static':
            scale = self.scale.clamp(min=1e-8)
            if self.fpq:
                if self.emit_int_codes:
                    raise ValueError("FPQ activation quantization cannot emit integer codes.")
                q_x = fp_scale(
                    x,
                    scale,
                    self.M,
                    self.bias.to(device=x.device),
                    self.max_float.to(device=x.device),
                    self.min_float.to(device=x.device),
                )
                r_x = q_x * scale
            else:
                q_x = torch.round(x / scale).clamp(self.q_min, self.q_max)
                if self.emit_int_codes:
                    return q_x.to(torch.int32)
                r_x = q_x * scale
            r_x = r_x.to(dtype)
            return r_x

        if self.mode == 'dynamic': # have not ready to use now !!!
            raise NotImplementedError
            x = pseudo_quantize_tensor_clip(
                x, n_bits=self.n_bits, q_group_size=self.q_group_size, 
                per_tensor=self.per_tensor, fpq=self.fpq, mantissa_bit=self.M, 
                clip=self.clip,
            )
            return x.to(dtype)
        
        # If mode is 'none' or something else, do nothing
        return x
