import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from .....quantization.clip import pseudo_quantize_tensor_clip
from .....quantization.quantizer import ActQuantizer

class MlpSigmoidApproximator_d8(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(MlpSigmoidApproximator_d8, self).__init__()

        fc1_weight_raw = torch.empty(8, input_dim)
        nn.init.uniform_(fc1_weight_raw, a=-5, b=0)  
        self.fc1_weight_raw = nn.Parameter(fc1_weight_raw)

        fc1_bias_raw = torch.empty(8)
        nn.init.uniform_(fc1_bias_raw, a=-5, b=0)
        self.fc1_bias_raw = nn.Parameter(fc1_bias_raw)
        self.relu1 = nn.ReLU()

        fc2_weight_raw = torch.empty(output_dim, 8)
        nn.init.uniform_(fc2_weight_raw, a=-5, b=0)
        self.fc2_weight_raw = nn.Parameter(fc2_weight_raw)

    def forward(self, x, args):
        positive_fc1_weight = torch.exp(self.fc1_weight_raw).to(device=x.device, dtype=x.dtype)
        positive_fc1_bias = torch.exp(self.fc1_bias_raw).to(device=x.device, dtype=x.dtype)
        
        if args.quant_pinn:
            positive_fc1_weight = pseudo_quantize_tensor_clip(positive_fc1_weight, n_bits=args.w_bits, zero_point=args.w_zero_point, q_group_size=args.w_group_size, 
                                                              fpq=args.fpq, mantissa_bit=args.w_mantissa_bit, clip=args.w_clip, per_tensor=args.w_per_tensor)
        linear1_output = F.linear(x, positive_fc1_weight, positive_fc1_bias)
        h1 = self.relu1(linear1_output)
        
        positive_fc2_weight = torch.exp(self.fc2_weight_raw).to(device=x.device, dtype=x.dtype)
        if args.quant_pinn:
            positive_fc2_weight = pseudo_quantize_tensor_clip(positive_fc2_weight, n_bits=args.w_bits, zero_point=args.w_zero_point, q_group_size=args.w_group_size, 
                                                              fpq=args.fpq, mantissa_bit=args.w_mantissa_bit, clip=args.w_clip, per_tensor=args.w_per_tensor)
        output = F.linear(h1, positive_fc2_weight, bias=None)
        
        return output
