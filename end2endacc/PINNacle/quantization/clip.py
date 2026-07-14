import torch

def fp_scale(tensor, S, M, bias, max_float, min_float):
    tensor_unscaled = (tensor / S)
    tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
    tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (tensor_log_scales - M - bias)
    tensor_q = (tensor_unscaled / scales).round()
    tensor_q = tensor_q * scales
    return tensor_q

@torch.no_grad()
def pseudo_quantize_tensor_clip(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False, 
                           fpq=False, mantissa_bit=-1, clip=False, maxshrink=0.8, grid=100, norm=2.4):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    # orig_dtype = tensor.dtype
    # tensor = tensor.to(torch.float)
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    
    # Initialize quantization parameters
    if fpq:
        M = mantissa_bit
        E = n_bits - 1 - M
        bias = 2 ** (E - 1) - 1
        max_float = (2 - 2 ** (-M)) * 2 ** (2**E - 1 - bias)
        min_float = -max_float
        
        if zero_point:
            max_val = tensor.amax(dim=1, keepdim=True)
            min_val = tensor.amin(dim=1, keepdim=True)
            zeros = (max_val + min_val) / 2
            tensor_centered = tensor - zeros
            max_val = tensor_centered.abs().amax(dim=1, keepdim=True)
            S = max_val / max_float
        else:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
            S = max_val / max_float
            zeros = torch.zeros_like(S)
            tensor_centered = tensor
    else:
        if zero_point:
            max_val = tensor.amax(dim=1, keepdim=True)
            min_val = tensor.amin(dim=1, keepdim=True)
            max_int = 2**n_bits - 1
            min_int = 0
            scales = (max_val - min_val) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
            # max_val = max_val
            max_int = 2 ** (n_bits - 1) - 1
            min_int = -max_int
            scales = max_val / max_int
            zeros = torch.zeros_like(scales)
    
    # Apply MSE clipping if enabled
    if clip:
        best_err = torch.full([tensor.shape[0]], float('inf'), device=tensor.device, dtype=tensor.dtype)
        if fpq:
            best_S = S.clone()
            best_zeros = zeros.clone()
            
            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid
                
                if zero_point:
                    # Scale down the range
                    max_val_clip = p * tensor.amax(dim=1, keepdim=True)
                    min_val_clip = p * tensor.amin(dim=1, keepdim=True)
                    zeros_clip = (max_val_clip + min_val_clip) / 2
                    tensor_centered_clip = tensor - zeros_clip
                    max_val_clip = tensor_centered_clip.abs().amax(dim=1, keepdim=True)
                    S_clip = max_val_clip / max_float
                    
                    # Quantize and dequantize
                    tensor_q_clip = fp_scale(tensor_centered_clip, S_clip, M, bias, max_float, min_float)
                    tensor_reconstructed = tensor_q_clip * S_clip + zeros_clip
                else:
                    max_val_clip = p * tensor.abs().amax(dim=1, keepdim=True)
                    S_clip = max_val_clip / max_float
                    
                    tensor_q_clip = fp_scale(tensor, S_clip, M, bias, max_float, min_float)
                    tensor_reconstructed = tensor_q_clip * S_clip
                
                # Calculate error
                err = (tensor_reconstructed - tensor).abs().pow_(norm).sum(dim=1)
                
                # Update best parameters
                improve_mask = err < best_err
                if torch.any(improve_mask):
                    best_err[improve_mask] = err[improve_mask]
                    best_S[improve_mask] = S_clip[improve_mask]
                    if zero_point:
                        best_zeros[improve_mask] = zeros_clip[improve_mask]
            
            # Apply best quantization
            if zero_point:
                tensor_centered = tensor - best_zeros
                tensor_q = fp_scale(tensor_centered, best_S, M, bias, max_float, min_float)
                tensor = tensor_q * best_S + best_zeros
            else:
                tensor_q = fp_scale(tensor, best_S, M, bias, max_float, min_float)
                tensor = tensor_q * best_S
                
        else:  # Regular integer quantization with clipping
            best_scales = scales.clone()
            best_zeros = zeros.clone()
            
            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid
                
                if zero_point:
                    max_val_clip = p * tensor.amax(dim=1, keepdim=True)
                    min_val_clip = p * tensor.amin(dim=1, keepdim=True)
                    scales_clip = (max_val_clip - min_val_clip).clamp(min=1e-5) / max_int
                    zeros_clip = (-torch.round(min_val_clip / scales_clip)).clamp_(min_int, max_int)
                    
                    tensor_reconstructed = (torch.clamp(torch.round(tensor / scales_clip) + zeros_clip, 
                                                      min_int, max_int) - zeros_clip) * scales_clip
                else:
                    max_val_clip = p * tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                    scales_clip = max_val_clip / max_int
                    
                    tensor_reconstructed = torch.clamp(torch.round(tensor / scales_clip), 
                                                     min_int, max_int) * scales_clip
                
                # Calculate error
                err = (tensor_reconstructed - tensor).abs().pow_(norm).sum(dim=1)
                
                # Update best parameters
                improve_mask = err < best_err
                if torch.any(improve_mask):
                    best_err[improve_mask] = err[improve_mask]
                    best_scales[improve_mask] = scales_clip[improve_mask]
                    if zero_point:
                        best_zeros[improve_mask] = zeros_clip[improve_mask]
            
            # Apply best quantization
            if inplace:
                (tensor.div_(best_scales).round_().add_(best_zeros)).clamp_(min_int, max_int).sub_(best_zeros).mul_(best_scales)
            else:
                tensor = (torch.clamp(torch.round(tensor / best_scales) + best_zeros, 
                                    min_int, max_int) - best_zeros) * best_scales
    else:
        # Original quantization without clipping
        if fpq:
            if zero_point:
                org_dtype = tensor.dtype
                tensor = tensor.to(torch.float)
                tensor_q = fp_scale(tensor_centered, S, M, bias, max_float, min_float)
                tensor = tensor_q * S + zeros
                tensor = tensor.to(org_dtype)
            else:
                # convert to float to prevent inf
                org_dtype = tensor.dtype
                tensor = tensor.to(torch.float)
                tensor_q = fp_scale(tensor, S, M, bias, max_float, min_float)
                tensor = tensor_q * S
                tensor = tensor.to(org_dtype)
        else:
            if inplace:
                (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros).mul_(scales)
            else:
                tensor = (torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros) * scales

    assert torch.isnan(tensor).sum() == 0
    tensor = tensor.reshape(org_tensor_shape)
    # tensor = tensor.to(orig_dtype)
    return tensor

