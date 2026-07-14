# Analysis

- Experiment: `prefill_attention_tpu_v3_v6_sweep`
- Longest context: `32768`
- HBM note: this run forces effectively infinite HBM bandwidth during simulation. The required-HBM columns below report the bandwidth needed to keep HBM off the critical path for the chosen schedule.
- TPU note: this run maps the modeled global-buffer capacity to device-memory capacity for compile-time tile feasibility.

## Longest-Context Summary

| Case | Dtype | Tensor/Vector | Baseline ms | Flash ms | CustomSA ms | Flash/CustomSA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TPUv3 FP16 | fp16 | 32.00 | 418.475 | 289.146 | 216.778 | 1.334x |
| TPUv3 INT8 | int8 | 32.00 | 306.152 | 143.634 | 108.539 | 1.323x |
| TPUv4 FP16 | fp16 | 64.00 | 362.271 | 129.593 | 96.685 | 1.340x |
| TPUv4 INT8 | int8 | 64.00 | 356.981 | 128.610 | 96.427 | 1.334x |
| TPUv5e FP16 | fp16 | 32.00 | 514.407 | 181.310 | 135.239 | 1.341x |
| TPUv5e INT8 | int8 | 32.00 | 374.148 | 90.123 | 67.769 | 1.330x |
| TPUv5p FP16 | fp16 | 32.00 | 175.651 | 77.876 | 58.131 | 1.340x |
| TPUv5p INT8 | int8 | 32.00 | 115.892 | 38.795 | 29.215 | 1.328x |
| TPUv6e FP16 | fp16 | 128.00 | 214.050 | 69.646 | 41.082 | 1.695x |
| TPUv6e INT8 | int8 | 128.00 | 180.116 | 38.877 | 20.691 | 1.879x |

## Required HBM Bandwidth

| Case | Configured HBM (TB/s) | Flash Required (TB/s) | CustomSA Required (TB/s) | Flash / Config | CustomSA / Config |
| --- | ---: | ---: | ---: | ---: | ---: |
| TPUv3 FP16 | 0.900 | 0.240 | 0.320 | 0.266x | 0.355x |
| TPUv3 INT8 | 0.900 | 0.242 | 0.320 | 0.268x | 0.355x |
| TPUv4 FP16 | 1.200 | 0.270 | 0.362 | 0.225x | 0.302x |
| TPUv4 INT8 | 1.200 | 0.136 | 0.182 | 0.113x | 0.151x |
| TPUv5e FP16 | 0.859 | 0.193 | 0.259 | 0.224x | 0.301x |
| TPUv5e INT8 | 0.859 | 0.194 | 0.259 | 0.226x | 0.301x |
| TPUv5p FP16 | 2.765 | 0.450 | 0.603 | 0.163x | 0.218x |
| TPUv5p INT8 | 2.765 | 0.453 | 0.603 | 0.164x | 0.218x |
| TPUv6e FP16 | 1.600 | 0.503 | 0.856 | 0.315x | 0.535x |
| TPUv6e INT8 | 1.600 | 0.452 | 0.856 | 0.283x | 0.535x |

## Longest-Context Bottlenecks

| Case | Variant | Total ms | Bottleneck | Bottleneck ms | Share |
| --- | --- | ---: | --- | ---: | ---: |
| TPUv3 FP16 | Unfused Attention | 418.475 | softmax | 192.574 | 46.02% |
| TPUv3 FP16 | FlashAttention | 289.146 | matmul_pair | 288.846 | 99.90% |
| TPUv3 FP16 | FlashAttention-CustomSA | 216.778 | fused_core | 216.478 | 99.86% |
| TPUv3 INT8 | Unfused Attention | 306.152 | softmax | 192.574 | 62.90% |
| TPUv3 INT8 | FlashAttention | 143.634 | matmul_pair | 143.334 | 99.79% |
| TPUv3 INT8 | FlashAttention-CustomSA | 108.539 | fused_core | 108.239 | 99.72% |
| TPUv4 FP16 | Unfused Attention | 362.271 | softmax | 161.404 | 44.55% |
| TPUv4 FP16 | FlashAttention | 129.593 | matmul_pair | 129.293 | 99.77% |
| TPUv4 FP16 | FlashAttention-CustomSA | 96.685 | fused_core | 96.385 | 99.69% |
| TPUv4 INT8 | Unfused Attention | 356.981 | softmax | 161.404 | 45.21% |
| TPUv4 INT8 | FlashAttention | 128.610 | matmul_pair | 128.310 | 99.77% |
| TPUv4 INT8 | FlashAttention-CustomSA | 96.427 | fused_core | 96.127 | 99.69% |
| TPUv5e FP16 | Unfused Attention | 514.407 | softmax | 233.367 | 45.37% |
| TPUv5e FP16 | FlashAttention | 181.310 | matmul_pair | 181.010 | 99.83% |
| TPUv5e FP16 | FlashAttention-CustomSA | 135.239 | fused_core | 134.939 | 99.78% |
| TPUv5e INT8 | Unfused Attention | 374.148 | softmax | 233.367 | 62.37% |
| TPUv5e INT8 | FlashAttention | 90.123 | matmul_pair | 89.823 | 99.67% |
| TPUv5e INT8 | FlashAttention-CustomSA | 67.769 | fused_core | 67.469 | 99.56% |
| TPUv5p FP16 | Unfused Attention | 175.651 | a_mul_v | 76.920 | 43.79% |
| TPUv5p FP16 | FlashAttention | 77.876 | matmul_pair | 77.576 | 99.61% |
| TPUv5p FP16 | FlashAttention-CustomSA | 58.131 | fused_core | 57.831 | 99.48% |
| TPUv5p INT8 | Unfused Attention | 115.892 | softmax | 56.340 | 48.61% |
| TPUv5p INT8 | FlashAttention | 38.795 | matmul_pair | 38.495 | 99.23% |
| TPUv5p INT8 | FlashAttention-CustomSA | 29.215 | fused_core | 28.915 | 98.97% |
| TPUv6e FP16 | Unfused Attention | 214.050 | softmax | 148.533 | 69.39% |
| TPUv6e FP16 | FlashAttention | 69.646 | matmul_pair | 69.346 | 99.57% |
| TPUv6e FP16 | FlashAttention-CustomSA | 41.082 | fused_core | 40.782 | 99.27% |
| TPUv6e INT8 | Unfused Attention | 180.116 | softmax | 148.533 | 82.46% |
| TPUv6e INT8 | FlashAttention | 38.877 | softmax | 38.577 | 99.23% |
| TPUv6e INT8 | FlashAttention-CustomSA | 20.691 | fused_core | 20.391 | 98.55% |

## Interpretation

- `TPUv3 FP16` at context 32768: `baseline` is dominated by `softmax` (192.574 ms), `flashattention` is dominated by `matmul_pair` (288.846 ms), and `customsa` is dominated by `fused_core` (216.478 ms).
- `TPUv3 INT8` at context 32768: `baseline` is dominated by `softmax` (192.574 ms), `flashattention` is dominated by `matmul_pair` (143.334 ms), and `customsa` is dominated by `fused_core` (108.239 ms).
- `TPUv5e FP16` at context 32768: `baseline` is dominated by `softmax` (233.367 ms), `flashattention` is dominated by `matmul_pair` (181.010 ms), and `customsa` is dominated by `fused_core` (134.939 ms).
- `TPUv5e INT8` at context 32768: `baseline` is dominated by `softmax` (233.367 ms), `flashattention` is dominated by `matmul_pair` (89.823 ms), and `customsa` is dominated by `fused_core` (67.469 ms).
- `TPUv5p FP16` at context 32768: `baseline` is dominated by `a_mul_v` (76.920 ms), `flashattention` is dominated by `matmul_pair` (77.576 ms), and `customsa` is dominated by `fused_core` (57.831 ms).
- `TPUv5p INT8` at context 32768: `baseline` is dominated by `softmax` (56.340 ms), `flashattention` is dominated by `matmul_pair` (38.495 ms), and `customsa` is dominated by `fused_core` (28.915 ms).
- `TPUv4 FP16` at context 32768: `baseline` is dominated by `softmax` (161.404 ms), `flashattention` is dominated by `matmul_pair` (129.293 ms), and `customsa` is dominated by `fused_core` (96.385 ms).
- `TPUv4 INT8` at context 32768: `baseline` is dominated by `softmax` (161.404 ms), `flashattention` is dominated by `matmul_pair` (128.310 ms), and `customsa` is dominated by `fused_core` (96.127 ms).
- `TPUv6e FP16` at context 32768: `baseline` is dominated by `softmax` (148.533 ms), `flashattention` is dominated by `matmul_pair` (69.346 ms), and `customsa` is dominated by `fused_core` (40.782 ms).
- `TPUv6e INT8` at context 32768: `baseline` is dominated by `softmax` (148.533 ms), `flashattention` is dominated by `softmax` (38.577 ms), and `customsa` is dominated by `fused_core` (20.391 ms).

Treat the component split as a directional bottleneck study, not a calibrated vendor-accurate timing model.
