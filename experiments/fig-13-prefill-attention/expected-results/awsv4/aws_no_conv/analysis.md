# Analysis

- Experiment: `prefill_attention_aws_v2_v4_sweep`
- Longest context: `32768`
- HBM note: this run forces effectively infinite HBM bandwidth during simulation. The required-HBM columns below report the bandwidth needed to keep HBM off the critical path for the chosen schedule.
- TPU note: this run maps the modeled global-buffer capacity to device-memory capacity for compile-time tile feasibility.

## Longest-Context Summary

| Case | Dtype | Tensor/Vector | Baseline ms | Flash ms | CustomSA ms | Flash/CustomSA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AWSv2 FP16 | fp16 | 44.04 | 202.781 | 186.193 | 140.686 | 1.323x |
| AWSv2 INT8 | int8 | 44.04 | 154.131 | 92.893 | 70.493 | 1.318x |
| AWSv3 FP16 | fp16 | 67.70 | 174.903 | 53.664 | 40.081 | 1.339x |
| AWSv3 INT8 | int8 | 67.70 | 119.575 | 26.781 | 20.191 | 1.326x |
| AWSv4 FP16 | fp16 | 68.27 | 166.242 | 53.330 | 39.833 | 1.339x |
| AWSv4 INT8 | int8 | 68.27 | 85.716 | 22.313 | 14.404 | 1.549x |

## Required HBM Bandwidth

| Case | Configured HBM (TB/s) | Flash Required (TB/s) | CustomSA Required (TB/s) | Flash / Config | CustomSA / Config |
| --- | ---: | ---: | ---: | ---: | ---: |
| AWSv2 FP16 | 0.820 | 0.742 | 0.983 | 0.905x | 1.199x |
| AWSv2 INT8 | 0.820 | 0.745 | 0.983 | 0.909x | 1.199x |
| AWSv3 FP16 | 2.900 | 0.654 | 0.877 | 0.225x | 0.302x |
| AWSv3 INT8 | 2.900 | 0.659 | 0.877 | 0.227x | 0.302x |
| AWSv4 FP16 | 4.900 | 0.658 | 0.883 | 0.134x | 0.180x |
| AWSv4 INT8 | 4.900 | 1.573 | 2.455 | 0.321x | 0.501x |

## Longest-Context Bottlenecks

| Case | Variant | Total ms | Bottleneck | Bottleneck ms | Share |
| --- | --- | ---: | --- | ---: | ---: |
| AWSv2 FP16 | Unfused Attention | 202.781 | softmax | 104.687 | 51.63% |
| AWSv2 FP16 | FlashAttention | 186.193 | matmul_pair | 185.893 | 99.84% |
| AWSv2 FP16 | FlashAttention-CustomSA | 140.686 | fused_core | 140.386 | 99.79% |
| AWSv2 INT8 | Unfused Attention | 154.131 | softmax | 104.687 | 67.92% |
| AWSv2 INT8 | FlashAttention | 92.893 | matmul_pair | 92.593 | 99.68% |
| AWSv2 INT8 | FlashAttention-CustomSA | 70.493 | fused_core | 70.193 | 99.57% |
| AWSv3 FP16 | Unfused Attention | 174.903 | softmax | 63.360 | 36.23% |
| AWSv3 FP16 | FlashAttention | 53.664 | matmul_pair | 53.364 | 99.44% |
| AWSv3 FP16 | FlashAttention-CustomSA | 40.081 | fused_core | 39.781 | 99.25% |
| AWSv3 INT8 | Unfused Attention | 119.575 | softmax | 63.360 | 52.99% |
| AWSv3 INT8 | FlashAttention | 26.781 | matmul_pair | 26.481 | 98.88% |
| AWSv3 INT8 | FlashAttention-CustomSA | 20.191 | fused_core | 19.891 | 98.51% |
| AWSv4 FP16 | Unfused Attention | 166.242 | a_mul_v | 55.484 | 33.38% |
| AWSv4 FP16 | FlashAttention | 53.330 | matmul_pair | 53.030 | 99.44% |
| AWSv4 FP16 | FlashAttention-CustomSA | 39.833 | fused_core | 39.533 | 99.25% |
| AWSv4 INT8 | Unfused Attention | 85.716 | softmax | 55.388 | 64.62% |
| AWSv4 INT8 | FlashAttention | 22.313 | softmax | 22.013 | 98.66% |
| AWSv4 INT8 | FlashAttention-CustomSA | 14.404 | onchip_io | 14.104 | 97.92% |

## Interpretation

- `AWSv2 FP16` at context 32768: `baseline` is dominated by `softmax` (104.687 ms), `flashattention` is dominated by `matmul_pair` (185.893 ms), and `customsa` is dominated by `fused_core` (140.386 ms).
- `AWSv2 INT8` at context 32768: `baseline` is dominated by `softmax` (104.687 ms), `flashattention` is dominated by `matmul_pair` (92.593 ms), and `customsa` is dominated by `fused_core` (70.193 ms).
- `AWSv3 FP16` at context 32768: `baseline` is dominated by `softmax` (63.360 ms), `flashattention` is dominated by `matmul_pair` (53.364 ms), and `customsa` is dominated by `fused_core` (39.781 ms).
- `AWSv3 INT8` at context 32768: `baseline` is dominated by `softmax` (63.360 ms), `flashattention` is dominated by `matmul_pair` (26.481 ms), and `customsa` is dominated by `fused_core` (19.891 ms).
- `AWSv4 FP16` at context 32768: `baseline` is dominated by `a_mul_v` (55.484 ms), `flashattention` is dominated by `matmul_pair` (53.030 ms), and `customsa` is dominated by `fused_core` (39.533 ms).
- `AWSv4 INT8` at context 32768: `baseline` is dominated by `softmax` (55.388 ms), `flashattention` is dominated by `softmax` (22.013 ms), and `customsa` is dominated by `onchip_io` (14.104 ms).

Treat the component split as a directional bottleneck study, not a calibrated vendor-accurate timing model.
