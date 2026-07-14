# Analysis

- Experiment: `prefill_attention_aws_v2_v4_sweep`
- Longest context: `32768`
- HBM note: this run forces effectively infinite HBM bandwidth during simulation. The required-HBM columns below report the bandwidth needed to keep HBM off the critical path for the chosen schedule.
- TPU note: this run maps the modeled global-buffer capacity to device-memory capacity for compile-time tile feasibility.

## Longest-Context Summary

| Case | Dtype | Tensor/Vector | Baseline ms | Flash ms | CustomSA ms | Flash/CustomSA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AWSv2 FP16 | fp16 | 44.04 | 202.781 | 186.193 | 140.686 | 1.323x |
| AWSv2 INT8 | int8 | 44.04 | 154.131 | 92.893 | 102.357 | 0.908x |
| AWSv3 FP16 | fp16 | 67.70 | 174.903 | 53.664 | 40.081 | 1.339x |
| AWSv3 INT8 | int8 | 67.70 | 119.575 | 38.755 | 28.575 | 1.356x |
| AWSv4 FP16 | fp16 | 68.27 | 166.242 | 53.330 | 39.833 | 1.339x |
| AWSv4 INT8 | int8 | 68.27 | 85.716 | 36.210 | 28.399 | 1.275x |

## Required HBM Bandwidth

| Case | Configured HBM (TB/s) | Flash Required (TB/s) | CustomSA Required (TB/s) | Flash / Config | CustomSA / Config |
| --- | ---: | ---: | ---: | ---: | ---: |
| AWSv2 FP16 | 0.820 | 0.742 | 0.983 | 0.905x | 1.199x |
| AWSv2 INT8 | 0.820 | 0.745 | 0.676 | 0.909x | 0.824x |
| AWSv3 FP16 | 2.900 | 0.654 | 0.877 | 0.225x | 0.302x |
| AWSv3 INT8 | 2.900 | 1.794 | 2.440 | 0.619x | 0.841x |
| AWSv4 FP16 | 4.900 | 0.658 | 0.883 | 0.134x | 0.180x |
| AWSv4 INT8 | 4.900 | 1.921 | 2.455 | 0.392x | 0.501x |

## Longest-Context Bottlenecks

| Case | Variant | Total ms | Bottleneck | Bottleneck ms | Share |
| --- | --- | ---: | --- | ---: | ---: |
| AWSv2 FP16 | Unfused Attention | 202.781 | softmax | 104.687 | 51.63% |
| AWSv2 FP16 | FlashAttention | 186.193 | matmul_pair | 185.893 | 99.84% |
| AWSv2 FP16 | FlashAttention-CustomSA | 140.686 | fused_core | 140.386 | 99.79% |
| AWSv2 INT8 | Unfused Attention | 154.131 | softmax | 104.687 | 67.92% |
| AWSv2 INT8 | FlashAttention | 92.893 | matmul_pair | 92.593 | 99.68% |
| AWSv2 INT8 | FlashAttention-CustomSA | 102.357 | fused_core | 102.057 | 99.71% |
| AWSv3 FP16 | Unfused Attention | 174.903 | softmax | 63.360 | 36.23% |
| AWSv3 FP16 | FlashAttention | 53.664 | matmul_pair | 53.364 | 99.44% |
| AWSv3 FP16 | FlashAttention-CustomSA | 40.081 | fused_core | 39.781 | 99.25% |
| AWSv3 INT8 | Unfused Attention | 119.575 | softmax | 63.360 | 52.99% |
| AWSv3 INT8 | FlashAttention | 38.755 | softmax | 38.455 | 99.23% |
| AWSv3 INT8 | FlashAttention-CustomSA | 28.575 | onchip_io | 28.275 | 98.95% |
| AWSv4 FP16 | Unfused Attention | 166.242 | a_mul_v | 55.484 | 33.38% |
| AWSv4 FP16 | FlashAttention | 53.330 | matmul_pair | 53.030 | 99.44% |
| AWSv4 FP16 | FlashAttention-CustomSA | 39.833 | fused_core | 39.533 | 99.25% |
| AWSv4 INT8 | Unfused Attention | 85.716 | softmax | 55.388 | 64.62% |
| AWSv4 INT8 | FlashAttention | 36.210 | softmax | 35.910 | 99.17% |
| AWSv4 INT8 | FlashAttention-CustomSA | 28.399 | onchip_io | 28.099 | 98.94% |

## Interpretation

- `AWSv2 FP16` at context 32768: `baseline` is dominated by `softmax` (104.687 ms), `flashattention` is dominated by `matmul_pair` (185.893 ms), and `customsa` is dominated by `fused_core` (140.386 ms).
- `AWSv2 INT8` at context 32768: `baseline` is dominated by `softmax` (104.687 ms), `flashattention` is dominated by `matmul_pair` (92.593 ms), and `customsa` is dominated by `fused_core` (102.057 ms).
- `AWSv3 FP16` at context 32768: `baseline` is dominated by `softmax` (63.360 ms), `flashattention` is dominated by `matmul_pair` (53.364 ms), and `customsa` is dominated by `fused_core` (39.781 ms).
- `AWSv3 INT8` at context 32768: `baseline` is dominated by `softmax` (63.360 ms), `flashattention` is dominated by `softmax` (38.455 ms), and `customsa` is dominated by `onchip_io` (28.275 ms).
- `AWSv4 FP16` at context 32768: `baseline` is dominated by `a_mul_v` (55.484 ms), `flashattention` is dominated by `matmul_pair` (53.030 ms), and `customsa` is dominated by `fused_core` (39.533 ms).
- `AWSv4 INT8` at context 32768: `baseline` is dominated by `softmax` (55.388 ms), `flashattention` is dominated by `softmax` (35.910 ms), and `customsa` is dominated by `onchip_io` (28.099 ms).

Treat the component split as a directional bottleneck study, not a calibrated vendor-accurate timing model.
