# Analysis

- Experiment: `prefill_attention_a100_h100_b200_sweep`
- Longest context: `32768`
- Blackwell note: the public dense spec used here is HGX B200 per-GPU Blackwell, not a standalone B100 product sheet.
- HBM note: this run forces effectively infinite HBM bandwidth during simulation. The required-HBM columns below report the bandwidth needed to keep HBM off the critical path for the chosen schedule.

## Longest-Context Summary

| Case | Dtype | Tensor/Vector | Baseline ms | Flash ms | CustomSA ms | Flash/CustomSA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A100 FP16 | fp16 | 4.00 | 116.458 | 66.235 | 60.624 | 1.093x |
| A100 INT8 | int8 | 4.00 | 86.374 | 32.357 | 30.323 | 1.067x |
| B200 FP16 | fp16 | 32.00 | 38.318 | 11.556 | 8.607 | 1.343x |
| B200 INT8 | int8 | 32.00 | 33.900 | 11.181 | 4.314 | 2.592x |
| H100 FP16 | fp16 | 16.00 | 61.851 | 21.820 | 20.224 | 1.079x |
| H100 INT8 | int8 | 16.00 | 51.894 | 13.211 | 10.111 | 1.307x |

## Required HBM Bandwidth

| Case | Configured HBM (TB/s) | Flash Required (TB/s) | CustomSA Required (TB/s) | Flash / Config | CustomSA / Config |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100 FP16 | 2.039 | 16.613 | 18.152 | 8.148x | 8.902x |
| A100 INT8 | 2.039 | 17.010 | 18.152 | 8.342x | 8.902x |
| B200 FP16 | 8.000 | 47.706 | 64.091 | 5.963x | 8.011x |
| B200 INT8 | 8.000 | 24.655 | 64.091 | 3.082x | 8.011x |
| H100 FP16 | 3.350 | 50.463 | 54.450 | 15.064x | 16.254x |
| H100 INT8 | 3.350 | 20.861 | 27.268 | 6.227x | 8.140x |

## Longest-Context Bottlenecks

| Case | Variant | Total ms | Bottleneck | Bottleneck ms | Share |
| --- | --- | ---: | --- | ---: | ---: |
| A100 FP16 | Unfused Attention | 116.458 | softmax | 57.465 | 49.34% |
| A100 FP16 | FlashAttention | 66.235 | matmul_pair | 66.214 | 99.97% |
| A100 FP16 | FlashAttention-CustomSA | 60.624 | fused_core | 60.603 | 99.97% |
| A100 INT8 | Unfused Attention | 86.374 | softmax | 57.465 | 66.53% |
| A100 INT8 | FlashAttention | 32.357 | matmul_pair | 32.336 | 99.94% |
| A100 INT8 | FlashAttention-CustomSA | 30.323 | fused_core | 30.302 | 99.93% |
| B200 FP16 | Unfused Attention | 38.318 | softmax | 29.706 | 77.52% |
| B200 FP16 | FlashAttention | 11.556 | softmax | 11.535 | 99.82% |
| B200 FP16 | FlashAttention-CustomSA | 8.607 | fused_core | 8.586 | 99.76% |
| B200 INT8 | Unfused Attention | 33.900 | softmax | 29.706 | 87.63% |
| B200 INT8 | FlashAttention | 11.181 | softmax | 11.160 | 99.81% |
| B200 INT8 | FlashAttention-CustomSA | 4.314 | fused_core | 4.293 | 99.51% |
| H100 FP16 | Unfused Attention | 61.851 | softmax | 42.075 | 68.03% |
| H100 FP16 | FlashAttention | 21.820 | matmul_pair | 21.799 | 99.90% |
| H100 FP16 | FlashAttention-CustomSA | 20.224 | fused_core | 20.203 | 99.90% |
| H100 INT8 | Unfused Attention | 51.894 | softmax | 42.075 | 81.08% |
| H100 INT8 | FlashAttention | 13.211 | softmax | 13.190 | 99.84% |
| H100 INT8 | FlashAttention-CustomSA | 10.111 | fused_core | 10.090 | 99.79% |

## Interpretation

- `A100 FP16` at context 32768: `baseline` is dominated by `softmax` (57.465 ms), `flashattention` is dominated by `matmul_pair` (66.214 ms), and `customsa` is dominated by `fused_core` (60.603 ms).
- `A100 INT8` at context 32768: `baseline` is dominated by `softmax` (57.465 ms), `flashattention` is dominated by `matmul_pair` (32.336 ms), and `customsa` is dominated by `fused_core` (30.302 ms).
- `H100 FP16` at context 32768: `baseline` is dominated by `softmax` (42.075 ms), `flashattention` is dominated by `matmul_pair` (21.799 ms), and `customsa` is dominated by `fused_core` (20.203 ms).
- `H100 INT8` at context 32768: `baseline` is dominated by `softmax` (42.075 ms), `flashattention` is dominated by `softmax` (13.190 ms), and `customsa` is dominated by `fused_core` (10.090 ms).
- `B200 FP16` at context 32768: `baseline` is dominated by `softmax` (29.706 ms), `flashattention` is dominated by `softmax` (11.535 ms), and `customsa` is dominated by `fused_core` (8.586 ms).
- `B200 INT8` at context 32768: `baseline` is dominated by `softmax` (29.706 ms), `flashattention` is dominated by `softmax` (11.160 ms), and `customsa` is dominated by `fused_core` (4.293 ms).

Treat the component split as a directional bottleneck study, not a calibrated vendor-accurate timing model.
