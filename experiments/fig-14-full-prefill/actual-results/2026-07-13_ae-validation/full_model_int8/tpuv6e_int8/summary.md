# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `reproduced-results/2026-07-13_ae-validation/performance/fixed_tiles_int8/tpuv6e_int8`
- Attention case: `tpuv6e_int8`
- System: `Google TPU v6e (official per-chip model) single-card`
- Tensor throughput: `1835.0080` TFLOP/s
- Vector throughput: `7.1680` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

- Baseline for speedups: `Unfused Attention`

| Context | Variant | Model ms | Tok/s | Attn core ms | Attn TFLOP/s | Model speedup | Attn-core speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | Unfused Attention | 86.719 | 23616.602 | 0.960 | 71.553 | 1.00000x | 1.00000x |
| 2048 | FlashAttention | 72.891 | 28096.856 | 0.528 | 130.082 | 1.18971x | 1.81798x |
| 2048 | FlashAttention-CustomSA | 69.789 | 29345.405 | 0.431 | 159.307 | 1.24258x | 2.22643x |
| 4096 | Unfused Attention | 146.590 | 27941.824 | 2.240 | 122.732 | 1.00000x | 1.00000x |
| 4096 | FlashAttention | 113.511 | 36084.502 | 1.206 | 227.935 | 1.29142x | 1.85718x |
| 4096 | FlashAttention-CustomSA | 100.119 | 40911.485 | 0.787 | 349.085 | 1.46417x | 2.84429x |
| 8192 | Unfused Attention | 346.864 | 23617.313 | 7.315 | 150.314 | 1.00000x | 1.00000x |
| 8192 | FlashAttention | 237.894 | 34435.507 | 3.909 | 281.244 | 1.45806x | 1.87105x |
| 8192 | FlashAttention-CustomSA | 182.348 | 44925.201 | 2.174 | 505.842 | 1.90221x | 3.36524x |
| 16384 | Unfused Attention | 1069.535 | 15318.814 | 27.531 | 159.747 | 1.00000x | 1.00000x |
| 16384 | FlashAttention | 660.142 | 24818.886 | 14.738 | 298.419 | 1.62016x | 1.86807x |
| 16384 | FlashAttention-CustomSA | 437.957 | 37410.094 | 7.795 | 564.249 | 2.44210x | 3.53214x |
| 32768 | Unfused Attention | 3803.365 | 8615.528 | 108.230 | 162.545 | 1.00000x | 1.00000x |
| 32768 | FlashAttention | 2197.654 | 14910.442 | 58.051 | 303.046 | 1.73065x | 1.86438x |
| 32768 | FlashAttention-CustomSA | 1308.911 | 25034.544 | 30.278 | 581.021 | 2.90575x | 3.57453x |
| 65536 | Unfused Attention | 14424.988 | 4543.227 | 430.688 | 163.387 | 1.00000x | 1.00000x |
| 65536 | FlashAttention | 8044.739 | 8146.442 | 231.305 | 304.225 | 1.79310x | 1.86199x |
| 65536 | FlashAttention-CustomSA | 4489.767 | 14596.750 | 120.212 | 585.371 | 3.21286x | 3.58273x |
| 131072 | Unfused Attention | 56284.076 | 2328.758 | 1719.849 | 163.663 | 1.00000x | 1.00000x |
| 131072 | FlashAttention | 30827.150 | 4251.836 | 924.320 | 304.521 | 1.82580x | 1.86066x |
| 131072 | FlashAttention-CustomSA | 16607.261 | 7892.452 | 479.949 | 586.469 | 3.38912x | 3.58340x |
| 262144 | Unfused Attention | 222465.624 | 1178.357 | 6875.152 | 163.764 | 1.00000x | 1.00000x |
| 262144 | FlashAttention | 120744.938 | 2171.056 | 3696.381 | 304.595 | 1.84244x | 1.85997x |
| 262144 | FlashAttention-CustomSA | 63865.383 | 4104.634 | 1918.894 | 586.744 | 3.48335x | 3.58287x |
| 524288 | Unfused Attention | 884682.209 | 592.629 | 27493.679 | 163.805 | 1.00000x | 1.00000x |
| 524288 | FlashAttention | 477992.385 | 1096.854 | 14784.622 | 304.614 | 1.85083x | 1.85961x |
| 524288 | FlashAttention-CustomSA | 250474.162 | 2093.182 | 7674.678 | 586.813 | 3.53203x | 3.58239x |
