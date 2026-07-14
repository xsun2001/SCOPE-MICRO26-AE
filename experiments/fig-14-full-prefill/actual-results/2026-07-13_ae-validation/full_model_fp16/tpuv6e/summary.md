# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `reproduced-results/2026-07-13_ae-validation/performance/fixed_tiles_fp16/tpuv6e`
- Attention case: `tpuv6e_fp16`
- System: `Google TPU v6e (official per-chip model) single-card`
- Tensor throughput: `917.5040` TFLOP/s
- Vector throughput: `7.1680` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

- Baseline for speedups: `Unfused Attention`

| Context | Variant | Model ms | Tok/s | Attn core ms | Attn TFLOP/s | Model speedup | Attn-core speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | Unfused Attention | 112.149 | 18261.470 | 1.212 | 56.696 | 1.00000x | 1.00000x |
| 2048 | FlashAttention | 91.929 | 22277.941 | 0.580 | 118.439 | 1.21994x | 2.08900x |
| 2048 | FlashAttention-CustomSA | 88.365 | 23176.647 | 0.469 | 146.582 | 1.26916x | 2.58538x |
| 4096 | Unfused Attention | 213.556 | 19179.988 | 3.246 | 84.674 | 1.00000x | 1.00000x |
| 4096 | FlashAttention | 153.948 | 26606.452 | 1.384 | 198.677 | 1.38720x | 2.34637x |
| 4096 | FlashAttention-CustomSA | 139.665 | 29327.224 | 0.937 | 293.291 | 1.52905x | 3.46375x |
| 8192 | Unfused Attention | 545.219 | 15025.144 | 11.341 | 96.948 | 1.00000x | 1.00000x |
| 8192 | FlashAttention | 330.591 | 24779.898 | 4.634 | 237.263 | 1.64923x | 2.44733x |
| 8192 | FlashAttention-CustomSA | 273.462 | 29956.643 | 2.849 | 385.945 | 1.99377x | 3.98097x |
| 16384 | Unfused Attention | 1723.942 | 9503.798 | 43.637 | 100.786 | 1.00000x | 1.00000x |
| 16384 | FlashAttention | 891.916 | 18369.448 | 17.637 | 249.370 | 1.93285x | 2.47425x |
| 16384 | FlashAttention-CustomSA | 663.401 | 24696.975 | 10.496 | 419.040 | 2.59864x | 4.15772x |
| 32768 | Unfused Attention | 6142.973 | 5334.225 | 172.654 | 101.893 | 1.00000x | 1.00000x |
| 32768 | FlashAttention | 2846.724 | 11510.777 | 69.646 | 252.593 | 2.15791x | 2.47901x |
| 32768 | FlashAttention-CustomSA | 1932.665 | 16954.828 | 41.082 | 428.220 | 3.17850x | 4.20266x |
| 65536 | Unfused Attention | 23227.371 | 2821.499 | 688.386 | 102.223 | 1.00000x | 1.00000x |
| 65536 | FlashAttention | 10084.969 | 6498.384 | 277.686 | 253.411 | 2.30317x | 2.47901x |
| 65536 | FlashAttention-CustomSA | 6428.734 | 10194.231 | 163.428 | 430.578 | 3.61306x | 4.21215x |
| 131072 | Unfused Attention | 90381.515 | 1450.208 | 2750.641 | 102.331 | 1.00000x | 1.00000x |
| 131072 | FlashAttention | 37875.978 | 3460.558 | 1109.843 | 253.617 | 2.38625x | 2.47841x |
| 131072 | FlashAttention-CustomSA | 23251.039 | 5637.253 | 652.814 | 431.172 | 3.88720x | 4.21352x |
| 262144 | Unfused Attention | 356631.199 | 735.056 | 10998.321 | 102.370 | 1.00000x | 1.00000x |
| 262144 | FlashAttention | 146716.069 | 1786.744 | 4438.473 | 253.668 | 2.43076x | 2.47795x |
| 262144 | FlashAttention-CustomSA | 88216.314 | 2971.605 | 2610.355 | 431.321 | 4.04269x | 4.21334x |
| 524288 | Unfused Attention | 1416896.149 | 370.026 | 43986.354 | 102.386 | 1.00000x | 1.00000x |
| 524288 | FlashAttention | 577428.548 | 907.970 | 17752.991 | 253.681 | 2.45380x | 2.47769x |
| 524288 | FlashAttention-CustomSA | 343429.526 | 1526.625 | 10440.522 | 431.358 | 4.12573x | 4.21304x |
