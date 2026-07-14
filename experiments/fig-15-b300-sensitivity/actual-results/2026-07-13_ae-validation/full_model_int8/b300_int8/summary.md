# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `reproduced-results/2026-07-13_ae-validation/performance/fixed_tiles_int8/b300_int8`
- Attention case: `b300_int8`
- System: `NVIDIA Blackwell B300 x4 single-card`
- Tensor throughput: `4485.9392` TFLOP/s
- Vector throughput: `70.0928` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

- Baseline for speedups: `Unfused Attention`

| Context | Variant | Model ms | Tok/s | Attn core ms | Attn TFLOP/s | Model speedup | Attn-core speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | Unfused Attention | 20.475 | 100022.761 | 0.129 | 534.036 | 1.00000x | 1.00000x |
| 2048 | FlashAttention | 18.179 | 112654.553 | 0.057 | 1207.013 | 1.12629x | 2.26017x |
| 2048 | FlashAttention-CustomSA | 17.714 | 115616.464 | 0.042 | 1621.528 | 1.15590x | 3.03636x |
| 4096 | Unfused Attention | 34.912 | 117323.793 | 0.353 | 779.314 | 1.00000x | 1.00000x |
| 4096 | FlashAttention | 28.896 | 141747.524 | 0.165 | 1668.617 | 1.20817x | 2.14114x |
| 4096 | FlashAttention-CustomSA | 26.691 | 153457.323 | 0.096 | 2868.448 | 1.30798x | 3.68073x |
| 8192 | Unfused Attention | 78.124 | 104859.529 | 1.249 | 880.404 | 1.00000x | 1.00000x |
| 8192 | FlashAttention | 57.230 | 143142.614 | 0.596 | 1845.016 | 1.36509x | 2.09565x |
| 8192 | FlashAttention-CustomSA | 48.410 | 169222.338 | 0.320 | 3432.622 | 1.61380x | 3.89892x |
| 16384 | Unfused Attention | 221.901 | 73834.847 | 4.833 | 909.912 | 1.00000x | 1.00000x |
| 16384 | FlashAttention | 141.493 | 115793.759 | 2.321 | 1895.102 | 1.56828x | 2.08273x |
| 16384 | FlashAttention-CustomSA | 106.213 | 154255.880 | 1.218 | 3610.135 | 2.08920x | 3.96756x |
| 32768 | Unfused Attention | 738.870 | 44348.799 | 19.172 | 917.601 | 1.00000x | 1.00000x |
| 32768 | FlashAttention | 420.407 | 77943.467 | 9.220 | 1908.051 | 1.75751x | 2.07939x |
| 32768 | FlashAttention-CustomSA | 279.288 | 117326.913 | 4.810 | 3657.420 | 2.64555x | 3.98585x |
| 65536 | Unfused Attention | 2690.470 | 24358.570 | 76.526 | 919.543 | 1.00000x | 1.00000x |
| 65536 | FlashAttention | 1419.787 | 46159.045 | 36.817 | 1911.316 | 1.89498x | 2.07855x |
| 65536 | FlashAttention-CustomSA | 849.837 | 77116.003 | 19.006 | 3702.457 | 3.16587x | 4.02641x |
| 131072 | Unfused Attention | 10264.314 | 12769.680 | 305.941 | 920.030 | 1.00000x | 1.00000x |
| 131072 | FlashAttention | 5184.749 | 25280.298 | 147.205 | 1912.134 | 1.97971x | 2.07834x |
| 131072 | FlashAttention-CustomSA | 2904.948 | 45120.251 | 75.961 | 3705.527 | 3.53339x | 4.02761x |
| 262144 | Unfused Attention | 40094.577 | 6538.141 | 1223.602 | 920.152 | 1.00000x | 1.00000x |
| 262144 | FlashAttention | 19779.487 | 13253.327 | 588.755 | 1912.339 | 2.02708x | 2.07829x |
| 262144 | FlashAttention-CustomSA | 10638.392 | 24641.319 | 303.096 | 3714.662 | 3.76886x | 4.03701x |
| 524288 | Unfused Attention | 158485.410 | 3308.115 | 4894.246 | 920.183 | 1.00000x | 1.00000x |
| 524288 | FlashAttention | 77228.216 | 6788.814 | 2354.959 | 1912.390 | 2.05217x | 2.07827x |
| 524288 | FlashAttention-CustomSA | 40663.835 | 12893.226 | 1212.322 | 3714.855 | 3.89745x | 4.03708x |
