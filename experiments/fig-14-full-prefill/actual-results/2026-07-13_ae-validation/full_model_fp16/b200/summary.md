# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `reproduced-results/2026-07-13_ae-validation/performance/fixed_tiles_fp16/b200`
- Attention case: `b200_fp16`
- System: `NVIDIA Blackwell B200 x4 single-card`
- Tensor throughput: `2242.9696` TFLOP/s
- Vector throughput: `70.0928` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

- Baseline for speedups: `Unfused Attention`

| Context | Variant | Model ms | Tok/s | Attn core ms | Attn TFLOP/s | Model speedup | Attn-core speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | Unfused Attention | 29.660 | 69049.594 | 0.194 | 355.021 | 1.00000x | 1.00000x |
| 2048 | FlashAttention | 25.580 | 80063.580 | 0.066 | 1040.274 | 1.15951x | 2.93017x |
| 2048 | FlashAttention-CustomSA | 25.364 | 80743.188 | 0.059 | 1158.243 | 1.16935x | 3.26246x |
| 4096 | Unfused Attention | 57.475 | 71266.304 | 0.614 | 448.015 | 1.00000x | 1.00000x |
| 4096 | FlashAttention | 44.281 | 92500.904 | 0.201 | 1365.948 | 1.29796x | 3.04889x |
| 4096 | FlashAttention-CustomSA | 42.806 | 95687.189 | 0.155 | 1771.602 | 1.34267x | 3.95433x |
| 8192 | Unfused Attention | 140.885 | 58146.572 | 2.322 | 473.585 | 1.00000x | 1.00000x |
| 8192 | FlashAttention | 90.334 | 90685.748 | 0.742 | 1481.934 | 1.55961x | 3.12918x |
| 8192 | FlashAttention-CustomSA | 84.436 | 97020.310 | 0.558 | 1971.754 | 1.66855x | 4.16346x |
| 16384 | Unfused Attention | 424.034 | 38638.449 | 9.373 | 469.218 | 1.00000x | 1.00000x |
| 16384 | FlashAttention | 217.046 | 75486.372 | 2.905 | 1514.075 | 1.95366x | 3.22680x |
| 16384 | FlashAttention-CustomSA | 193.454 | 84692.057 | 2.168 | 2029.063 | 2.19191x | 4.32435x |
| 32768 | Unfused Attention | 1465.263 | 22363.214 | 38.318 | 459.113 | 1.00000x | 1.00000x |
| 32768 | FlashAttention | 608.891 | 53815.893 | 11.556 | 1522.329 | 2.40645x | 3.31580x |
| 32768 | FlashAttention-CustomSA | 514.523 | 63686.202 | 8.607 | 2043.915 | 2.84781x | 4.45188x |
| 65536 | Unfused Attention | 4925.798 | 13304.646 | 139.272 | 505.262 | 1.00000x | 1.00000x |
| 65536 | FlashAttention | 1946.266 | 33672.691 | 46.161 | 1524.406 | 2.53090x | 3.01706x |
| 65536 | FlashAttention-CustomSA | 1558.981 | 42037.720 | 34.059 | 2066.098 | 3.15963x | 4.08916x |
| 131072 | Unfused Attention | 18750.718 | 6990.239 | 556.925 | 505.409 | 1.00000x | 1.00000x |
| 131072 | FlashAttention | 6835.755 | 19174.474 | 184.583 | 1524.927 | 2.74304x | 3.01721x |
| 131072 | FlashAttention-CustomSA | 5286.616 | 24793.176 | 136.172 | 2067.054 | 3.54683x | 4.08986x |
| 262144 | Unfused Attention | 73130.374 | 3584.612 | 2227.539 | 505.446 | 1.00000x | 1.00000x |
| 262144 | FlashAttention | 25473.691 | 10290.774 | 738.268 | 1525.057 | 2.87082x | 3.01725x |
| 262144 | FlashAttention-CustomSA | 19237.885 | 13626.446 | 543.399 | 2071.960 | 3.80137x | 4.09927x |
| 524288 | Unfused Attention | 288808.962 | 1815.345 | 8909.993 | 505.455 | 1.00000x | 1.00000x |
| 524288 | FlashAttention | 98185.398 | 5339.776 | 2953.007 | 1525.089 | 2.94147x | 3.01726x |
| 524288 | FlashAttention-CustomSA | 73242.172 | 7158.280 | 2173.531 | 2072.020 | 3.94321x | 4.09932x |
