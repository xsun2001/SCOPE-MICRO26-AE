# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `reproduced-results/2026-07-13_ae-validation/performance/fixed_tiles_fp16/b300`
- Attention case: `b300_fp16`
- System: `NVIDIA Blackwell B300 x4 single-card`
- Tensor throughput: `2242.9696` TFLOP/s
- Vector throughput: `70.0928` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

- Baseline for speedups: `Unfused Attention`

| Context | Variant | Model ms | Tok/s | Attn core ms | Attn TFLOP/s | Model speedup | Attn-core speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | Unfused Attention | 28.074 | 72950.766 | 0.144 | 477.224 | 1.00000x | 1.00000x |
| 2048 | FlashAttention | 25.291 | 80978.937 | 0.057 | 1205.113 | 1.11005x | 2.52526x |
| 2048 | FlashAttention-CustomSA | 25.364 | 80743.188 | 0.059 | 1158.243 | 1.10682x | 2.42704x |
| 4096 | Unfused Attention | 51.089 | 80174.002 | 0.414 | 663.967 | 1.00000x | 1.00000x |
| 4096 | FlashAttention | 43.124 | 94981.749 | 0.165 | 1664.988 | 1.18470x | 2.50764x |
| 4096 | FlashAttention-CustomSA | 42.806 | 95687.189 | 0.155 | 1771.602 | 1.19349x | 2.66821x |
| 8192 | Unfused Attention | 114.399 | 71609.106 | 1.494 | 735.965 | 1.00000x | 1.00000x |
| 8192 | FlashAttention | 85.708 | 95580.751 | 0.597 | 1840.581 | 1.33476x | 2.50091x |
| 8192 | FlashAttention-CustomSA | 84.436 | 97020.310 | 0.558 | 1971.754 | 1.35486x | 2.67914x |
| 16384 | Unfused Attention | 310.138 | 52828.179 | 5.814 | 756.472 | 1.00000x | 1.00000x |
| 16384 | FlashAttention | 198.541 | 82522.158 | 2.326 | 1890.423 | 1.56209x | 2.49900x |
| 16384 | FlashAttention-CustomSA | 193.454 | 84692.057 | 2.168 | 2029.063 | 1.60316x | 2.68227x |
| 32768 | Unfused Attention | 978.090 | 33502.043 | 23.094 | 761.779 | 1.00000x | 1.00000x |
| 32768 | FlashAttention | 534.870 | 61263.481 | 9.243 | 1903.308 | 1.82865x | 2.49850x |
| 32768 | FlashAttention-CustomSA | 514.523 | 63686.202 | 8.607 | 2043.915 | 1.90096x | 2.68308x |
| 65536 | Unfused Attention | 3419.893 | 19163.173 | 92.212 | 763.117 | 1.00000x | 1.00000x |
| 65536 | FlashAttention | 1650.183 | 39714.392 | 36.909 | 1906.557 | 2.07243x | 2.49838x |
| 65536 | FlashAttention-CustomSA | 1558.981 | 42037.720 | 34.059 | 2066.098 | 2.19367x | 2.70745x |
| 131072 | Unfused Attention | 12727.096 | 10298.657 | 368.687 | 763.452 | 1.00000x | 1.00000x |
| 131072 | FlashAttention | 5651.423 | 23192.743 | 147.572 | 1907.370 | 2.25202x | 2.49835x |
| 131072 | FlashAttention-CustomSA | 5286.616 | 24793.176 | 136.172 | 2067.054 | 2.40742x | 2.70751x |
| 262144 | Unfused Attention | 49035.890 | 5345.962 | 1474.586 | 763.536 | 1.00000x | 1.00000x |
| 262144 | FlashAttention | 20736.365 | 12641.753 | 590.226 | 1907.574 | 2.36473x | 2.49834x |
| 262144 | FlashAttention-CustomSA | 19237.885 | 13626.446 | 543.399 | 2071.960 | 2.54892x | 2.71364x |
| 524288 | Unfused Attention | 192431.025 | 2724.550 | 5898.183 | 763.557 | 1.00000x | 1.00000x |
| 524288 | FlashAttention | 79236.090 | 6616.783 | 2360.841 | 1907.625 | 2.42858x | 2.49834x |
| 524288 | FlashAttention-CustomSA | 73242.172 | 7158.280 | 2173.531 | 2072.020 | 2.62733x | 2.71364x |
