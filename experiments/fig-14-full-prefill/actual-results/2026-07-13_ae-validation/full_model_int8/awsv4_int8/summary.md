# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `reproduced-results/2026-07-13_ae-validation/performance/fixed_tiles_int8/awsv4_int8`
- Attention case: `awsv4_int8`
- System: `AWS NeuronCore v4 (official per-chip model) single-card`
- Tensor throughput: `2684.3546` TFLOP/s
- Vector throughput: `9.8304` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

- Baseline for speedups: `Unfused Attention`

| Context | Variant | Model ms | Tok/s | Attn core ms | Attn TFLOP/s | Model speedup | Attn-core speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | Unfused Attention | 75.889 | 26986.630 | 0.807 | 85.154 | 1.00000x | 1.00000x |
| 2048 | FlashAttention | 64.227 | 31886.687 | 0.443 | 155.276 | 1.18157x | 1.82347x |
| 2048 | FlashAttention-CustomSA | 61.358 | 33378.081 | 0.353 | 194.738 | 1.23684x | 2.28689x |
| 4096 | Unfused Attention | 115.582 | 35438.104 | 1.641 | 167.540 | 1.00000x | 1.00000x |
| 4096 | FlashAttention | 90.761 | 45129.381 | 0.865 | 317.769 | 1.27347x | 1.89667x |
| 4096 | FlashAttention-CustomSA | 79.396 | 51589.324 | 0.510 | 539.115 | 1.45576x | 3.21782x |
| 8192 | Unfused Attention | 247.447 | 33106.139 | 4.948 | 222.213 | 1.00000x | 1.00000x |
| 8192 | FlashAttention | 170.698 | 47991.259 | 2.550 | 431.249 | 1.44962x | 1.94070x |
| 8192 | FlashAttention-CustomSA | 155.564 | 52660.037 | 2.077 | 529.460 | 1.59064x | 2.38266x |
| 16384 | Unfused Attention | 721.096 | 22720.968 | 18.123 | 242.682 | 1.00000x | 1.00000x |
| 16384 | FlashAttention | 438.054 | 37401.789 | 9.278 | 474.050 | 1.64614x | 1.95338x |
| 16384 | FlashAttention-CustomSA | 376.435 | 43524.148 | 7.352 | 598.211 | 1.91559x | 2.46500x |
| 32768 | Unfused Attention | 2508.075 | 13065.000 | 70.712 | 248.786 | 1.00000x | 1.00000x |
| 32768 | FlashAttention | 1404.024 | 23338.635 | 36.210 | 485.832 | 1.78635x | 1.95281x |
| 32768 | FlashAttention-CustomSA | 1154.048 | 28393.957 | 28.399 | 619.472 | 2.17328x | 2.48998x |
| 65536 | Unfused Attention | 9440.753 | 6941.819 | 280.851 | 250.556 | 1.00000x | 1.00000x |
| 65536 | FlashAttention | 5059.663 | 12952.642 | 143.942 | 488.870 | 1.86589x | 1.95114x |
| 65536 | FlashAttention-CustomSA | 4052.764 | 16170.694 | 112.476 | 625.633 | 2.32946x | 2.49698x |
| 131072 | Unfused Attention | 36740.989 | 3567.460 | 1120.968 | 251.100 | 1.00000x | 1.00000x |
| 131072 | FlashAttention | 19265.738 | 6803.373 | 574.866 | 489.635 | 1.90706x | 1.94996x |
| 131072 | FlashAttention-CustomSA | 15224.146 | 8609.481 | 448.567 | 627.499 | 2.41334x | 2.49900x |
| 262144 | Unfused Attention | 145080.981 | 1806.881 | 4480.563 | 251.285 | 1.00000x | 1.00000x |
| 262144 | FlashAttention | 75257.075 | 3483.314 | 2298.566 | 489.827 | 1.92781x | 1.94929x |
| 262144 | FlashAttention-CustomSA | 59062.720 | 4438.400 | 1792.492 | 628.120 | 2.45639x | 2.49963x |
| 524288 | Unfused Attention | 576719.045 | 909.087 | 17917.192 | 251.356 | 1.00000x | 1.00000x |
| 524288 | FlashAttention | 297556.497 | 1761.978 | 9193.362 | 489.875 | 1.93818x | 1.94893x |
| 524288 | FlashAttention-CustomSA | 232723.098 | 2252.840 | 7167.319 | 628.352 | 2.47813x | 2.49985x |
