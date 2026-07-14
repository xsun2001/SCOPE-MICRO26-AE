# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `reproduced-results/2026-07-13_ae-validation/performance/fixed_tiles_int8/b200_int8`
- Attention case: `b200_int8`
- System: `NVIDIA Blackwell B200 x4 single-card`
- Tensor throughput: `4485.9392` TFLOP/s
- Vector throughput: `70.0928` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

- Baseline for speedups: `Unfused Attention`

| Context | Variant | Model ms | Tok/s | Attn core ms | Attn TFLOP/s | Model speedup | Attn-core speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | Unfused Attention | 21.999 | 93093.866 | 0.176 | 389.780 | 1.00000x | 1.00000x |
| 2048 | FlashAttention | 18.670 | 109696.619 | 0.072 | 951.104 | 1.17834x | 2.44010x |
| 2048 | FlashAttention-CustomSA | 17.714 | 115616.464 | 0.042 | 1621.528 | 1.24193x | 4.16011x |
| 4096 | Unfused Attention | 41.047 | 99787.529 | 0.544 | 504.877 | 1.00000x | 1.00000x |
| 4096 | FlashAttention | 30.857 | 132740.226 | 0.226 | 1216.223 | 1.33023x | 2.40895x |
| 4096 | FlashAttention-CustomSA | 26.691 | 153457.323 | 0.096 | 2868.448 | 1.53784x | 5.68148x |
| 8192 | Unfused Attention | 103.617 | 79060.160 | 2.046 | 537.514 | 1.00000x | 1.00000x |
| 8192 | FlashAttention | 65.073 | 125889.589 | 0.841 | 1307.327 | 1.59233x | 2.43217x |
| 8192 | FlashAttention-CustomSA | 48.410 | 169222.338 | 0.320 | 3432.622 | 2.14042x | 6.38611x |
| 16384 | Unfused Attention | 331.826 | 49375.321 | 8.269 | 531.895 | 1.00000x | 1.00000x |
| 16384 | FlashAttention | 172.866 | 94778.630 | 3.301 | 1332.277 | 1.91955x | 2.50478x |
| 16384 | FlashAttention-CustomSA | 106.213 | 154255.880 | 1.218 | 3610.135 | 3.12415x | 6.78731x |
| 32768 | Unfused Attention | 1210.160 | 27077.415 | 33.900 | 518.947 | 1.00000x | 1.00000x |
| 32768 | FlashAttention | 545.899 | 60025.715 | 13.142 | 1338.664 | 2.21682x | 2.57957x |
| 32768 | FlashAttention-CustomSA | 279.288 | 117326.913 | 4.810 | 3657.420 | 4.33302x | 7.04777x |
| 65536 | Unfused Attention | 4196.375 | 15617.288 | 123.585 | 569.394 | 1.00000x | 1.00000x |
| 65536 | FlashAttention | 1921.755 | 34102.159 | 52.503 | 1340.270 | 2.18362x | 2.35385x |
| 65536 | FlashAttention-CustomSA | 849.837 | 77116.003 | 19.006 | 3702.457 | 4.93786x | 6.50245x |
| 131072 | Unfused Attention | 16287.935 | 8047.184 | 494.179 | 569.581 | 1.00000x | 1.00000x |
| 131072 | FlashAttention | 7192.623 | 18223.116 | 209.951 | 1340.672 | 2.26453x | 2.35379x |
| 131072 | FlashAttention-CustomSA | 2904.948 | 45120.251 | 75.961 | 3705.527 | 5.60696x | 6.50571x |
| 262144 | Unfused Attention | 64189.062 | 4083.936 | 1976.555 | 569.628 | 1.00000x | 1.00000x |
| 262144 | FlashAttention | 27810.982 | 9425.917 | 839.740 | 1340.773 | 2.30805x | 2.35377x |
| 262144 | FlashAttention-CustomSA | 10638.392 | 24641.319 | 303.096 | 3714.662 | 6.03372x | 6.52121x |
| 524288 | Unfused Attention | 254863.347 | 2057.134 | 7906.057 | 569.639 | 1.00000x | 1.00000x |
| 524288 | FlashAttention | 109354.195 | 4794.402 | 3358.896 | 1340.798 | 2.33062x | 2.35377x |
| 524288 | FlashAttention-CustomSA | 40663.835 | 12893.226 | 1212.322 | 3714.855 | 6.26757x | 6.52142x |
