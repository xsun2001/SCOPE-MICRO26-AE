# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `reproduced-results/2026-07-13_ae-validation/performance/fixed_tiles_fp16/awsv4`
- Attention case: `awsv4_fp16`
- System: `AWS NeuronCore v4 (official per-chip model) single-card`
- Tensor throughput: `671.0886` TFLOP/s
- Vector throughput: `9.8304` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

- Baseline for speedups: `Unfused Attention`

| Context | Variant | Model ms | Tok/s | Attn core ms | Attn TFLOP/s | Model speedup | Attn-core speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | Unfused Attention | 116.774 | 17538.180 | 0.971 | 70.772 | 1.00000x | 1.00000x |
| 2048 | FlashAttention | 102.031 | 20072.251 | 0.510 | 134.665 | 1.14449x | 1.90280x |
| 2048 | FlashAttention-CustomSA | 100.270 | 20424.892 | 0.455 | 150.949 | 1.16460x | 2.13289x |
| 4096 | Unfused Attention | 207.845 | 19706.957 | 2.297 | 119.686 | 1.00000x | 1.00000x |
| 4096 | FlashAttention | 170.467 | 24028.076 | 1.129 | 243.557 | 1.21927x | 2.03497x |
| 4096 | FlashAttention-CustomSA | 163.718 | 25018.561 | 0.918 | 299.529 | 1.26953x | 2.50263x |
| 8192 | Unfused Attention | 473.957 | 17284.286 | 7.572 | 145.208 | 1.00000x | 1.00000x |
| 8192 | FlashAttention | 347.313 | 23586.771 | 3.614 | 304.203 | 1.36464x | 2.09495x |
| 8192 | FlashAttention-CustomSA | 320.318 | 25574.575 | 2.771 | 396.821 | 1.47964x | 2.73278x |
| 16384 | Unfused Attention | 1342.051 | 12208.182 | 28.619 | 153.678 | 1.00000x | 1.00000x |
| 16384 | FlashAttention | 860.097 | 19049.021 | 13.558 | 324.397 | 1.56035x | 2.11089x |
| 16384 | FlashAttention-CustomSA | 752.116 | 21783.877 | 10.183 | 431.892 | 1.78437x | 2.81038x |
| 32768 | Unfused Attention | 4421.727 | 7410.678 | 112.696 | 156.103 | 1.00000x | 1.00000x |
| 32768 | FlashAttention | 2522.028 | 12992.717 | 53.330 | 329.872 | 1.75324x | 2.11317x |
| 32768 | FlashAttention-CustomSA | 2090.105 | 15677.681 | 39.833 | 441.651 | 2.11555x | 2.82923x |
| 65536 | Unfused Attention | 15955.033 | 4107.544 | 448.787 | 156.798 | 1.00000x | 1.00000x |
| 65536 | FlashAttention | 8391.350 | 7809.947 | 212.422 | 331.269 | 1.90137x | 2.11272x |
| 65536 | FlashAttention-CustomSA | 6663.658 | 9834.839 | 158.431 | 444.160 | 2.39434x | 2.83269x |
| 131072 | Unfused Attention | 60517.451 | 2165.855 | 1792.712 | 157.011 | 1.00000x | 1.00000x |
| 131072 | FlashAttention | 30311.832 | 4324.120 | 848.786 | 331.621 | 1.99650x | 2.11209x |
| 131072 | FlashAttention-CustomSA | 23401.061 | 5601.114 | 632.825 | 444.791 | 2.58610x | 2.83287x |
| 262144 | Unfused Attention | 235625.520 | 1112.545 | 7167.539 | 157.083 | 1.00000x | 1.00000x |
| 262144 | FlashAttention | 114880.141 | 2281.891 | 3394.246 | 331.708 | 2.05106x | 2.11167x |
| 262144 | FlashAttention-CustomSA | 87237.057 | 3004.962 | 2530.399 | 444.950 | 2.70098x | 2.83257x |
| 524288 | Unfused Attention | 929774.585 | 563.887 | 28665.096 | 157.111 | 1.00000x | 1.00000x |
| 524288 | FlashAttention | 446926.150 | 1173.098 | 13576.082 | 331.730 | 2.08038x | 2.11144x |
| 524288 | FlashAttention-CustomSA | 336353.811 | 1558.740 | 10120.697 | 444.989 | 2.76428x | 2.83232x |
