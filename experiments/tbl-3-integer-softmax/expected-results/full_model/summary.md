# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `experiments/llmcompass_llama3_8b_full_model_attention/results/2026-06-18_05-01-54_attention_int_softmax_h100_int8_512k/h100_int8`
- Attention case: `h100_int8`
- System: `NVIDIA H100 SXM x4 single-card`
- Tensor throughput: `1897.7587` TFLOP/s
- Vector throughput: `59.3050` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

- Baseline for speedups: `FlashAttention`

| Context | Variant | Model ms | Tok/s | Attn core ms | Attn TFLOP/s | Model speedup | Attn-core speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | FlashAttention | 28.402 | 72108.795 | 0.073 | 947.558 | 1.00000x | 1.00000x |
| 2048 | FlashAttention-CustomSA | 28.025 | 73076.662 | 0.061 | 1130.859 | 1.01342x | 1.19344x |
| 2048 | FlashAttention + I-LLM | 29.508 | 69405.048 | 0.107 | 641.650 | 0.96250x | 0.67716x |
| 2048 | FlashAttention + IntAttention | 28.555 | 71720.884 | 0.077 | 888.732 | 0.99462x | 0.93792x |
| 4096 | FlashAttention | 50.338 | 81370.012 | 0.227 | 1210.432 | 1.00000x | 1.00000x |
| 4096 | FlashAttention-CustomSA | 48.833 | 83877.200 | 0.180 | 1526.505 | 1.03081x | 1.26112x |
| 4096 | FlashAttention + I-LLM | 54.856 | 74667.690 | 0.368 | 746.359 | 0.91763x | 0.61661x |
| 4096 | FlashAttention + IntAttention | 51.116 | 80130.700 | 0.251 | 1093.302 | 0.98477x | 0.90323x |
| 8192 | FlashAttention | 104.103 | 78691.174 | 0.845 | 1300.639 | 1.00000x | 1.00000x |
| 8192 | FlashAttention-CustomSA | 98.085 | 83519.805 | 0.657 | 1672.820 | 1.06136x | 1.28615x |
| 8192 | FlashAttention + I-LLM | 122.625 | 66805.291 | 1.424 | 772.036 | 0.84896x | 0.59358x |
| 8192 | FlashAttention + IntAttention | 107.113 | 76479.844 | 0.939 | 1170.408 | 0.97190x | 0.89987x |
| 16384 | FlashAttention | 251.203 | 65222.153 | 3.318 | 1325.331 | 1.00000x | 1.00000x |
| 16384 | FlashAttention-CustomSA | 227.128 | 72135.399 | 2.566 | 1713.889 | 1.10600x | 1.29318x |
| 16384 | FlashAttention + I-LLM | 324.965 | 50417.812 | 5.623 | 782.084 | 0.77302x | 0.59010x |
| 16384 | FlashAttention + IntAttention | 272.423 | 60141.862 | 3.982 | 1104.603 | 0.92211x | 0.83345x |
| 32768 | FlashAttention | 703.680 | 46566.599 | 13.211 | 1331.651 | 1.00000x | 1.00000x |
| 32768 | FlashAttention-CustomSA | 607.382 | 53949.567 | 10.201 | 1724.473 | 1.15855x | 1.29499x |
| 32768 | FlashAttention + I-LLM | 979.752 | 33445.201 | 21.838 | 805.575 | 0.71822x | 0.60494x |
| 32768 | FlashAttention + IntAttention | 795.728 | 41179.883 | 16.087 | 1093.545 | 0.88432x | 0.82119x |
| 65536 | FlashAttention | 2241.746 | 29234.360 | 52.780 | 1333.241 | 1.00000x | 1.00000x |
| 65536 | FlashAttention-CustomSA | 1856.553 | 35299.834 | 40.743 | 1727.139 | 1.20748x | 1.29544x |
| 65536 | FlashAttention + I-LLM | 3344.294 | 19596.363 | 87.235 | 806.659 | 0.67032x | 0.60504x |
| 65536 | FlashAttention + IntAttention | 2650.494 | 24725.958 | 65.554 | 1073.453 | 0.84578x | 0.80515x |
| 131072 | FlashAttention | 7850.319 | 16696.391 | 211.058 | 1333.639 | 1.00000x | 1.00000x |
| 131072 | FlashAttention-CustomSA | 6228.103 | 21045.252 | 160.364 | 1755.229 | 1.26047x | 1.31612x |
| 131072 | FlashAttention + I-LLM | 10572.217 | 12397.778 | 296.117 | 950.553 | 0.74254x | 0.71275x |
| 131072 | FlashAttention + IntAttention | 9660.283 | 13568.133 | 267.619 | 1051.774 | 0.81264x | 0.78865x |
| 262144 | FlashAttention | 29197.237 | 8978.384 | 844.169 | 1333.738 | 1.00000x | 1.00000x |
| 262144 | FlashAttention-CustomSA | 22545.485 | 11627.339 | 636.301 | 1769.444 | 1.29504x | 1.32668x |
| 262144 | FlashAttention + I-LLM | 38439.408 | 6819.668 | 1132.986 | 993.745 | 0.75957x | 0.74508x |
| 262144 | FlashAttention + IntAttention | 36576.142 | 7167.076 | 1074.759 | 1047.583 | 0.79826x | 0.78545x |
| 524288 | FlashAttention | 112410.157 | 4664.063 | 3376.611 | 1333.763 | 1.00000x | 1.00000x |
| 524288 | FlashAttention-CustomSA | 85477.375 | 6133.647 | 2534.962 | 1776.595 | 1.31509x | 1.33202x |
| 524288 | FlashAttention + I-LLM | 147543.383 | 3553.450 | 4474.525 | 1006.498 | 0.76188x | 0.75463x |
| 524288 | FlashAttention + IntAttention | 142574.673 | 3677.287 | 4319.252 | 1042.680 | 0.78843x | 0.78176x |
