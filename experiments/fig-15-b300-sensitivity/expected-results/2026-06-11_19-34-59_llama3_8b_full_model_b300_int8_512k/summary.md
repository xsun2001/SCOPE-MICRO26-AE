# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `experiments/llmcompass_llama3_8b_full_model_attention/results/2026-06-11_19-33-10_attention_fixed_tiles_int8_512k/b300_int8`
- Attention case: `b300_int8`
- System: `NVIDIA Blackwell B300 x4 single-card`
- Tensor throughput: `4485.9392` TFLOP/s
- Vector throughput: `70.0928` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

| Context | Flash model ms | CustomSA model ms | Model speedup | Attention-core speedup | Attention-total speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 18.179 | 17.714 | 1.02629x | 1.34342x | 1.07756x |
| 4096 | 28.896 | 26.691 | 1.08261x | 1.71906x | 1.22788x |
| 8192 | 57.230 | 48.410 | 1.18219x | 1.86048x | 1.42442x |
| 16384 | 141.493 | 106.213 | 1.33216x | 1.90498x | 1.61508x |
| 32768 | 420.407 | 279.288 | 1.50528x | 1.91684x | 1.75071x |
| 65536 | 1419.787 | 849.837 | 1.67066x | 1.93712x | 1.84609x |
| 131072 | 5184.749 | 2904.948 | 1.78480x | 1.93790x | 1.89092x |
| 262144 | 19779.487 | 10638.392 | 1.85926x | 1.94247x | 1.91845x |
| 524288 | 77228.216 | 40663.835 | 1.89919x | 1.94252x | 1.93042x |
