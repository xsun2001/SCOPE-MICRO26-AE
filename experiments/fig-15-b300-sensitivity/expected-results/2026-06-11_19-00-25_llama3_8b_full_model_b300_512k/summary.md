# Llama 3 8B Derived Full-Model Attention Summary

- Attention source: `experiments/llmcompass_llama3_8b_full_model_attention/results/2026-06-11_18-59-58_attention_fixed_tiles_b300_512k/b300`
- Attention case: `b300_fp16`
- System: `NVIDIA Blackwell B300 x4 single-card`
- Tensor throughput: `2242.9696` TFLOP/s
- Vector throughput: `70.0928` TFLOP/s
- Scope: `32 * one_decoder_layer + final_norm + lm_head_for_last_token`
- Limitation: GQA is not modeled; attention uses the active 32-head MHA simulator interface.

| Context | Flash model ms | CustomSA model ms | Model speedup | Attention-core speedup | Attention-total speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 25.291 | 25.364 | 0.99709x | 0.96111x | 0.99132x |
| 4096 | 43.124 | 42.806 | 1.00743x | 1.06403x | 1.02052x |
| 8192 | 85.708 | 84.436 | 1.01506x | 1.07127x | 1.03511x |
| 16384 | 198.541 | 193.454 | 1.02629x | 1.07334x | 1.04919x |
| 32768 | 534.870 | 514.523 | 1.03955x | 1.07388x | 1.05969x |
| 65536 | 1650.183 | 1558.981 | 1.05850x | 1.08368x | 1.07487x |
| 131072 | 5651.423 | 5286.616 | 1.06901x | 1.08372x | 1.07911x |
| 262144 | 20736.365 | 19237.885 | 1.07789x | 1.08618x | 1.08374x |
| 524288 | 79236.090 | 73242.172 | 1.08184x | 1.08618x | 1.08495x |
