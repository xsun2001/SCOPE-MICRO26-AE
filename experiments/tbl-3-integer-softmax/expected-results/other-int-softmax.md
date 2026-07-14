# Long-Sequence Softmax H100 Microbenchmark
These are actual CUDA/Triton timings on H100.
The benchmark uses fixed total elements per sequence length rather than a square attention matrix, so 512k sequence length is physically runnable.
All three measured kernels read the same INT8 logits and write U8 probabilities.
- Score scale: `x_real = x_i * 1 / 2^4`.
- IntAttention LUT zero threshold: `6.6` real units, `106` integer units.
- I-LLM DI clip: `15` real units, `240` integer units.
- I-LLM DI-Exp uses `m_e=185`, `k_e=7`, `exp_bits=16`.
## Latency
| Seq Len | Rows | Elements | Triton FP ms | I-LLM DI ms | IntAttention ms | I-LLM / FP | IntAttention / FP | I-LLM / IntAttention |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 32768 | 67108864 | 0.107 | 0.179 | 0.117 | 1.671x | 1.093x | 1.528x |
| 4096 | 16384 | 67108864 | 0.108 | 0.181 | 0.120 | 1.685x | 1.118x | 1.508x |
| 8192 | 8192 | 67108864 | 0.111 | 0.188 | 0.123 | 1.702x | 1.114x | 1.528x |
| 16384 | 4096 | 67108864 | 0.120 | 0.203 | 0.144 | 1.699x | 1.201x | 1.415x |
| 32768 | 2048 | 67108864 | 0.130 | 0.215 | 0.158 | 1.654x | 1.218x | 1.358x |
| 65536 | 1024 | 67108864 | 0.154 | 0.254 | 0.191 | 1.653x | 1.242x | 1.332x |
| 131072 | 512 | 67108864 | 0.223 | 0.313 | 0.283 | 1.403x | 1.268x | 1.106x |
| 262144 | 256 | 67108864 | 0.383 | 0.514 | 0.488 | 1.342x | 1.273x | 1.054x |
| 524288 | 128 | 67108864 | 0.708 | 0.938 | 0.905 | 1.325x | 1.279x | 1.036x |
## Throughput
| Seq Len | Triton FP Gelem/s | I-LLM DI Gelem/s | IntAttention Gelem/s | Memory Floor Gelem/s |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 627.89 | 375.80 | 574.33 | 1178.51 |
| 4096 | 624.25 | 370.42 | 558.57 | 1157.37 |
| 8192 | 606.20 | 356.08 | 544.01 | 1137.90 |
| 16384 | 561.04 | 330.23 | 467.12 | 1134.82 |
| 32768 | 516.79 | 312.42 | 424.31 | 1099.71 |
| 65536 | 436.04 | 263.71 | 351.16 | 976.56 |
| 131072 | 301.08 | 214.65 | 237.49 | 778.89 |
| 262144 | 175.09 | 130.50 | 137.54 | 483.33 |
| 524288 | 94.81 | 71.55 | 74.12 | 287.75 |
Notes:
- Baseline is a fused Triton online FP softmax: INT8 load, scale conversion, online max/sum, exp, normalization, U8 store.
- I-LLM is a fused Triton implementation of DI-Exp plus clipped DI-Softmax. The row-common `2^(kx + k_e)` factor is removed before normalization to keep the intended 16-bit exp scale on long rows.
- The I-LLM kernel uses FP32 for the final row sum and U8 normalization so the 16-bit DI-Exp path remains runnable at 512k without 64-bit division dominating the H100 measurement.
- IntAttention is a fused Triton IndexSoftmax implementation using a 5-bit index and U8 exponential LUT.
- The memory floor is one INT8 read plus one U8 write pass and is included only as a bandwidth diagnostic.
- Device: NVIDIA H100 80GB HBM3, torch 2.10.0+cu128, Triton 3.6.0, CUDA 12.8.
