# Long-Sequence Softmax H100 Microbenchmark

These are actual CUDA/Triton timings on H100.
The benchmark uses fixed total elements per sequence length rather than a square attention matrix, so 512k sequence length is physically runnable.
All three measured kernels read the same INT8 logits and write U8 probabilities.
`raw_samples.csv` contains every timed repetition; the latency table reports medians and the CSV also records means and sample standard deviations.

- Score scale: `x_real = x_i * 1 / 2^4`.
- IntAttention LUT zero threshold: `6.6` real units, `106` integer units.
- I-LLM DI clip: `15` real units, `240` integer units.
- I-LLM DI-Exp uses `m_e=185`, `k_e=7`, `exp_bits=16`.

## Latency

| Seq Len | Rows | Elements | Triton FP ms | I-LLM DI ms | IntAttention ms | I-LLM / FP | IntAttention / FP | I-LLM / IntAttention |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 32768 | 67108864 | 0.109 | 0.180 | 0.119 | 1.650x | 1.091x | 1.512x |
| 4096 | 16384 | 67108864 | 0.110 | 0.184 | 0.122 | 1.675x | 1.115x | 1.503x |
| 8192 | 8192 | 67108864 | 0.113 | 0.191 | 0.126 | 1.692x | 1.117x | 1.514x |
| 16384 | 4096 | 67108864 | 0.122 | 0.205 | 0.146 | 1.677x | 1.196x | 1.402x |
| 32768 | 2048 | 67108864 | 0.131 | 0.217 | 0.161 | 1.658x | 1.230x | 1.348x |
| 65536 | 1024 | 67108864 | 0.155 | 0.256 | 0.192 | 1.650x | 1.240x | 1.330x |
| 131072 | 512 | 67108864 | 0.224 | 0.314 | 0.286 | 1.401x | 1.274x | 1.100x |
| 262144 | 256 | 67108864 | 0.385 | 0.514 | 0.488 | 1.337x | 1.269x | 1.053x |
| 524288 | 128 | 67108864 | 0.708 | 0.938 | 0.904 | 1.324x | 1.276x | 1.038x |

## Throughput

| Seq Len | Triton FP Gelem/s | I-LLM DI Gelem/s | IntAttention Gelem/s | Memory Floor Gelem/s |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 613.92 | 372.10 | 562.69 | 1142.24 |
| 4096 | 612.31 | 365.52 | 549.28 | 1131.76 |
| 8192 | 595.36 | 351.87 | 532.88 | 1112.84 |
| 16384 | 547.85 | 326.76 | 458.19 | 1095.98 |
| 32768 | 512.56 | 309.09 | 416.72 | 1074.09 |
| 65536 | 432.63 | 262.28 | 348.80 | 956.51 |
| 131072 | 299.25 | 213.56 | 234.87 | 764.27 |
| 262144 | 174.46 | 130.52 | 137.47 | 482.05 |
| 524288 | 94.78 | 71.57 | 74.27 | 286.50 |

Notes:
- Baseline is a fused Triton online FP softmax: INT8 load, scale conversion, online max/sum, exp, normalization, U8 store.
- I-LLM is a fused Triton implementation of DI-Exp plus clipped DI-Softmax. The row-common `2^(kx + k_e)` factor is removed before normalization to keep the intended 16-bit exp scale on long rows.
- The I-LLM kernel uses FP32 for the final row sum and U8 normalization so the 16-bit DI-Exp path remains runnable at 512k without 64-bit division dominating the H100 measurement.
- IntAttention is a fused Triton IndexSoftmax implementation using a 5-bit index and U8 exponential LUT.
- The memory floor is one INT8 read plus one U8 write pass and is included only as a bandwidth diagnostic.
- Device: NVIDIA H100 80GB HBM3, torch 2.10.0+cu128, Triton 3.6.0, CUDA 12.8.
