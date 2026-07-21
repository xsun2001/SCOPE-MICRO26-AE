# OSTQuant SCNA Mask-Fix Final Report

Run root: `2026-07-13_ae-validation` (selected with `--run-root`)

Accuracy protocol: Unweighted arithmetic mean over ARC-Easy, HellaSwag, PIQA, and WinoGrande.

## Root Cause

- The large eager-attention PPL regression was caused by `QuantSoftmax._align_attn_mask` trimming oversized mask dimensions from the end.
- For the LLaMA eager no-user-mask path, the key dimension can be one column longer than the attention scores. LLaMA attention keeps the first `key_len` columns; the old helper dropped column 0 and kept the extra masked column, shifting the causal mask.
- SCNA was evaluated through the explicit attention path, so the same mask bug made SCNA look much worse than it was.

## README Check

| Model | Quant | SDPA PPL | README PPL | Delta | SDPA Acc % | README Acc % | Delta % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | 5.9418 | 5.9100 | +0.0318 | 72.40 | 72.40 | +0.00 |
| llama3_8b | w4a4kv4 | 7.3301 | 7.2900 | +0.0401 | 74.20 | 74.20 | +0.00 |

## Exact Attention Sanity

| Model | Quant | SDPA PPL | SDPA Acc % | Fixed Eager PPL | Delta PPL | Fixed Eager Acc % | Delta Acc % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | 5.9418 | 72.40 | 5.9463 | +0.0045 | 71.81 | -0.59 |
| llama2_7b | w6a6kv6 | 5.5007 | 74.52 | 5.5019 | +0.0012 | 74.84 | +0.32 |
| llama3_8b | w4a4kv4 | 7.3301 | 74.20 | 7.3298 | -0.0003 | 74.48 | +0.28 |
| llama3_8b | w6a6kv6 | 6.2285 | 77.62 | 6.2296 | +0.0011 | 77.66 | +0.04 |

## Final SCNA Results

SCNA is full precision in these rows: `scna_input_quant_bits=0`, `scna_input_scale=1.0`. It is not forced to 4-bit or 6-bit.

| Model | Quant | Mode | PPL | Delta PPL vs Fixed Eager | Delta PPL vs SDPA | Acc % | Delta Acc % vs Fixed Eager | Delta Acc % vs SDPA | SCNA Input Quant Bits | RC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | SCNA-8 | 5.9993 | +0.0530 | +0.0575 | 72.20 | +0.39 | -0.20 | 0 | 0 |
| llama2_7b | w4a4kv4 | SCNA-16 | 5.9571 | +0.0108 | +0.0153 | 72.57 | +0.76 | +0.17 | 0 | 0 |
| llama2_7b | w4a4kv4 | SCNA-32 | 5.9525 | +0.0062 | +0.0107 | 72.39 | +0.58 | -0.01 | 0 | 0 |
| llama2_7b | w6a6kv6 | SCNA-8 | 5.5355 | +0.0336 | +0.0348 | 74.80 | -0.04 | +0.28 | 0 | 0 |
| llama2_7b | w6a6kv6 | SCNA-16 | 5.5033 | +0.0014 | +0.0026 | 74.99 | +0.15 | +0.47 | 0 | 0 |
| llama2_7b | w6a6kv6 | SCNA-32 | 5.5044 | +0.0025 | +0.0037 | 74.75 | -0.09 | +0.23 | 0 | 0 |
| llama3_8b | w4a4kv4 | SCNA-8 | 7.4449 | +0.1151 | +0.1148 | 74.08 | -0.40 | -0.12 | 0 | 0 |
| llama3_8b | w4a4kv4 | SCNA-16 | 7.3449 | +0.0151 | +0.0148 | 74.60 | +0.12 | +0.40 | 0 | 0 |
| llama3_8b | w4a4kv4 | SCNA-32 | 7.3467 | +0.0169 | +0.0166 | 74.16 | -0.32 | -0.04 | 0 | 0 |
| llama3_8b | w6a6kv6 | SCNA-8 | 6.3153 | +0.0857 | +0.0868 | 77.62 | -0.04 | +0.00 | 0 | 0 |
| llama3_8b | w6a6kv6 | SCNA-16 | 6.2380 | +0.0084 | +0.0095 | 77.61 | -0.05 | -0.01 | 0 | 0 |
| llama3_8b | w6a6kv6 | SCNA-32 | 6.2377 | +0.0081 | +0.0092 | 77.62 | -0.04 | +0.00 | 0 | 0 |

## Best SCNA By PPL

| Model | Quant | Best Mode | PPL | Delta PPL vs Fixed Eager | Acc % | Delta Acc % vs Fixed Eager |
| --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | SCNA-32 | 5.9525 | +0.0062 | 72.39 | +0.58 |
| llama2_7b | w6a6kv6 | SCNA-16 | 5.5033 | +0.0014 | 74.99 | +0.15 |
| llama3_8b | w4a4kv4 | SCNA-16 | 7.3449 | +0.0151 | 74.60 | +0.12 |
| llama3_8b | w6a6kv6 | SCNA-32 | 6.2377 | +0.0081 | 77.62 | -0.04 |

## Diagnostics

- Prefix LLaMA2 W4 exact eager was 9.4410 PPL; fixed eager is 5.9463 PPL.
- Prefix LLaMA3 W4 exact eager was 10.2964 PPL; fixed eager is 7.3298 PPL.
- The metric path is now the decisive signal: fixed eager matches SDPA PPL closely, and SCNA-16/32 stay close to fixed eager across both models and bit-widths.
