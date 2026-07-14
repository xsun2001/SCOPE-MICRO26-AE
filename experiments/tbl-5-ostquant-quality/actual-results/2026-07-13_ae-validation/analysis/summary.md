# OSTQuant SCNA Corrected Protocol Results

Run root: `/data/user/cxu930/projects/pinn-fullstack/ae-exp/experiments/tbl-5-ostquant-quality/actual`

## README W4A4KV4 Check

| Model | Quant | Measured PPL | README PPL | Delta | Measured Acc % | README Acc % | Delta % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | 5.9418 | 5.9100 | 0.0318 | 62.89 | 63.18 | -0.29 |
| llama3_8b | w4a4kv4 | 7.3301 | 7.2900 | 0.0401 | 64.42 | 65.37 | -0.95 |

## Exact Eager and SCNA Resume Evaluation

| Model | Quant | Mode | PPL | Delta PPL vs Exact SDPA | Delta PPL vs Exact Eager | Acc % | Delta Acc % vs Exact SDPA | Delta Acc % vs Exact Eager | RC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | exact_eager_maskfix_acc | 5.9463 | 0.0045 |  | 62.58 | -0.31 |  | 0 |
| llama2_7b | w4a4kv4 | scna_d16_maskfix_acc | 5.9571 | 0.0153 |  | 62.93 | 0.04 |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_maskfix_acc | 5.9525 | 0.0107 |  | 62.77 | -0.12 |  | 0 |
| llama2_7b | w4a4kv4 | scna_d8_maskfix_acc | 5.9993 | 0.0575 |  | 62.44 | -0.45 |  | 0 |
| llama2_7b | w6a6kv6 | exact_eager_maskfix_acc | 5.5019 | 0.0012 |  | 65.09 | 0.08 |  | 0 |
| llama2_7b | w6a6kv6 | scna_d16_maskfix_acc | 5.5033 | 0.0026 |  | 65.04 | 0.03 |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_maskfix_acc | 5.5044 | 0.0037 |  | 65.06 | 0.05 |  | 0 |
| llama2_7b | w6a6kv6 | scna_d8_maskfix_acc | 5.5355 | 0.0348 |  | 64.99 | -0.02 |  | 0 |
| llama3_8b | w4a4kv4 | exact_eager_maskfix_acc | 7.3298 | -0.0003 |  | 64.64 | 0.22 |  | 0 |
| llama3_8b | w4a4kv4 | scna_d16_maskfix_acc | 7.3449 | 0.0148 |  | 65.11 | 0.69 |  | 0 |
| llama3_8b | w4a4kv4 | scna_d32_maskfix_acc | 7.3467 | 0.0166 |  | 64.72 | 0.30 |  | 0 |
| llama3_8b | w4a4kv4 | scna_d8_maskfix_acc | 7.4449 | 0.1148 |  | 64.44 | 0.02 |  | 0 |
| llama3_8b | w6a6kv6 | exact_eager_maskfix_acc | 6.2296 | 0.0011 |  | 68.03 | -0.02 |  | 0 |
| llama3_8b | w6a6kv6 | scna_d16_maskfix_acc | 6.2380 | 0.0095 |  | 68.02 | -0.03 |  | 0 |
| llama3_8b | w6a6kv6 | scna_d32_maskfix_acc | 6.2377 | 0.0092 |  | 67.97 | -0.08 |  | 0 |
| llama3_8b | w6a6kv6 | scna_d8_maskfix_acc | 6.3153 | 0.0868 |  | 68.01 | -0.04 |  | 0 |

## Qmodel GPTQ Verification

| Model | Quant | Suffix | Qmodel PPL | Delta PPL vs Matching Exact | RC | Qmodel File |
| --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | sdpa | 5.9418 | 0.0000 | 0 | False |
| llama2_7b | w6a6kv6 | sdpa | 5.5007 | 0.0000 | 0 | False |
| llama3_8b | w4a4kv4 | sdpa | 7.3301 | 0.0000 | 0 | False |
| llama3_8b | w6a6kv6 | sdpa | 6.2285 | 0.0000 | 0 | False |

## Best SCNA by PPL

| Model | Quant | Best Mode | PPL | Delta PPL vs Exact SDPA | Delta PPL vs Exact Eager | Acc % | Delta Acc % vs Exact SDPA | Delta Acc % vs Exact Eager |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | scna_d32_maskfix_acc | 5.9525 | 0.0107 |  | 62.77 | -0.12 |  |
| llama2_7b | w6a6kv6 | scna_d16_maskfix_acc | 5.5033 | 0.0026 |  | 65.04 | 0.03 |  |
| llama3_8b | w4a4kv4 | scna_d16_maskfix_acc | 7.3449 | 0.0148 |  | 65.11 | 0.69 |  |
| llama3_8b | w6a6kv6 | scna_d32_maskfix_acc | 6.2377 | 0.0092 |  | 67.97 | -0.08 |  |

## Notes

- `exact_sdpa` uses SDPA attention and `gradient_accumulation_steps=8` to match the README script's effective training batch size; direct `torchrun` was not usable in this environment.
- `qmodel_*_sdpa` regenerates GPTQ weights under the SDPA exact path from the learned OST transform and saves `qmodel.pt`.
- `exact_eager` and all SCNA rows load both the `exact_sdpa` learned OST transform and the SDPA-generated `qmodel.pt`, then use explicit attention for nonlinear calculation.
- Additional suffixed rows such as `exact_eager20` or `*_eager20` are investigation probes and are compared against their matching qmodel suffix when present.
- Rows with blank accuracy were intentionally run as PPL-only probes.
- Deltas against `exact_sdpa` are the requested README-aligned baseline deltas; deltas against `exact_eager` isolate SCNA from the explicit-attention fallback.
