# OSTQuant SCNA Corrected Protocol Results

Run root: `/data/user/cxu930/projects/pinn-fullstack/experiments/ostquant_scna_investigation/results/2026-06-12_140246_corrected_protocol`

## README W4A4KV4 Check

| Model | Quant | Measured PPL | README PPL | Delta | Measured Acc % | README Acc % | Delta % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | 5.9418 | 5.9100 | 0.0318 | 62.89 | 63.18 | -0.29 |
| llama3_8b | w4a4kv4 | 7.3301 | 7.2900 | 0.0401 | 64.42 | 65.37 | -0.95 |

## Exact Eager and SCNA Resume Evaluation

| Model | Quant | Mode | PPL | Delta PPL vs Exact SDPA | Delta PPL vs Exact Eager | Acc % | Delta Acc % vs Exact SDPA | Delta Acc % vs Exact Eager | RC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | exact_eager20 | 9.5140 | 3.5722 | 0.0730 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | exact_eager | 9.4410 | 3.4993 | 0.0000 | 53.39 | -9.50 | 0.00 | 0 |
| llama2_7b | w4a4kv4 | exact_eager_eager20 | 9.5140 | 3.5722 | 0.0730 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | exact_eager_maskfix | 5.9463 | 0.0045 | -3.4947 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | exact_eager_maskfix_acc | 5.9463 | 0.0045 | -3.4947 | 62.58 | -0.31 | 9.19 | 0 |
| llama2_7b | w4a4kv4 | scna_d16 | 207.7085 | 201.7668 | 198.2675 | 42.33 | -20.56 | -11.06 | 0 |
| llama2_7b | w4a4kv4 | scna_d16_clip6 | 2526.5510 | 2520.6092 | 2517.1100 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d16_clip8 | 198.4149 | 192.4731 | 188.9739 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d16_clip8_q6in | 202.8160 | 196.8742 | 193.3750 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d16_maskfix | 5.9571 | 0.0153 | -3.4839 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d16_maskfix_acc | 5.9571 | 0.0153 | -3.4839 | 62.93 | 0.04 | 9.54 | 0 |
| llama2_7b | w4a4kv4 | scna_d16_q6in | 212.1608 | 206.2191 | 202.7198 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32 | 211.4780 | 205.5362 | 202.0369 | 42.13 | -20.76 | -11.26 | 0 |
| llama2_7b | w4a4kv4 | scna_d32_clip8 | 206.2618 | 200.3200 | 196.8208 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_fit_L16_g15 | 193.8326 | 187.8908 | 184.3916 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_fit_L16_g2 | 198.5395 | 192.5977 | 189.0984 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_fit_L24_g2 | 189.4967 | 183.5550 | 180.0557 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_fit_L24_g2_eager20 | 74.6650 | 68.7232 | 65.2240 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_fit_attn_l2w4 | 200.5030 | 194.5612 | 191.0620 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_fit_roll_l2w4 | 203.7942 | 197.8524 | 194.3531 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_fit_roll_l2w4_eager20 | 81.4209 | 75.4791 | 71.9799 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_floor12 | 209.1453 | 203.2035 | 199.7042 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_floor16 | 211.6482 | 205.7064 | 202.2072 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_maskfix | 5.9525 | 0.0107 | -3.4886 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d32_maskfix_acc | 5.9525 | 0.0107 | -3.4886 | 62.77 | -0.12 | 9.38 | 0 |
| llama2_7b | w4a4kv4 | scna_d8 | 238.3766 | 232.4348 | 228.9355 | 42.10 | -20.79 | -11.29 | 0 |
| llama2_7b | w4a4kv4 | scna_d8_maskfix | 5.9993 | 0.0575 | -3.4418 |  |  |  | 0 |
| llama2_7b | w4a4kv4 | scna_d8_maskfix_acc | 5.9993 | 0.0575 | -3.4418 | 62.44 | -0.45 | 9.05 | 0 |
| llama2_7b | w6a6kv6 | exact_eager20 | 8.3030 | 2.8023 | -0.0168 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | exact_eager | 8.3198 | 2.8191 | 0.0000 | 58.50 | -6.51 | 0.00 | 0 |
| llama2_7b | w6a6kv6 | exact_eager_eager20 | 8.3030 | 2.8023 | -0.0168 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | exact_eager_maskfix | 5.5019 | 0.0012 | -2.8179 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | exact_eager_maskfix_acc | 5.5019 | 0.0012 | -2.8179 | 65.09 | 0.08 | 6.59 | 0 |
| llama2_7b | w6a6kv6 | scna_d16 | 62.3296 | 56.8289 | 54.0098 | 46.17 | -18.84 | -12.33 | 0 |
| llama2_7b | w6a6kv6 | scna_d16_clip6 | 1853.3317 | 1847.8310 | 1845.0119 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d16_clip8 | 60.4251 | 54.9244 | 52.1054 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d16_clip8_q6in | 60.7031 | 55.2024 | 52.3833 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d16_maskfix | 5.5033 | 0.0026 | -2.8165 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d16_maskfix_acc | 5.5033 | 0.0026 | -2.8165 | 65.04 | 0.03 | 6.54 | 0 |
| llama2_7b | w6a6kv6 | scna_d16_q6in | 63.0819 | 57.5812 | 54.7622 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32 | 63.0371 | 57.5364 | 54.7173 | 45.92 | -19.09 | -12.58 | 0 |
| llama2_7b | w6a6kv6 | scna_d32_clip8 | 65.7730 | 60.2723 | 57.4532 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_fit_L16_g15 | 58.8838 | 53.3831 | 50.5640 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_fit_L16_g2 | 59.0035 | 53.5028 | 50.6837 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_fit_L24_g2 | 57.7179 | 52.2172 | 49.3981 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_fit_L24_g2_eager20 | 61.6584 | 56.1578 | 53.3387 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_fit_attn_l2w4 | 61.5211 | 56.0205 | 53.2014 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_fit_roll_l2w6 | 61.1955 | 55.6948 | 52.8757 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_fit_roll_l2w6_eager20 | 65.2464 | 59.7457 | 56.9266 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_floor12 | 62.5217 | 57.0210 | 54.2020 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_floor16 | 62.9986 | 57.4979 | 54.6788 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_maskfix | 5.5044 | 0.0037 | -2.8154 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d32_maskfix_acc | 5.5044 | 0.0037 | -2.8154 | 65.06 | 0.05 | 6.56 | 0 |
| llama2_7b | w6a6kv6 | scna_d8 | 69.1288 | 63.6281 | 60.8091 | 45.86 | -19.15 | -12.64 | 0 |
| llama2_7b | w6a6kv6 | scna_d8_maskfix | 5.5355 | 0.0348 | -2.7843 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | scna_d8_maskfix_acc | 5.5355 | 0.0348 | -2.7843 | 64.99 | -0.02 | 6.49 | 0 |
| llama2_7b | w6a6kv6 | train_scnafit5_d32_pwlL24 |  |  |  |  |  |  | 1 |
| llama2_7b | w6a6kv6 | train_scnafit5b_d32_pwlL24 | 63.6683 | 58.1676 | 55.3485 |  |  |  | 0 |
| llama2_7b | w6a6kv6 | train_scnafit_d32_pwlL24 |  |  |  |  |  |  | None |
| llama3_8b | w4a4kv4 | exact_eager20 | 10.8693 | 3.5392 | 0.5728 |  |  |  | 0 |
| llama3_8b | w4a4kv4 | exact_eager | 10.2964 | 2.9664 | 0.0000 | 55.48 | -8.94 | 0.00 | 0 |
| llama3_8b | w4a4kv4 | exact_eager_maskfix | 7.3298 | -0.0003 | -2.9667 |  |  |  | 0 |
| llama3_8b | w4a4kv4 | exact_eager_maskfix_acc | 7.3298 | -0.0003 | -2.9667 | 64.64 | 0.22 | 9.16 | 0 |
| llama3_8b | w4a4kv4 | scna_d16 | 51.7333 | 44.4033 | 41.4369 | 38.93 | -25.49 | -16.55 | 0 |
| llama3_8b | w4a4kv4 | scna_d16_maskfix | 7.3449 | 0.0148 | -2.9516 |  |  |  | 0 |
| llama3_8b | w4a4kv4 | scna_d16_maskfix_acc | 7.3449 | 0.0148 | -2.9516 | 65.11 | 0.69 | 9.63 | 0 |
| llama3_8b | w4a4kv4 | scna_d32 | 51.8153 | 44.4853 | 41.5189 | 39.12 | -25.30 | -16.36 | 0 |
| llama3_8b | w4a4kv4 | scna_d32_fit_attn_l2w4 | 52.0540 | 44.7240 | 41.7576 |  |  |  | 0 |
| llama3_8b | w4a4kv4 | scna_d32_maskfix | 7.3467 | 0.0166 | -2.9497 |  |  |  | 0 |
| llama3_8b | w4a4kv4 | scna_d32_maskfix_acc | 7.3467 | 0.0166 | -2.9497 | 64.72 | 0.30 | 9.24 | 0 |
| llama3_8b | w4a4kv4 | scna_d8 | 55.4879 | 48.1579 | 45.1915 | 39.26 | -25.16 | -16.22 | 0 |
| llama3_8b | w4a4kv4 | scna_d8_maskfix | 7.4449 | 0.1148 | -2.8515 |  |  |  | 0 |
| llama3_8b | w4a4kv4 | scna_d8_maskfix_acc | 7.4449 | 0.1148 | -2.8515 | 64.44 | 0.02 | 8.96 | 0 |
| llama3_8b | w6a6kv6 | exact_eager20 | 9.2793 | 3.0508 | 0.1696 |  |  |  | 0 |
| llama3_8b | w6a6kv6 | exact_eager | 9.1097 | 2.8812 | 0.0000 | 58.49 | -9.56 | 0.00 | 0 |
| llama3_8b | w6a6kv6 | exact_eager_maskfix | 6.2296 | 0.0011 | -2.8801 |  |  |  | 0 |
| llama3_8b | w6a6kv6 | exact_eager_maskfix_acc | 6.2296 | 0.0011 | -2.8801 | 68.03 | -0.02 | 9.54 | 0 |
| llama3_8b | w6a6kv6 | scna_d16 | 36.6882 | 30.4597 | 27.5785 | 42.85 | -25.20 | -15.64 | 0 |
| llama3_8b | w6a6kv6 | scna_d16_maskfix | 6.2380 | 0.0095 | -2.8717 |  |  |  | 0 |
| llama3_8b | w6a6kv6 | scna_d16_maskfix_acc | 6.2380 | 0.0095 | -2.8717 | 68.02 | -0.03 | 9.53 | 0 |
| llama3_8b | w6a6kv6 | scna_d32 | 36.5411 | 30.3126 | 27.4314 | 42.96 | -25.09 | -15.53 | 0 |
| llama3_8b | w6a6kv6 | scna_d32_fit_attn_l2w4 | 36.5079 | 30.2794 | 27.3982 |  |  |  | 0 |
| llama3_8b | w6a6kv6 | scna_d32_maskfix | 6.2377 | 0.0092 | -2.8720 |  |  |  | 0 |
| llama3_8b | w6a6kv6 | scna_d32_maskfix_acc | 6.2377 | 0.0092 | -2.8720 | 67.97 | -0.08 | 9.48 | 0 |
| llama3_8b | w6a6kv6 | scna_d8 | 38.5825 | 32.3540 | 29.4728 | 42.94 | -25.11 | -15.55 | 0 |
| llama3_8b | w6a6kv6 | scna_d8_maskfix | 6.3153 | 0.0868 | -2.7944 |  |  |  | 0 |
| llama3_8b | w6a6kv6 | scna_d8_maskfix_acc | 6.3153 | 0.0868 | -2.7944 | 68.01 | -0.04 | 9.52 | 0 |

## Qmodel GPTQ Verification

| Model | Quant | Suffix | Qmodel PPL | Delta PPL vs Matching Exact | RC | Qmodel File |
| --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | eager20 | 9.5140 | 0.0000 | 0 | True |
| llama2_7b | w4a4kv4 | sdpa | 5.9418 | 0.0000 | 0 | True |
| llama2_7b | w6a6kv6 | eager20 | 8.3030 | 0.0000 | 0 | True |
| llama2_7b | w6a6kv6 | sdpa | 5.5007 | 0.0000 | 0 | True |
| llama3_8b | w4a4kv4 | sdpa | 7.3301 | 0.0000 | 0 | True |
| llama3_8b | w6a6kv6 | sdpa | 6.2285 | 0.0000 | 0 | True |

## Best SCNA by PPL

| Model | Quant | Best Mode | PPL | Delta PPL vs Exact SDPA | Delta PPL vs Exact Eager | Acc % | Delta Acc % vs Exact SDPA | Delta Acc % vs Exact Eager |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | w4a4kv4 | scna_d32_maskfix | 5.9525 | 0.0107 | -3.4886 |  |  |  |
| llama2_7b | w6a6kv6 | scna_d16_maskfix | 5.5033 | 0.0026 | -2.8165 |  |  |  |
| llama3_8b | w4a4kv4 | scna_d16_maskfix | 7.3449 | 0.0148 | -2.9516 |  |  |  |
| llama3_8b | w6a6kv6 | scna_d32_maskfix | 6.2377 | 0.0092 | -2.8720 |  |  |  |

## Notes

- `exact_sdpa` uses SDPA attention and `gradient_accumulation_steps=8` to match the README script's effective training batch size; direct `torchrun` was not usable in this environment.
- `qmodel_*_sdpa` regenerates GPTQ weights under the SDPA exact path from the learned OST transform and saves `qmodel.pt`.
- `exact_eager` and all SCNA rows load both the `exact_sdpa` learned OST transform and the SDPA-generated `qmodel.pt`, then use explicit attention for nonlinear calculation.
- Additional suffixed rows such as `exact_eager20` or `*_eager20` are investigation probes and are compared against their matching qmodel suffix when present.
- Rows with blank accuracy were intentionally run as PPL-only probes.
- Deltas against `exact_sdpa` are the requested README-aligned baseline deltas; deltas against `exact_eager` isolate SCNA from the explicit-attention fallback.

## Incomplete Or Failed Rows

| Model | Quant | Mode | Return Code | Result Dir |
| --- | --- | --- | --- | --- |
| llama2_7b | w6a6kv6 | train_scnafit5_d32_pwlL24 | 1 | train_llama2_7b_w6a6kv6_scnafit5_d32_pwlL24 |
| llama2_7b | w6a6kv6 | train_scnafit_d32_pwlL24 | None | train_llama2_7b_w6a6kv6_scnafit_d32_pwlL24 |
