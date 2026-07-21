# Table 5 — OSTQuant and SCNA model quality

Each result is `WikiText-2 PPL / four-task accuracy (%)`, where accuracy is the unweighted mean of ARC-Easy, HellaSwag, PIQA, and WinoGrande.

| Method | W6A6 Llama-2-7B | W6A6 Llama-3-8B | W4A4 Llama-2-7B | W4A4 Llama-3-8B |
| --- | ---: | ---: | ---: | ---: |
| OSTQuant | 5.50 / 74.84 | 6.23 / 77.66 | 5.95 / 71.81 | 7.33 / 74.48 |
| w/ SCNA-8 | 5.54 / 74.80 | 6.32 / 77.62 | 6.00 / 72.20 | 7.44 / 74.08 |
| w/ SCNA-16 | 5.50 / 74.99 | 6.24 / 77.61 | 5.96 / 72.57 | 7.34 / 74.60 |
| w/ SCNA-32 | 5.50 / 74.75 | 6.24 / 77.62 | 5.95 / 72.39 | 7.35 / 74.16 |
| FP16 Baseline | 5.47 / 74.63 | 6.14 / 77.61 | 5.47 / 74.63 | 6.14 / 77.61 |
