# Source provenance

This directory contains the shared trainer used by Figures 17 and 20 and the SCNA side of Table 4. It began from parent commit `20b562040e1d07c888b1c1e3efbedc6f71048453` and now includes the corrected split between reciprocal square root (`rsqrt`, evaluated as `1 / sqrt(-x)` on `[-256, -1]`) and reciprocal (`recip`, evaluated as `-(1 / x)` on `[-16, -1]`). The `baselines/` directory and Taylor diagnostic preserve the partial Table 4 baseline implementations found under the same workspace source directory.
