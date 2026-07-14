# Figure 21: Scale-Conversion Fusion

This experiment reports the INT8 softmax scale-conversion fusion ablation. The paper gives B200 gains from 1.06x at 2K to 1.11x at 32K, TPUv6e gains from 1.10x to 1.46x, and AWSv4 gains of 1.12x/1.73x/1.91x/1.97x at 4K/8K/16K/32K.

From the bundle root, `make fig-21` first produces the matching conversion/no-conversion Figure 13 conditions. Direct invocation can reuse the included run:

```bash
make run FIG13_RUN_DIR=../fig-13-prefill-attention/actual-results/2026-07-13_ae-validation
```

The Makefile copies the paired inputs into its timestamped `actual-results/` directory and invokes `generate_paper_figures.py`. The ablation PNG/PDF and exact result CSV inputs are present under both actual and expected result trees.
