# Figure 20: impact of shape constraints

This directory contains only the 18 width-16 runs used by the ablation: nine shape-constrained SCNA runs and nine unconstrained neural approximators. The copied trainer is the same paper-era snapshot used by Figure 17.

- `../../train/`: shared trainer and original analysis utilities, also used by Figure 17.
- `data/manifest.tsv`: exact run matrix.
- `expected/`: paper-aligned width-16 source values.
- `actual/`: fresh validated histories and summaries from 2026-07-13/14.
- `scripts/`: worker, collector, and Figure 20 generator.

Run through Slurm with `make reproduce`, or on an already allocated GPU with `make reproduce EXECUTOR=local WORKERS=1`. Use `make evidence` to validate the bundled actual results and regenerate `generated/figure20.{png,pdf}` without a GPU.

The fresh run matched 18/18 best-MSE values and reproduced the 47.1×–2264.3× improvement range.
