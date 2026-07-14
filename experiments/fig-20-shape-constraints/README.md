# Figure 20: impact of shape constraints

This directory contains only the 18 width-16 runs used by the ablation: nine shape-constrained SCNA runs and nine unconstrained neural approximators. The copied trainer is the same paper-era snapshot used by Figure 17.

- `../../train/`: shared trainer and original analysis utilities, also used by Figure 17.
- `data/manifest.tsv`: exact run matrix.
- `expected-results/`: paper-aligned width-16 source values.
- `actual-results/2026-07-13_ae-validation/`: validated raw histories, configs, summaries, analysis, and generated figures. The paper figure plots eight functions; Exp2 is retained in the nine-function numerical validation.
- `scripts/`: worker, collector, and Figure 20 generator.

Run through Slurm with `make reproduce`, or on an already allocated GPU with `make reproduce EXECUTOR=local WORKERS=1`. Use `make evidence` to recollect all 18 best-MSE comparisons from the bundled summaries and regenerate `actual-results/2026-07-13_ae-validation/generated/figure20.{png,pdf}` from the bundled raw histories without a GPU.

A fresh run needs an NVIDIA CUDA GPU from the Ampere generation or newer with at least 16 GB of device memory. H100 80 GB (Hopper) is the validated reference, but A100 (Ampere), H100/H200 (Hopper), and B100/B200 (Blackwell) devices are also suitable with a compatible PyTorch/CUDA installation. The 18 required runs consumed 12.51 H100 GPU-hours, extracted from timestamped tasks in shared Slurm job `410238` and bounded by each array element's Slurm start and end times. They were part of a larger 72-task sweep, so standalone Figure 20 wall time was not measured. The value excludes environment setup and queue delay.

The fresh run matched 18/18 best-MSE values and reproduced the 47.1×–2264.3× improvement range.
