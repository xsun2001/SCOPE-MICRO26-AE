# Figure 17: neuron-count scalability

This directory contains only the 36 shape-constrained runs used for the 4/8/16/32-neuron comparison (nine functions × four widths). The copied trainer is the paper-era snapshot from commit `20b562040e1d07c888b1c1e3efbedc6f71048453`, SHA-256 `8aa2816d76343b3ae294fbc80da03f51e617c8e78da13c0c9b90ab8237a5010a`.

- `../../train/`: shared trainer and original analysis utilities.
- `data/manifest.tsv`: exact run matrix.
- `expected-results/`: paper-aligned best-MSE source data.
- `actual-results/2026-07-13_ae-validation/`: validated raw histories, configs, summaries, analysis, and generated figures. The paper figure plots eight functions; Exp2 is retained in the nine-function numerical validation.
- `scripts/`: worker, collector, and Figure 17 generator.

Run through Slurm with `make reproduce`, or directly on an allocated GPU with `make reproduce EXECUTOR=local WORKERS=1`. Use `make evidence` to recollect all 36 best-MSE comparisons from the bundled summaries and regenerate `actual-results/2026-07-13_ae-validation/generated/figure17.{png,pdf}` from the bundled raw histories without a GPU.

A fresh run needs an NVIDIA CUDA GPU from the Ampere generation or newer with at least 16 GB of device memory. H100 80 GB (Hopper) is the validated reference, but A100 (Ampere), H100/H200 (Hopper), and B100/B200 (Blackwell) devices are also suitable with a compatible PyTorch/CUDA installation. The 36 required runs consumed 25.74 H100 GPU-hours, extracted from timestamped tasks in shared Slurm job `410238` and bounded by each array element's Slurm start and end times. They were part of a larger 72-task sweep, so standalone Figure 17 wall time was not measured. The value excludes environment setup and queue delay.

The fresh run matched 36/36 constrained best-MSE values. Its 32-vs-4 gain range is 97.2×–2837.8×. The paper prose says 32 neurons are lowest in every case; the saved Exp sweep instead has width 16 at `5.52e-9` versus width 32 at `1.35e-8`. The bundle preserves the saved data and does not conceal that exception.
