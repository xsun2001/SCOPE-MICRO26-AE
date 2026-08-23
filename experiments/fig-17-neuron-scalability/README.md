# Figure 17: neuron-count scalability

This directory contains only the 36 shape-constrained runs used for the 4/8/16/32-neuron comparison (nine functions × four widths). The shared trainer descends from paper-era commit `20b562040e1d07c888b1c1e3efbedc6f71048453` and now distinguishes reciprocal square root from reciprocal.

- `../../train/`: shared trainer and original analysis utilities.
- `data/manifest.tsv`: exact run matrix.
- `expected-results/`: paper-aligned best-MSE source data and the canonical paper-style `figure17.{png,pdf}` outcome.
- `actual-results/`: ignored runtime destination for fresh user executions; rerun histories are not committed.
- `scripts/`: worker, collector, and Figure 17 generator. The generator reuses the original paper renderer in `../../train/merge_shape_fix_results.py`, including its compact dimensions, smoothing, typography, legend, colors, and panel formatting.

Run through Slurm with `make reproduce`, or directly on an allocated GPU with `make reproduce EXECUTOR=local WORKERS=1`. Use `make evidence` to audit the 36-entry reference matrix and stage the canonical `figure17.{png,pdf}` outcome under `runs/<run-id>/evidence/fig-17-neuron-scalability/` without runtime histories. A fresh `make reproduce` writes its histories only under the ignored run directory and compares its summaries with the reference matrix.

A fresh run needs an NVIDIA CUDA GPU from the Ampere generation or newer with at least 16 GB of device memory. H100 80 GB (Hopper) is the validated reference, but A100 (Ampere), H100/H200 (Hopper), and B100/B200 (Blackwell) devices are also suitable with a compatible PyTorch/CUDA installation. The 36 required runs consumed 25.74 H100 GPU-hours, extracted from timestamped tasks in shared Slurm job `410238` and bounded by each array element's Slurm start and end times. They were part of a larger 72-task sweep, so standalone Figure 17 wall time was not measured. The value excludes environment setup and queue delay.

The corrected Rsqrt rows use `1 / sqrt(-x)` on `[-256, -1]`; they replace the earlier reciprocal curves. The fresh data match 36/36 configurations and retain the reported 32-vs-4 gain range of 97.2×–2837.8×. Validation uses a fixed-seed 20% relative best-MSE portability envelope; exact CUDA determinism across stacks is not claimed, and the width-scaling gains are reported separately.
