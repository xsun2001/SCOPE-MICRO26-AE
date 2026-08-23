# Figure 20: impact of shape constraints

This directory contains only the 18 width-16 runs used by the ablation: nine shape-constrained SCNA runs and nine unconstrained neural approximators. It uses the same corrected shared trainer as Figure 17.

- `../../train/`: shared trainer and original analysis utilities, also used by Figure 17.
- `data/manifest.tsv`: exact run matrix.
- `expected-results/`: paper-aligned width-16 source values and the canonical paper-style `figure20.{png,pdf}` outcome.
- `actual-results/`: ignored runtime destination for fresh user executions; rerun histories are not committed.
- `scripts/`: worker, collector, and Figure 20 generator. The generator reuses the original paper renderer in `../../train/merge_shape_fix_results.py`, including its compact dimensions, smoothing, typography, legend, colors, annotations, and panel formatting.

Run through Slurm with `make reproduce`, or on an already allocated GPU with `make reproduce EXECUTOR=local WORKERS=1`. Use `make evidence` to audit the 18-entry reference matrix and stage the canonical `figure20.{png,pdf}` outcome under `runs/<run-id>/evidence/fig-20-shape-constraints/` without runtime histories. A fresh `make reproduce` writes its histories only under the ignored run directory and compares its summaries with the reference matrix.

A fresh run needs an NVIDIA CUDA GPU from the Ampere generation or newer with at least 16 GB of device memory. H100 80 GB (Hopper) is the validated reference, but A100 (Ampere), H100/H200 (Hopper), and B100/B200 (Blackwell) devices are also suitable with a compatible PyTorch/CUDA installation. The 18 required runs consumed 12.51 H100 GPU-hours, extracted from timestamped tasks in shared Slurm job `410238` and bounded by each array element's Slurm start and end times. They were part of a larger 72-task sweep, so standalone Figure 20 wall time was not measured. The value excludes environment setup and queue delay.

The corrected Rsqrt pair uses `1 / sqrt(-x)` on `[-256, -1]`; it replaces the earlier reciprocal curves. The fresh data match 18/18 best-MSE values and produce a 47.1×–976.3× improvement range. Validation uses either 20% relative error or a 2e-4 absolute MSE floor for near-zero values. Independently of that numerical envelope, every one of the nine constrained runs must improve over its unconstrained pair. Seeds are fixed; exact CUDA determinism across stacks is not claimed.
