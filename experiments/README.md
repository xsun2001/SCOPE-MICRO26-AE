# Experiment Index

All experiment directories use `fig-X-description` or `tbl-X-description`, include a Makefile and README, store bundled measurements under `actual-results/<validated-run>/`, and store paper targets under `expected-results/`.

## Hardware groups

- **CPU performance/modeling:** Table 3 and Figures 13, 14, 15, and 21. The complete CPU suite took 37 minutes on a dual-socket AMD EPYC 9654 host with `JOBS=8`.
- **CPU report analysis:** Figures 18 and 19. Each takes less than one minute from the bundled reports.
- **80 GB CUDA GPU:** Figure 16 and Table 5. The validated reference is H100 80 GB; A100 80 GB, H100/H200, and B100/B200 are suitable with a compatible PyTorch/CUDA stack.
- **16 GB CUDA GPU:** Figures 17 and 20. Ampere or newer is supported.
- **Table 4:** reproduces SCNA accuracy on CPU for 11 functions. Other method columns are literature reference values; refer to their papers for baseline reproduction.

## GPU timing reference

- Figure 16: 18.80 H100 GPU-hours, or 2:26:32 with 16 GPUs.
- Figure 17: 25.74 H100 GPU-hours, or about 2 hours with 16 GPUs.
- Figure 20: 12.51 H100 GPU-hours, or about 1 hour with 16 GPUs.
- Table 5: 14.6 H100 GPU-hours, or about 3 hours with 15 GPUs.
- Complete suite: allow about 5 hours when up to 16 H100 GPUs are shared across concurrent experiments.

At the repository root, use `make evidence && make validate` for a hardware-free audit, `make reproduce-cpu` for the CPU suite, and `make reproduce-gpu` for the GPU suite. Setup, downloads, and scheduler queue time are additional. See the root `README.md` for detailed CPU/GPU concurrency commands.
