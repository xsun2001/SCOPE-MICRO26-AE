# Experiment Index

All experiment directories use `fig-X-description` or `tbl-X-description`, include a Makefile and README, store bundled measurements under `actual-results/<validated-run>/`, and store paper targets under `expected-results/`.

Fresh GPU experiments use an NVIDIA CUDA GPU from the Ampere generation or newer. H100 80 GB (Hopper) is the validated timing reference, not a device lock: A100 80 GB (Ampere), H100/H200 (Hopper), and B100/B200 (Blackwell) are compatible examples when supported by the installed PyTorch/CUDA stack. Figure 16 and Table 5 require at least 80 GB of device memory; Figures 17 and 20 require at least 16 GB.

| Directory | Paper item | Fresh-run hardware | Expected reference time | Bundled evidence |
| --- | --- | --- | ---: | --- |
| `tbl-3-integer-softmax` | Table 3 | CPU, 8 workers | less than 1 minute | Reproduced |
| `tbl-4-function-approximation-accuracy` | Table 4 | source audit only | less than 1 minute | Evidence-only; missing common raw baseline protocol |
| `tbl-5-ostquant-quality` | Table 5 | NVIDIA CUDA GPU (Ampere or newer), at least 80 GB | 14.6 H100 GPU-hours; about 3 hours with 15 H100 GPUs | 16/16 configurations pass |
| `fig-13-prefill-attention` | Figure 13 | CPU, 8 workers | 33 minutes measured | Reproduced |
| `fig-14-full-prefill` | Figure 14 | CPU, 8 workers | 3 minutes measured after Figure 13 | Reproduced |
| `fig-15-b300-sensitivity` | Figure 15 | CPU, 8 workers | less than 1 minute | Reproduced |
| `fig-16-end-to-end-quality` | Figure 16 | NVIDIA CUDA GPU (Ampere or newer), at least 80 GB | about 16--24 H100 GPU-hours; 2--4 hours with 15 H100 GPUs | 80/80 comparisons pass |
| `fig-17-neuron-scalability` | Figure 17 | NVIDIA CUDA GPU (Ampere or newer), at least 16 GB | about 12--18 H100 GPU-hours; 1--2 hours with 15 H100 GPUs | 36/36 configurations pass |
| `fig-18-pe-area-power` | Figure 18 | CPU report analysis | less than 1 minute | 112 report sets; 18 fitted result rows pass |
| `fig-19-hardware-comparison` | Figure 19 | CPU calculation | less than 1 minute after Figure 18 | 14 plotted result rows pass |
| `fig-20-shape-constraints` | Figure 20 | NVIDIA CUDA GPU (Ampere or newer), at least 16 GB | about 6--10 H100 GPU-hours; 0.5--1 hour with 15 H100 GPUs | 18/18 configurations pass |
| `fig-21-scale-fusion` | Figure 21 | CPU, 8 workers | less than 1 minute after Figure 13 | Reproduced |

The four required fresh GPU experiments total **48.6--66.6 H100 GPU-hours**: 16--24 for Figure 16, 14.6 for Table 5, 12--18 for Figure 17, and 6--10 for Figure 20. For allocation requests, round this to **49--67 H100 GPU-hours**. The documented 15-GPU execution is expected to take about **7--10 hours of elapsed time** because Table 5 has dependent checkpoint-generation and evaluation stages; GPU-hours should not be divided by 15 as if every job were fully parallel.

At the repository root, use `make evidence && make validate` for a hardware-free audit, `make reproduce-cpu` for the CPU suite, and `make reproduce-gpu` for the GPU suite.

The complete `make reproduce-cpu` suite took 2,219 seconds (37 minutes) with `JOBS=8` on a dual-socket AMD EPYC 9654 host. This clean-bundle measurement had populated Python and sbt dependency caches; allow extra time for setup on a new host. The GPU ranges exclude setup/download and scheduler queue time. Each Slurm worker has one GPU. Table 5 is measured from the included job timestamps; the other GPU values are planning estimates.
