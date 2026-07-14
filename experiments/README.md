# Experiment Index

All experiment directories use `fig-X-description` or `tbl-X-description`, include a Makefile and README, store bundled measurements under `actual-results/<validated-run>/`, and store paper targets under `expected-results/`.

Fresh GPU experiments use an NVIDIA CUDA GPU from the Ampere generation or newer. H100 80 GB (Hopper) is the validated timing reference, not a device lock: A100 80 GB (Ampere), H100/H200 (Hopper), and B100/B200 (Blackwell) are compatible examples when supported by the installed PyTorch/CUDA stack. Figure 16 and Table 5 require at least 80 GB of device memory; Figures 17 and 20 require at least 16 GB.

| Directory | Paper item | Fresh-run hardware | Expected reference time | Bundled evidence |
| --- | --- | --- | ---: | --- |
| `tbl-3-integer-softmax` | Table 3 | CPU, 8 workers | less than 1 minute | Reproduced |
| `tbl-4-function-approximation-accuracy` | Table 4 | source audit only | less than 1 minute | Evidence-only; missing common raw baseline protocol |
| `tbl-5-ostquant-quality` | Table 5 | NVIDIA CUDA GPU (Ampere or newer), at least 80 GB | 14.6 H100 GPU-hours measured; about 3 hours with 15 H100 GPUs | 16/16 configurations pass |
| `fig-13-prefill-attention` | Figure 13 | CPU, 8 workers | 33 minutes measured | Reproduced |
| `fig-14-full-prefill` | Figure 14 | CPU, 8 workers | 3 minutes measured after Figure 13 | Reproduced |
| `fig-15-b300-sensitivity` | Figure 15 | CPU, 8 workers | less than 1 minute | Reproduced |
| `fig-16-end-to-end-quality` | Figure 16 | NVIDIA CUDA GPU (Ampere or newer), at least 80 GB | 18.80 H100 GPU-hours measured; 2:26:32 with 16 H100 GPUs | 80/80 comparisons pass |
| `fig-17-neuron-scalability` | Figure 17 | NVIDIA CUDA GPU (Ampere or newer), at least 16 GB | 25.74 H100 GPU-hours measured for 36 required tasks | 36/36 configurations pass |
| `fig-18-pe-area-power` | Figure 18 | CPU report analysis | less than 1 minute | 112 report sets; 18 fitted result rows pass |
| `fig-19-hardware-comparison` | Figure 19 | CPU calculation | less than 1 minute after Figure 18 | 14 plotted result rows pass |
| `fig-20-shape-constraints` | Figure 20 | NVIDIA CUDA GPU (Ampere or newer), at least 16 GB | 12.51 H100 GPU-hours measured for 18 required tasks | 18/18 configurations pass |
| `fig-21-scale-fusion` | Figure 21 | CPU, 8 workers | less than 1 minute after Figure 13 | Reproduced |

The four required fresh GPU targets total **71.66 H100 GPU-hours** when run separately: 18.80 for Figure 16, 14.6 for Table 5, 25.74 for Figure 17, and 12.51 for Figure 20. The Figure 17 and Figure 20 task sets overlap by nine width-16 constrained runs totaling 6.29 H100 GPU-hours; reusing those runs reduces the total to **65.37 H100 GPU-hours**. No complete-suite wall time is reported because the required Figure 17 and Figure 20 subsets were measured within one larger shared sweep rather than as standalone jobs.

At the repository root, use `make evidence && make validate` for a hardware-free audit, `make reproduce-cpu` for the CPU suite, and `make reproduce-gpu` for the GPU suite.

The complete `make reproduce-cpu` suite took 2,219 seconds (37 minutes) with `JOBS=8` on a dual-socket AMD EPYC 9654 host. This clean-bundle measurement had populated Python and sbt dependency caches; allow extra time for setup on a new host. GPU hours exclude setup/download and scheduler queue time, and each Slurm worker has one GPU. Figure 16 is measured from Slurm accounting for job `410219`. Figures 17 and 20 are task-level sums from timestamped task transitions bounded by each array element's Slurm start and end times for shared job `410238`; its complete 72-task sweep consumed 51.33 H100 GPU-hours and 3:51:36 wall time with 16 workers. Table 5 is measured from the included job timestamps.
