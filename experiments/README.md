# Experiment Index

All experiment directories use `fig-X-description` or `tbl-X-description`, include a Makefile and README, store bundled measurements under `actual-results/<validated-run>/`, and store paper targets under `expected-results/`.

| Directory | Paper item | Host for fresh run | Bundled evidence |
| --- | --- | --- | --- |
| `tbl-3-integer-softmax` | Table 3 | CPU | Reproduced |
| `tbl-4-function-approximation-accuracy` | Table 4 | GPU-side source audit | Evidence-only; missing common raw baseline protocol |
| `tbl-5-ostquant-quality` | Table 5 | NVIDIA H100 80 GB | 16/16 configurations pass |
| `fig-13-prefill-attention` | Figure 13 | CPU | Reproduced |
| `fig-14-full-prefill` | Figure 14 | CPU | Reproduced |
| `fig-15-b300-sensitivity` | Figure 15 | CPU | Reproduced |
| `fig-16-end-to-end-quality` | Figure 16 | NVIDIA H100 80 GB | 80/80 comparisons pass |
| `fig-17-neuron-scalability` | Figure 17 | CUDA GPU | 36/36 configurations pass |
| `fig-18-pe-area-power` | Figure 18 | CPU report analysis | 112 report sets; 18 fitted result rows pass |
| `fig-19-hardware-comparison` | Figure 19 | CPU calculation | 14 plotted result rows pass |
| `fig-20-shape-constraints` | Figure 20 | CUDA GPU | 18/18 configurations pass |
| `fig-21-scale-fusion` | Figure 21 | CPU | Reproduced |

At the repository root, use `make evidence && make validate` for a hardware-free audit, `make reproduce-cpu` for the CPU suite, and `make reproduce-gpu` for the GPU suite.
