# SCOPE Artifact Evaluation Submission

## Key Results to be Reproduced

This unified artifact contains the paper's CPU performance and hardware experiments together with its GPU accuracy and numerical-precision experiments. The following results are claimed for the **Results Reproduced** badge:

1. **Table 3 — Integer-softmax throughput.** Reproduces modeled H100 INT8 attention throughput for SCOPE, I-LLM, and IntAttention at 2K, 4K, 8K, and 16K. SCOPE obtains 1130.86, 1526.51, 1672.82, and 1713.89 TFLOP/s.
2. **Figure 13 — Prefill-attention performance.** Reproduces FP16 and INT8 SCOPE-versus-FlashAttention results on modeled B200, AWSv4, and TPUv6e devices for 2K--32K context. At 32K, the FP16 speedups are 1.34x, 1.34x, and 1.70x; the INT8 speedups are 3.05x, 2.51x, and 2.81x.
3. **Figure 14 — Llama 3 8B full-prefill performance.** Reproduces modeled full-prefill results through 512K context. At 512K, the FP16 speedups on AWSv4, B200, and TPUv6e are 1.329x, 1.341x, and 1.681x; the INT8 speedups are 1.28x, 2.69x, and 1.91x.
4. **Figure 15 — B300 sensitivity.** Reproduces the doubled-SFU-throughput study. At 512K, the attention/full-prefill speedups are 1.09x/1.08x for FP16 and 1.94x/1.90x for INT8.
5. **Figure 16 — End-to-end model quality.** Reproduces WikiText-2 perplexity and four-task zero-shot mean accuracy for OPT-6.7B, Llama-2-7B, Llama-3-8B, Qwen2.5-7B, and Qwen3-8B. It covers exact and SCNA-8/16/32 variants under full-precision and INT8 conditions. The bundled validation contains 80/80 matching comparisons.
6. **Figure 17 — Neuron-count scalability.** Reproduces MSE for 4, 8, 16, and 32 neurons on nine nonlinear functions. The bundled validation contains 36/36 matching configurations and a 97.2x--2837.8x gain from 4 to 32 neurons.
7. **Figure 18 — Per-PE area and power.** Extracts 112 filtered native synthesis report sets, computes `whole_array / N^2`, fits per-PE values across completed array sizes, checks the paper-rounded CSV, and redraws the figure. SCOPE requires 1.09--1.44x baseline area and 1.18--1.34x baseline power per PE.
8. **Figure 19 — 32x32 hardware comparison.** Reproduces throughput-normalized incremental overhead over a 32x32 baseline array from the verified Figure 18 fit and bundled literature values. SCNA-8 provides geometric-mean reductions of 12.8x in area and 9.5x in power relative to the plotted competitors. Large-array values are explicitly calculated from fitted per-PE results rather than presented as completed full 32x32 synthesis runs.
9. **Figure 20 — Effect of shape constraints.** Reproduces the width-16 constrained-versus-unconstrained ablation on nine functions. The bundled validation contains 18/18 matching configurations and a 47.1x--2264.3x MSE improvement.
10. **Figure 21 — Scale-conversion fusion.** Reproduces the INT8 scale-fusion ablation. The speedup reaches 1.11x on B200, 1.97x on AWSv4, and 1.46x on TPUv6e at the longest reported context for each device.
11. **Table 4 — Nonlinear approximation accuracy.** Reproduces our SCNA accuracy for 11 nonlinear functions. The other method columns are literature reference values; please refer to their papers for baseline reproduction.
12. **Table 5 — OSTQuant low-bit model quality.** Reproduces perplexity and four-task mean accuracy for Llama-2-7B and Llama-3-8B under W6A6 and W4A4, comparing OSTQuant, SCNA-8/16/32, and the FP16 baseline. The bundled validation contains 20/20 matching table entries.
13. **RTL generation and synthesis-report audit.** Regenerates four current N=8 SCOPE/Pinnacle SystemVerilog configurations from Chisel and audits the archived Design Compiler area, power, and timing reports without requiring a Synopsys license.

Review the bundled evidence without a GPU or proprietary tools with:

```bash
make setup
make evidence
make validate
```

Fresh CPU and GPU execution is separated into `make reproduce-cpu` and `make reproduce-gpu`. See each experiment's README for its exact command and expected output.

## Hardware Dependencies

- **Bundled evidence and validation:** no accelerator, 2 CPU cores, 8 GB host memory, and 5 GB free storage plus dependency caches.
- **Fresh CPU suite:** no accelerator, 8 CPU cores, 16 GB host memory, and 20 GB free storage. The modeled B200/B300/H100/AWSv4/TPUv6e results do not require those physical devices.
- **Fresh Figure 16:** NVIDIA CUDA GPU (Ampere or newer) with at least 80 GB device memory, 8 CPU cores, and 96 GB host memory.
- **Fresh Table 5:** NVIDIA CUDA GPU (Ampere or newer) with at least 80 GB device memory, 8 CPU cores, 192 GB host memory, and about 200 GB free storage including generated intermediates.
- **Fresh Figures 17 and 20:** NVIDIA CUDA GPU (Ampere or newer) with at least 16 GB device memory, 8 CPU cores, 16 GB host memory, and less than 10 GB additional storage.

Expected execution time on our hardware is:

- About 15 seconds for evidence validation.
- 37 minutes for the CPU suite on a dual-socket AMD EPYC 9654 host with `JOBS=8`.
- 2 hours 27 minutes for Figure 16 with 16 H100 GPUs.
- About 2 hours for Figure 17 and 1 hour for Figure 20 with 16 H100 GPUs.
- About 3 hours for Table 5 with 15 H100 GPUs.
- About 5 hours for the complete GPU suite with up to 16 H100 GPUs shared across concurrent experiments.

Setup, downloads, and scheduler queue time are additional.

Other setups can use `JOBS=<n>` and `WORKERS=<n>` to match available CPU/GPU resources. Detailed local and concurrent execution instructions are in `README.md`. Slurm is optional, and no FPGA board, microcontroller, or custom accelerator is required.

## Software Dependencies

The artifact targets 64-bit GNU/Linux and uses:

- GNU Make 4.2 or later and Bash.
- Python 3.12; Python 3.10 or newer is expected to work for the CPU workflows.
- `uv` for convenient environment creation, or standard `venv`/`pip`.
- CPU simulation and plotting packages from the bundled SCALE-Sim and LLMCompass requirements, including NumPy, SciPy, pandas, Numba, Matplotlib, Seaborn, Cython, PyTorch, and SCALE-Sim. Source snapshots are included.
- GPU packages pinned in `requirements/accuracy.txt`, including PyTorch 2.10.0, Transformers 4.51.0, Datasets 2.17.1, LM Evaluation Harness 0.4.4, Accelerate 0.33.0, NumPy 1.26.4, and Matplotlib 3.10.8.
- NVIDIA CUDA 12.8-compatible drivers/runtime for fresh GPU reproduction. The validated PyTorch build is `torch 2.10.0+cu128`.
- Optional Slurm with `sbatch`; local execution is supported.
- OpenJDK 21 and an sbt launcher for RTL regeneration. The RTL project pins sbt 1.12.5, Scala 2.13.18, and Chisel 7.9.0.
- Standard command-line utilities such as `tar` and `find`.

Network access is normally needed during `make setup`, the first sbt invocation, and model/dataset acquisition. An existing environment can be selected with `PYTHON=/path/to/python`.

No proprietary software is required for the submitted validation workflow. The archived synthesis was originally performed with **Synopsys Design Compiler V-2023.12** and TSMC 28 nm libraries at a 1 GHz target. Synopsys and those technology files are required only to repeat synthesis from RTL; the native area, power, and timing reports and all extraction/fitting/plotting scripts are included.

## Data Dependencies

All inputs for the claimed CPU experiments are included: LLMCompass and SCALE-Sim sources, device and workload configurations, expected CSVs, filtered native synthesis reports, synthesis-time RTL, current Chisel generators, generated RTL examples, and the literature values used by Figure 19.

Fresh GPU reproduction requires Hugging Face-format checkpoint directories under `MODEL_ROOT` for:

- `facebook-opt-6.7b`
- `Llama-2-7b-hf`
- `Llama-3-8b`
- `Qwen2.5-7B`
- `Qwen3-8B`

The weights are not redistributed. Reviewers must obtain them and accept applicable licenses or gated-access terms. Dataset libraries download these datasets unless they are already cached:

- WikiText-2 raw (`wikitext-2-raw-v1`) for perplexity, OSTQuant training, calibration, and evaluation.
- `mit-han-lab/pile-val-backup` for Figure 16 activation calibration.
- ARC-Easy, HellaSwag, PIQA, and WinoGrande for Figures 16 and Table 5.

Figures 17 and 20 generate samples synthetically and require no external dataset. Table 5's large `model.bin` and `qmodel.pt` intermediates are omitted; they can be regenerated or supplied through `TABLE5_CHECKPOINT_SOURCE`. Compact actual results, expected paper values, synthesis evidence, and `paper/SCOPE-revision.pdf` are included.
