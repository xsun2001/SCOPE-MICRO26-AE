# Artifact Evaluation Submission Information

## Key Results to be Reproduced

The artifact targets the paper's accuracy and numerical-precision results that were executed on the GPU host. The following results are claimed for the **Results Reproduced** badge:

1. **Figure 16: end-to-end model quality.** Reproduce WikiText-2 perplexity and zero-shot mean accuracy for OPT-6.7B, Llama-2-7B, Llama-3-8B, Qwen2.5-7B, and Qwen3-8B. The experiment covers the exact and SCNA-8/16/32 variants under the paper's full-precision and INT8 conditions; the bundled runtime implements the paper's FP16-labeled full-precision condition with BF16. The zero-shot mean uses ARC-Easy, HellaSwag, PIQA, and WinoGrande. The bundled validation contains 80/80 matching comparisons and regenerates `experiments/fig-16-end-to-end-quality/generated/figure16.pdf` and `.png`.

2. **Table 5: OSTQuant low-bit model quality.** Reproduce perplexity and zero-shot accuracy for Llama-2-7B and Llama-3-8B under W6A6 and W4A4, comparing OSTQuant with SCNA-8/16/32. The accuracy is averaged over the nine configured LM Evaluation Harness tasks. The bundled validation contains 16/16 matching configurations and regenerates `experiments/tbl-5-ostquant-quality/generated/table5.md`.

3. **Figure 17: neuron-count scalability.** Reproduce the best MSE for 4, 8, 16, and 32 neurons on nine nonlinear functions: Exp, Exp2, Sigmoid, Erf, Rsqrt, Sin, Tanh, Softsign, and Arctan. The bundled validation contains 36/36 matching configurations and reproduces a 97.2×–2837.8× gain from 4 to 32 neurons. The saved Exp sweep has one disclosed exception to the paper's general trend: its 16-neuron MSE is lower than its 32-neuron MSE. The artifact preserves the measured data rather than changing that run.

4. **Figure 20: effect of shape constraints.** Reproduce the width-16 constrained-versus-unconstrained ablation for the same nine functions. The bundled validation contains 18/18 matching configurations and reproduces a 47.1×–2264.3× MSE improvement from applying the shape constraints.

Table 4 is included only as an evidence audit and is **not** claimed as freshly reproducible. The original workspace lacks the shared evaluation grid, raw baseline outputs, and unified harness needed to rerun all Table 4 methods under one protocol.

Run `make reproduce` for fresh GPU execution. Run `make evidence && make validate` to check the bundled actual results and regenerate the reviewer-facing figures and tables without rerunning the GPU workloads.

## Hardware Dependencies

The complete reproduction can run sequentially on one machine. Slurm improves throughput but is not required.

| Scope | GPU | CPU | Host memory | Storage |
|---|---:|---:|---:|---:|
| Bundled evidence validation only | None | 2 cores | 8 GB | Less than 1 GB beyond the extracted artifact |
| Figure 16 | 1 NVIDIA H100 80 GB | 8 cores | 96 GB | Model and dataset caches |
| Table 5 | 1 NVIDIA H100 80 GB | 8 cores | 192 GB | Approximately 60 GB for generated OSTQuant checkpoints, in addition to models and datasets |
| Figures 17 and 20 | 1 CUDA-capable NVIDIA GPU | 8 cores | 16 GB | Less than 10 GB beyond the environment |

The validated full-reproduction configuration is therefore one NVIDIA H100 80 GB GPU, 8 CPU cores, 192 GB host memory, and approximately 200 GB of free storage for the artifact, five model checkpoints, Hugging Face caches, and Table 5 intermediates. An equivalent NVIDIA GPU with at least 80 GB VRAM may work but has not been validated. Up to 15 independent GPU workers may be used to shorten the run, but this is optional; `EXECUTOR=local WORKERS=1` runs sequentially without Slurm.

No FPGA board, custom accelerator, or other hardware device is needed. Hardware performance, synthesis, area, power, and end-to-end latency experiments are outside this artifact's scope.

## Software Dependencies

The validated environment uses:

- A GNU/Linux x86-64 system, Bash, and GNU Make 4.2 or later.
- Python 3.12. The bundle can create its environment with `make setup`; `uv` is supported but optional.
- NVIDIA CUDA 12.8-compatible drivers/runtime for fresh GPU reproduction. The validated PyTorch build is `torch 2.10.0+cu128`.
- Slurm with `sbatch` for the default executor. Slurm is optional; set `EXECUTOR=local` when running directly on an allocated GPU.
- The packages pinned in `requirements/accuracy.txt`: `torch==2.10.0`, `transformers==4.51.0`, `datasets==2.17.1`, `lm-eval==0.4.4`, `accelerate==0.33.0`, `matplotlib==3.10.8`, `numpy==1.26.4`, `scipy==1.17.1`, `geoopt==0.5.1`, `loguru==0.7.3`, `einops==0.8.2`, and `tqdm==4.67.3`.

No proprietary software is required. In particular, the claimed accuracy/precision experiments do not require Cadence, Synopsys, Xilinx/Vivado, or other commercial EDA tools.

## Data Dependencies

The pretrained model weights are not included in the artifact. Set `MODEL_ROOT` to a directory containing these Hugging Face-format checkpoint directories:

- `facebook-opt-6.7b`
- `Llama-2-7b-hf`
- `Llama-3-8b`
- `Qwen2.5-7B`
- `Qwen3-8B`

The reviewer is responsible for obtaining the weights and accepting any applicable model licenses or gated-access terms, particularly for the Llama checkpoints.

Fresh reproduction also requires the following datasets, which the Hugging Face `datasets` and LM Evaluation Harness libraries download automatically unless already cached:

- WikiText-2 raw (`wikitext-2-raw-v1`) for perplexity, OSTQuant training, calibration, and evaluation.
- `mit-han-lab/pile-val-backup` for the static activation-calibration path in Figure 16.
- ARC-Easy, HellaSwag, PIQA, and WinoGrande for Figure 16 zero-shot accuracy.
- ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, LAMBADA OpenAI, OpenBookQA, PIQA, Social IQA, and WinoGrande for Table 5 zero-shot accuracy.

Figures 17 and 20 generate their function-approximation samples synthetically and require no external dataset.

The large Table 5 `model.bin` and `qmodel.pt` intermediates are not included in the portable archive. They can be regenerated, or the reviewer can set `TABLE5_CHECKPOINT_SOURCE` to a directory containing the four `exact_*_sdpa` and four `qmodel_*_sdpa` checkpoint directories. The compact actual results, expected paper values, experiment manifests, and data needed by `make evidence` are included in the artifact.
