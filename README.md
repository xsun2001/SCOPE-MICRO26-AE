# SCOPE Artifact Evaluation Bundle

This repository is the unified reviewer bundle for `paper/SCOPE-revision.pdf`. It combines the CPU-host performance, RTL, and synthesis evidence with the GPU-host accuracy and numerical-precision experiments. All paper experiment directories use the `fig-X-description` or `tbl-X-description` naming convention and contain a Makefile, README, scripts, bundled actual results, and expected paper results.

Submission-facing result and dependency information is in `AE_SUBMISSION.md`.

## Quick start: no GPU or proprietary tools

```bash
make setup
make evidence
make validate
```

`make evidence` regenerates compact GPU-side figures/tables under ignored staging directories and audits the packaged CPU evidence without modifying tracked files. `make validate` checks both suites: the CPU validator currently reports 68/68 checks passing, Table 3 additionally recomputes 36 statistics from 720 raw H100 samples, and the GPU validator recomputes Figure 16 (80 comparisons), Table 5 (20), Figure 17 (36), Figure 20 (18), and all 11 Table 4 SCNA rows.

`make all` is the safe alias for evidence plus validation; it does not launch long CPU simulations or GPU jobs.

## Fresh reproduction

CPU-only performance, hardware plots, and RTL:

```bash
make reproduce-cpu
```

GPU accuracy and precision:

```bash
cp config/local.env.example config/local.env
# Set MODEL_ROOT and execution settings.
make reproduce-gpu                 # Slurm by default
make reproduce-gpu EXECUTOR=local WORKERS=1  # already allocated GPU
```

Fresh GPU runs require an NVIDIA CUDA GPU from the Ampere generation or newer. The validated reference is an 80 GB H100 (Hopper), but H100 is not an architectural requirement: A100 80 GB (Ampere), H100/H200 (Hopper), and B100/B200 (Blackwell) are suitable when the installed PyTorch/CUDA stack supports the device. Figure 16 and Table 5 should be given at least 80 GB of device memory; Figures 17 and 20 need at least 16 GB. The memory thresholds, rather than the H100 model name, are the relevant limits.

Run both suites with `make reproduce`.

## Experiment index

| Paper item | Directory | Fresh execution |
| --- | --- | --- |
| Table 3 | `experiments/tbl-3-integer-softmax` | CPU |
| Figures 13, 14, 15, 21 | matching `experiments/fig-*` directories | CPU |
| Figures 18 and 19 | matching `experiments/fig-*` directories | report extraction/calculation on CPU |
| Figure 16 | `experiments/fig-16-end-to-end-quality` | NVIDIA CUDA GPU (Ampere or newer), 80 GB or more |
| Figure 17 | `experiments/fig-17-neuron-scalability` | NVIDIA CUDA GPU (Ampere or newer), 16 GB or more |
| Figure 20 | `experiments/fig-20-shape-constraints` | NVIDIA CUDA GPU (Ampere or newer), 16 GB or more |
| Table 5 | `experiments/tbl-5-ostquant-quality` | NVIDIA CUDA GPU (Ampere or newer), 80 GB or more |
| Table 4 | `experiments/tbl-4-function-approximation-accuracy` | SCNA accuracy reproduction on CPU; other methods are literature references |

Every experiment stores bundled measurements under `actual-results/<validated-run>/` and paper targets under `expected-results/`. Fresh CPU runs use the same timestamped experiment directories. Fresh GPU runs are written under the ignored `runs/<RUN_ID>/` tree because Table 5 intermediates can exceed 60 GB.

## Expected runtimes

Expected execution time on our hardware is:

- About 15 seconds for `make evidence && make validate` on the reference CPU.
- 37 minutes for `make reproduce-cpu` on a dual-socket AMD EPYC 9654 host with `JOBS=8`.
- 2 hours 27 minutes for Figure 16 with 16 H100 GPUs.
- About 2 hours for Figure 17 with 16 H100 GPUs and about 1 hour for Figure 20 with 16 H100 GPUs, estimated from measured task GPU-hours.
- About 3 hours for Table 5 with 15 H100 GPUs.
- About 5 hours for the complete GPU suite when up to 16 H100 GPUs are shared across concurrent experiments.

These times exclude initial setup, downloads, and scheduler queue delay.

## Concurrent CPU/GPU execution

The Make dependency graph supports concurrent independent experiments. On a CPU host, `-j` controls concurrent experiment targets and `JOBS` controls workers inside each target:

```bash
make -j2 reproduce-cpu JOBS=4
```

Choose `-j × JOBS` conservatively for the available CPU and memory. The command above uses at most roughly eight CPU workers while preserving dependencies such as Figure 13 before Figures 14 and 21.

With Slurm and up to 16 GPUs, run four independent GPU targets concurrently with four one-GPU array workers each:

```bash
make -j4 reproduce-gpu EXECUTOR=slurm WORKERS=4
```

The maximum simultaneous GPU allocation is approximately the number of concurrent GPU targets times `WORKERS`; keep that product within the allocation. To give one target all 16 GPUs, use `make fig-16 WORKERS=16` (and similarly for the other GPU targets).

On an already allocated multi-GPU host, bind each local target to a different GPU because local mode uses one GPU per target:

```bash
CUDA_VISIBLE_DEVICES=0 make fig-16 EXECUTOR=local WORKERS=1 &
CUDA_VISIBLE_DEVICES=1 make tbl-5 EXECUTOR=local WORKERS=1 &
CUDA_VISIBLE_DEVICES=2 make fig-17 EXECUTOR=local WORKERS=1 &
CUDA_VISIBLE_DEVICES=3 make fig-20 EXECUTOR=local WORKERS=1 &
wait
```

## Shared source layout

- `LLMCompass/` and `SCALE-Sim/`: CPU performance simulator snapshots.
- `hardware/`: Chisel RTL, generated SystemVerilog, synthesis-time RTL snapshots, and filtered native reports.
- `end2endacc/`: Figure 16 inference and evaluation harness.
- `OSTQuant/`: Table 5 workflow and SCNA integration.
- `train/`: shared Figure 17/20 function trainer and approximation baselines.
- `config/` and `requirements/`: GPU execution configuration and pinned accuracy environment.
- `validation/`: CPU paper-value and report validation.
- `tools/`: unified GPU evidence validation and archive creation.
- `PAPER_RESULTS.md`: paper-to-artifact result map.

## Correct experiment semantics

Figure 13 compares FlashAttention with INT8 softmax conversion against SCOPE with fused scale conversion. Figure 21 reports that conversion-fusion ablation separately. Figure 14 reuses Figure 13 rows through 32K and applies the fixed-tile model through 512K. Figure 15 models the B300 doubled-SFU sensitivity configuration.

Figure 18 extracts 112 filtered native Synopsys reports and fits constant per-PE area/power values across completed array sizes. Figure 19 is incremental overhead over a 32x32 baseline systolic array, calculated from the Figure 18 per-PE fit; it is not presented as completed full 32x32 synthesis for every design.

Figure 16 and Table 5 use the same model-level quality protocol: WikiText-2 perplexity plus the unweighted mean accuracy of ARC-Easy, HellaSwag, PIQA, and WinoGrande. Table 5 retains any additional raw task outputs only as diagnostics; they are not included in its reported accuracy.

## Hardware evidence without Synopsys

`make hardware` reparses the bundled Design Compiler V-2023.12 area, power, and timing reports, reproduces the Figure 18 fit, calculates Figure 19, verifies both CSVs, and redraws the figures. Synopsys and the TSMC 28 nm technology files are needed only to repeat synthesis; they are not required to inspect or reproduce the submitted plots. `make rtl` regenerates four current N=8 SCOPE/Pinnacle SystemVerilog configurations with Chisel.

## Archive

```bash
make archive
```

The archive excludes `.git/`, `.venv/`, `runs/`, caches, model weights, build targets, and machine-local `config/local.env`. It includes all compact CPU/GPU evidence, expected values, source snapshots, native synthesis reports, and the paper.
