# SCOPE Artifact Evaluation Bundle

This repository is the unified reviewer bundle for `paper/SCOPE-revision.pdf`. It combines the CPU-host performance, RTL, and synthesis evidence with the GPU-host accuracy and numerical-precision experiments. All paper experiment directories use the `fig-X-description` or `tbl-X-description` naming convention and contain a Makefile, README, scripts, bundled actual results, and expected paper results.

Submission-facing result and dependency information is in `AE_SUBMISSION.md`.

## Quick start: no GPU or proprietary tools

```bash
make setup
make evidence
make validate
```

`make evidence` regenerates the compact GPU-side figures/tables from bundled results and audits the packaged CPU evidence. `make validate` checks both suites: the CPU validator currently reports 68/68 checks passing, and the GPU validator checks Figure 16 (80 comparisons), Table 5 (16), Figure 17 (36), Figure 20 (18), and the evidence-only Table 4 status.

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

Run both suites with `make reproduce`. Table 4 remains evidence-only because the shared evaluation grid and raw common-protocol baseline outputs are unavailable.

## Experiment index

| Paper item | Directory | Fresh execution |
| --- | --- | --- |
| Table 3 | `experiments/tbl-3-integer-softmax` | CPU |
| Figures 13, 14, 15, 21 | matching `experiments/fig-*` directories | CPU |
| Figures 18 and 19 | matching `experiments/fig-*` directories | report extraction/calculation on CPU |
| Figure 16 | `experiments/fig-16-end-to-end-quality` | NVIDIA GPU |
| Figure 17 | `experiments/fig-17-neuron-scalability` | CUDA GPU |
| Figure 20 | `experiments/fig-20-shape-constraints` | CUDA GPU |
| Table 5 | `experiments/tbl-5-ostquant-quality` | NVIDIA H100-class GPU |
| Table 4 | `experiments/tbl-4-function-approximation-accuracy` | evidence-only |

Every experiment stores bundled measurements under `actual-results/<validated-run>/` and paper targets under `expected-results/`. Fresh CPU runs use the same timestamped experiment directories. Fresh GPU runs are written under the ignored `runs/<RUN_ID>/` tree because Table 5 intermediates can exceed 60 GB.

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

The GPU workflows preserve their documented protocol limitations and provenance. In particular, Figure 16's full-precision-labeled run uses BF16, and Table 5 includes only the corrected causal-mask lineage.

## Hardware evidence without Synopsys

`make hardware` reparses the bundled Design Compiler V-2023.12 area, power, and timing reports, reproduces the Figure 18 fit, calculates Figure 19, verifies both CSVs, and redraws the figures. Synopsys and the TSMC 28 nm technology files are needed only to repeat synthesis; they are not required to inspect or reproduce the submitted plots. `make rtl` regenerates four current N=8 SCOPE/Pinnacle SystemVerilog configurations with Chisel.

## Archive

```bash
make archive
```

The archive excludes `.git/`, `.venv/`, `runs/`, caches, model weights, build targets, and machine-local `config/local.env`. It includes all compact CPU/GPU evidence, expected values, source snapshots, native synthesis reports, and the paper.
