# SCOPE Artifact Evaluation bundle

This directory is a standalone reviewer bundle for the accuracy and numerical-precision claims in `paper/SCOPE-revision.pdf`. It contains the relevant workspace code, exact experiment matrices, paper targets, fresh actual results, validators, and figure/table generators. End-to-end performance, simulator, synthesis, area, power, and hardware experiments are intentionally absent because they were not executed on this GPU host.

Submission-facing result and dependency information is summarized in `AE_SUBMISSION.md`.

## Layout

```text
ae-exp/
├── Makefile
├── end2endacc/
├── OSTQuant/
├── train/
├── config/
├── paper/SCOPE-revision.pdf
├── experiments/
│   ├── tbl-4-function-approximation-accuracy/
│   ├── fig-16-end-to-end-quality/
│   ├── tbl-5-ostquant-quality/
│   ├── fig-17-neuron-scalability/
│   └── fig-20-shape-constraints/
├── requirements/
└── tools/
```

Each relevant implementation directory retains its workspace name and is stored once, directly at the bundle's first layer: `end2endacc/`, `OSTQuant/`, and `train/`. Each experiment directory contains only its run matrix/configuration in `data/`, expected and actual results, thin runner and generation scripts, generated artifacts, a Makefile, and a README. No experiment contains a private code copy, and runtime scripts do not reach back into the parent research checkout.

## Quick start

Inspect and regenerate all bundled evidence without a GPU:

```bash
make setup
make evidence
make validate
```

Configure a reviewer machine:

```bash
cp config/local.env.example config/local.env
# Edit MODEL_ROOT and execution settings.
```

Slurm is the default executor. Run one experiment with `make fig-16`, `make tbl-5`, `make fig-17`, or `make fig-20`. To bypass Slurm on an already allocated GPU:

```bash
make fig-17 EXECUTOR=local WORKERS=1
```

Run the complete accuracy/precision set with `make reproduce`. Table 4 remains evidence-only because the workspace lacks the unified baseline harness and raw shared-grid outputs.

## Dependencies and data

The validated host environment is recorded in `requirements/accuracy.txt`; OSTQuant's original environment snapshot is also preserved under `OSTQuant/`. Pretrained Llama, OPT, and Qwen weights are not redistributed. Set `MODEL_ROOT` to a directory containing the model folders named in `config/local.env.example`. Hugging Face datasets use the normal local/download cache.

Table 5 intermediates are roughly 60 GB. They are not part of the portable archive. A reviewer may either regenerate them or set `TABLE5_CHECKPOINT_SOURCE` to a cache with the four exact and four GPTQ checkpoint directories.

## Validated results

- Figure 16: pass, 80/80 plotted PPL and four-task mean-accuracy comparisons.
- Table 5: pass, 16/16 PPL/accuracy configurations.
- Figure 17: pass, 36/36 constrained best-MSE configurations; 97.2×–2837.8× 32-vs-4 gain.
- Figure 20: pass, 18/18 width-16 configurations; 47.1×–2264.3× shape-constraint gain.
- Table 4: paper values preserved, fresh reproduction not claimed because required shared baseline material is absent.

Use `make archive` to create `../scope-ae-bundle.tar.gz` and its SHA-256 sidecar. The archive excludes `runs/`, `.venv/`, `cache/`, and machine-local `config/local.env`; it includes all reviewer code, compact actual evidence, expected results, histories required to regenerate Figures 17/20, and the paper.
