# SCOPE Artifact Evaluation Submission

## Key Results to be Reproduced

The artifact reproduces the CPU performance, hardware-report analysis, and RTL-generation results below. Model training, approximation-accuracy, perplexity, zero-shot accuracy, and quantization-quality experiments are outside this CPU artifact and are not claimed for the Results Reproduced badge.

- **Figure 13 — Prefill-attention performance.** Reproduces FP16 and INT8 SCOPE-versus-FlashAttention results on modeled B200, AWSv4, and TPUv6e devices for 2K--32K context. At 32K, the reported FP16 speedups are 1.34x, 1.34x, and 1.70x; the INT8 speedups are 3.05x, 2.51x, and 2.81x, respectively.
- **Figure 14 — Llama 3 8B full-prefill performance.** Reproduces end-to-end modeled prefill results through 512K context. At 512K, the reported FP16 speedups on AWSv4, B200, and TPUv6e are 1.329x, 1.341x, and 1.681x; the INT8 speedups are 1.28x, 2.69x, and 1.91x.
- **Figure 15 — B300 sensitivity.** Reproduces the doubled-SFU-throughput sensitivity study. At 512K, the reported attention/full-prefill speedups are 1.09x/1.08x for FP16 and 1.94x/1.90x for INT8.
- **Table 3 — Integer-softmax throughput.** Reproduces useful modeled H100 INT8 attention throughput for SCOPE, I-LLM, and IntAttention at 2K, 4K, 8K, and 16K. SCOPE obtains 1130.86, 1526.51, 1672.82, and 1713.89 TFLOP/s.
- **Figure 18 — Per-PE area and power.** Extracts 112 native synthesis report sets, computes `whole_array / N^2`, fits the per-PE values across completed array sizes, verifies the paper-rounded CSV, and redraws the figure. SCOPE requires 1.09--1.44x baseline area and 1.18--1.34x baseline power per PE.
- **Figure 19 — 32x32 hardware comparison.** Reproduces throughput-normalized incremental overhead over a 32x32 baseline systolic array from the verified Figure 18 fit and the bundled literature values. SCNA-8 provides geometric-mean reductions of 12.8x in area and 9.5x in power relative to the plotted competing designs. The large-array values are explicitly calculated from fitted per-PE results; they are not presented as completed full 32x32 synthesis runs.
- **Figure 21 — Scale-conversion fusion.** Reproduces the INT8 scale-fusion ablation. The reported speedup reaches 1.11x on B200, 1.97x on AWSv4, and 1.46x on TPUv6e at the longest reported context for each device.
- **RTL generation and synthesis-report audit.** Regenerates four current N=8 SCOPE/Pinnacle SystemVerilog configurations from Chisel. The artifact also checks the archived Design Compiler area, power, and timing reports without requiring reviewers to possess a Synopsys license.

The complete CPU experiment set is run with:

```bash
make setup
make all
```

The included paper-matched results can be checked without rerunning the simulations with:

```bash
make validate-packaged
```

Figures 18 and 19 alone can be reproduced from the native reports with:

```bash
make hardware
```

## Hardware Dependencies

Minimum practical configuration:

- 64-bit Linux workstation or compute node.
- 8 logical CPU cores.
- 16 GB system memory.
- 20 GB free disk space for the extracted artifact, Python environment, dependency caches, and RTL build outputs.

A 32-core machine with 64 GB memory is recommended for a faster full rerun. The Makefiles use up to eight parallel workers by default and accept `JOBS=<n>` for smaller or larger hosts. A lower-core-count machine can run the artifact but will take longer.

No GPU, FPGA board, microcontroller, or particular CPU model is required. All B200, B300, H100, AWSv4, and TPUv6e results in this artifact are obtained from the bundled analytical/cycle-level device models; access to those physical accelerators is not required.

The optional `FULL_BASELINE=1` Figure 13 mode evaluates an additional unfused baseline. It is CPU intensive and is not needed to reproduce the paper's reported SCOPE-over-FlashAttention results.

## Software Dependencies

The artifact is tested on 64-bit Linux with the following toolchain:

- GNU Make 4.2.1 and Bash.
- Python 3.12.8; Python 3.10 or newer is expected to work.
- `uv` 0.9.16 for creating the virtual environment and installing dependencies.
- Python packages listed in the bundled requirements files, including PyTorch, torchvision, NumPy, SciPy, pandas, Numba, Matplotlib, Seaborn, Cython, `absl-py`, `tqdm`, and SCALE-Sim. LLMCompass and SCALE-Sim source snapshots are included in the artifact.
- OpenJDK 21 and an sbt launcher for RTL regeneration. The RTL project pins sbt 1.12.5, Scala 2.13.18, Chisel 7.9.0, and ScalaTest 3.2.15.
- Standard command-line utilities used by packaging and validation: `tar`, `find`, and `sha256sum`.

Network access is normally needed only during `make setup` and the first sbt invocation to obtain Python and Maven dependencies. Reviewers may instead use pre-populated package caches or provide an existing Python environment through `PYTHON=/path/to/python`.

No proprietary software is required to reproduce or validate the submitted results. The archived synthesis was originally performed with **Synopsys Design Compiler V-2023.12** and TSMC 28 nm libraries at a 1 GHz target. Those proprietary tools and technology files are required only if a reviewer wishes to repeat synthesis from RTL. They are not required for the artifact workflow: the native area, power, and timing reports are included, and the provided scripts extract, fit, verify, and plot their results directly.

## Data Dependencies

All experiment inputs needed by the claimed CPU evaluation are included in the artifact:

- The exact LLMCompass and SCALE-Sim source snapshots used by the experiments.
- Device-model configurations for B200, B300, H100, AWSv4, and TPUv6e.
- Attention and full-prefill workload configurations, SCALE-Sim lookup tables, and paper-matched expected CSVs.
- The filtered set of 112 native Design Compiler area, power, and timing report sets used by Figures 18 and 19.
- Synthesis-time SystemVerilog snapshots, the current Chisel RTL generator, and generated RTL examples.
- The literature values used only for the standalone NN-LUT, T-LUT, and PICACHU bars in Figure 19.
- A copy of the submitted paper, `SCOPE-revision.pdf`.

No external benchmark dataset, model checkpoint, pretrained weight file, trace, accelerator access, or proprietary device model is required for the claimed results. WikiText-2, LLM checkpoints, approximation-training data, and GPU-based accuracy/precision experiments are not needed because Figures 16, 17, and 20 and Tables 4 and 5 are explicitly outside this CPU artifact's reproducibility scope.
