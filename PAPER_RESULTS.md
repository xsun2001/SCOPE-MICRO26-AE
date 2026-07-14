# Paper Experiment and Result Index

The following values were extracted from the evaluation section of `SCOPE-revision.pdf` and are the source of truth for this AE bundle.

## CPU-reproduced performance

- Figure 13, attention: FP16 geometric-mean speedup is reported as 1.28x; at 32K the device gains are approximately 1.34x (B200), 1.34x (AWSv4), and 1.70x (TPUv6e). INT8 maximum/32K gains are 3.05x, 2.51x, and 2.81x respectively.
- Figure 14, full prefill: at 32K, AWSv4/B200/TPUv6e achieve 1.207x/1.183x/1.473x. At 512K FP16 they achieve 1.329x/1.341x/1.681x; at 512K INT8 they achieve 1.28x/2.69x/1.91x.
- Figure 15, B300: at the longest context, FP16 attention/full-prefill gains are 1.09x/1.08x and INT8 gains are 1.94x/1.90x.
- Table 3, useful H100 INT8 attention throughput at 2K/4K/8K/16K: SCOPE is 1130.86/1526.51/1672.82/1713.89 TFLOP/s; I-LLM is 641.65/746.36/772.04/782.08 TFLOP/s; IntAttention is 888.73/1093.30/1170.41/1104.60 TFLOP/s.
- Figure 21, scale fusion: B200 rises from 1.06x at 2K to 1.11x at 32K; TPUv6e rises from 1.10x to 1.46x; AWSv4 is 1.12x/1.73x/1.91x/1.97x at 4K/8K/16K/32K.

## Hardware evidence and locally reproduced plots

- Figure 18: SCOPE adds 1.09--1.44x area and 1.18--1.34x power per PE. OneSA adds 1.59--1.98x area and 2.06--2.63x power; FuseMax reaches 8.05x area and 7.30x power.
- Figure 19: SCNA-8 has a 12.8x geometric-mean area reduction and 9.5x power reduction relative to the plotted prior designs. FP16 reductions span 5.1--52.9x area and 5.2--15.8x power; INT32 reductions span 2.5--35.7x area and 3.1--35.2x power.
- Seventeen Figure 18 cells are constant least-squares fits (arithmetic means) of per-PE samples from completed whole meshes. The FSA FP8 cell is the named hierarchy evidence `mesh_1_2` area = 1838.214 and `mesh_3_3` power = 0.588 mW from the filtered corrected N=4 report.
- RTL is elaborated from Chisel for four current N=8 SCOPE/Pinnacle data-type configurations. The paper-input SystemVerilog snapshots are retained separately with per-job hashes and provenance. Archived synthesis used Synopsys Design Compiler V-2023.12, TSMC 28 nm libraries, and a 1 GHz target.

## GPU/model-host experiments retained as paper claims only

- Table 4: the function-approximation geometric-mean MSE improvement is 431x over NN-LUT and 14.9x over T-LUT.
- Figure 20: shape constraints reduce MSE by 47.1--2264.3x (Exp 705.5x, Exp2 976.3x, Rsqrt 2264.3x, Sigmoid 54.8x, Erf 47.1x, Tanh 94.9x).
- Table 5, Figure 16, and Figure 17 cover low-bit quantization, end-to-end LLM accuracy/perplexity, and neuron-count convergence; they are not rerun on this CPU machine.
