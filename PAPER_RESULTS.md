# Paper-to-Artifact Result Map

The paper source of truth is `paper/SCOPE-revision.pdf`. Bundled measurements are under each experiment's `actual-results/2026-07-13_ae-validation/` directory; paper targets are under `expected-results/`.

## CPU performance and hardware

- Table 3 → `experiments/tbl-3-integer-softmax/`: SCOPE H100 INT8 useful throughput is 1130.86/1526.51/1672.82/1713.89 TFLOP/s at 2K/4K/8K/16K.
- Figure 13 → `experiments/fig-13-prefill-attention/`: at 32K, FP16 speedups are 1.34x/1.34x/1.70x and INT8 speedups are 3.05x/2.51x/2.81x on B200/AWSv4/TPUv6e.
- Figure 14 → `experiments/fig-14-full-prefill/`: 512K FP16 speedups are 1.341x/1.329x/1.681x on B200/AWSv4/TPUv6e; INT8 speedups are 2.69x/1.28x/1.91x.
- Figure 15 → `experiments/fig-15-b300-sensitivity/`: longest-context FP16 attention/full-prefill gains are 1.09x/1.08x; INT8 gains are 1.94x/1.90x.
- Figure 18 → `experiments/fig-18-pe-area-power/`: SCOPE uses 1.09--1.44x area and 1.18--1.34x power per PE. Seventeen cells are constant least-squares fits of completed meshes; FSA FP8 uses the disclosed corrected N=4 hierarchy rows.
- Figure 19 → `experiments/fig-19-hardware-comparison/`: SCNA-8 has 12.8x geometric-mean area and 9.5x power reductions over plotted prior designs under the paper's 32x32 incremental-overhead accounting.
- Figure 21 → `experiments/fig-21-scale-fusion/`: longest reported scale-fusion gains reach 1.11x on B200, 1.97x on AWSv4, and 1.46x on TPUv6e.

## GPU accuracy and numerical precision

- Table 4 → `experiments/tbl-4-function-approximation-accuracy/`: paper values are preserved for evidence, but fresh reproduction is not claimed because the common raw baseline grid and unified harness are absent.
- Figure 16 → `experiments/fig-16-end-to-end-quality/`: 80/80 bundled perplexity and four-task mean-accuracy comparisons pass.
- Table 5 → `experiments/tbl-5-ostquant-quality/`: 16/16 corrected OSTQuant PPL/accuracy configurations pass.
- Figure 17 → `experiments/fig-17-neuron-scalability/`: 36/36 configurations pass; measured 32-vs-4 MSE gain is 97.2x--2837.8x. The saved Exp sweep exception is documented in the experiment README.
- Figure 20 → `experiments/fig-20-shape-constraints/`: 18/18 width-16 configurations pass; shape constraints improve MSE by 47.1x--2264.3x.

## RTL and synthesis provenance

Four current N=8 SCOPE/Pinnacle configurations can be regenerated from Chisel. The synthesis-time SystemVerilog snapshots and 112 filtered native Design Compiler V-2023.12 area, power, and timing report sets are bundled under `hardware/`.
