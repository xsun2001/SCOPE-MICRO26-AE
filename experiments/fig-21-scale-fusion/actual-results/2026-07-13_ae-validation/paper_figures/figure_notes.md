# Paper Figure Notes

- GPU source suite: `/home/CONNECT/cxu930/Documents/pinn-fullstack/ae-exp/experiments/fig-21-scale-fusion/actual-results/2026-07-13_ae-validation/b200`
- AWS source suite: `/home/CONNECT/cxu930/Documents/pinn-fullstack/ae-exp/experiments/fig-21-scale-fusion/actual-results/2026-07-13_ae-validation/awsv4`
- TPU source suite: `/home/CONNECT/cxu930/Documents/pinn-fullstack/ae-exp/experiments/fig-21-scale-fusion/actual-results/2026-07-13_ae-validation/tpuv6e`
- Unique source suites: `/home/CONNECT/cxu930/Documents/pinn-fullstack/ae-exp/experiments/fig-21-scale-fusion/actual-results/2026-07-13_ae-validation/b200`, `/home/CONNECT/cxu930/Documents/pinn-fullstack/ae-exp/experiments/fig-21-scale-fusion/actual-results/2026-07-13_ae-validation/tpuv6e`, `/home/CONNECT/cxu930/Documents/pinn-fullstack/ae-exp/experiments/fig-21-scale-fusion/actual-results/2026-07-13_ae-validation/awsv4`
- Requested-only generation: `1`
- Selected devices: `b200, awsv4, tpuv6e`
- `figure_paper_requested_e2e_latency_lineary`: requested all-device 2x3 grid using baseline, FlashAttention with conversion, and CustomSA without conversion, with linear y-axis.
- `figure_paper_requested_e2e_latency_logy`: same requested all-device 2x3 grid with logarithmic y-axis.
- `figure_paper_requested_e2e_throughput`: requested all-device 2x3 grid of useful attention GEMM throughput in TFLOPS using only `QK + AV` FLOPs divided by end-to-end latency, with linear y-axis, annotated above the SCNA line with SCNA speedup over FlashAttention at the same prefill length.
- `figure_paper_requested_e2e_throughput_with_speedup`: requested all-device 4x3 composite figure; for each dtype row, throughput curves are shown above and SCNA-vs-FlashAttention speedup bars are shown directly below with a `0.7/0.3` height split.
- `figure_paper_requested_speedup_vs_flashattention`: requested 1x3 grouped-bar figure showing CustomSA speedup over FlashAttention by prefill length for FP16 and INT8.
- `figure_paper_requested_customsa_conversion_ablation`: requested 1x3 figure showing CustomSA type-conversion overhead ratio versus no-conversion across prefill lengths.
- All latencies come from runs with `--ignore-hbm-bottleneck`, so the figures emphasize end-to-end compute and on-chip scheduling behavior rather than HBM throttling.
- Throughput is computed as `(QK FLOPs + AV FLOPs) / total_latency_s`, intentionally excluding softmax and exp work so the figure focuses on useful GEMM throughput.
- Each SCNA throughput point is annotated with `SCNA / FlashAttention` throughput speedup at the same prefill length.
- In the composite throughput-with-speedup figure, the top subpanel keeps only the curves; the speedup labels are moved into the lower bar-chart subpanel.
- The GPU source suite also uses `--ignore-onchip-io-bottleneck`, so the GPU figures reflect compute-limited behavior after removing the on-chip/global-buffer bottleneck.
