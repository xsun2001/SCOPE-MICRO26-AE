# Experiment Index

Every experiment uses `make run`, writes timestamped outputs to `actual-results/<RUN_ID>/`, and retains the paper-matched archive under `expected-results/`.

| Directory | Paper item | CPU action |
| --- | --- | --- |
| `fig-13-prefill-attention` | Figure 13 | Simulate FP16/INT8 prefill attention on B200, AWSv4, and TPUv6e and render throughput/speedup plots. |
| `fig-14-full-prefill` | Figure 14 | Extend attention to 512K, derive Llama 3 8B full-prefill latency, and render the main-device panel. |
| `fig-15-b300-sensitivity` | Figure 15 | Model the doubled-SFU B300 sensitivity configuration and render its panel. |
| `tbl-3-integer-softmax` | Table 3 | Model SCOPE, I-LLM, and IntAttention useful H100 INT8 throughput. |
| `fig-18-pe-area-power` | Figure 18 | Extract 112 filtered native report sets, fit per-PE values across completed array sizes, and render area/power. |
| `fig-19-hardware-comparison` | Figure 19 | Calculate x16/x32 incremental overhead over a 32x32 baseline SA and render the comparison. |
| `fig-21-scale-fusion` | Figure 21 | Render the INT8 scale-conversion fusion ablation from paired conditions. |

The shared simulator implementation is at the bundle root in `LLMCompass/` and `SCALE-Sim/`. The single filtered native-report tree is in `hardware/synthesis/reports/`; corresponding RTL snapshots are in `hardware/rtl/`.
