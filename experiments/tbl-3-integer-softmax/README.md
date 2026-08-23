# Table 3: Integer Softmax Throughput

This experiment models useful H100 INT8 attention throughput for SCOPE, I-LLM, and IntAttention at 2K, 4K, 8K, and 16K context. The paper values are 1130.86/1526.51/1672.82/1713.89 TFLOP/s for SCOPE, 641.65/746.36/772.04/782.08 for I-LLM, and 888.73/1093.30/1170.41/1104.60 for IntAttention.

Run:

```bash
make run
```

`extend_attention_fixed_tiles.py --int-softmax-comparison` reproduces the paper table from the archived H100 calibration ratios and writes the latency and speedup CSVs. The validator converts modeled latency to useful attention TFLOP/s and checks the exact paper values. Fresh runs go to the ignored `actual-results/` tree; `expected-results/` includes the paper-matched attention/full-model data and the other integer-softmax values used for audit.

## Physical softmax microbenchmark

`bench_softmax_h100.py` is an optional physical calibration check, not the exact Table 3 result generator. It contains fused Triton kernels for an online FP softmax baseline, I-LLM DI-Exp/clipped DI-Softmax, IntAttention 5-bit IndexSoftmax, and a one-read/one-write memory diagnostic. Each kernel reads the same INT8 logits and writes U8 probabilities. The default sweep uses 67,108,864 elements at every sequence length from 2K through 512K, checks the FP kernel's U8 output against Torch, performs five warmups, and records 20 repetitions.

On an allocated H100 with PyTorch and Triton installed, run:

```bash
make microbenchmark RUN_ID=<new-run>
make validate-microbenchmark REFERENCE_MICROBENCHMARK="$PWD/actual-results/<new-run>/softmax-microbenchmark"
```

The compact calibration dataset is under `data/h100-softmax-microbenchmark/`. `raw_samples.csv` stores every CUDA-event timing; `softmax_latency.csv` stores medians, means, sample standard deviations, throughput, and ratios; `metadata.json` records the software/device configuration. `make validate-microbenchmark` rejects missing, duplicate, or extra repetitions and recomputes every reported statistic from the raw samples. Physical timings vary with clocks, thermal state, software versions, and GPU contention, so fresh user runs are expected to validate the calibration trend rather than match the paper TFLOP/s bit-for-bit.
