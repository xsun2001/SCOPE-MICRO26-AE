# Table 3: Integer Softmax Throughput

This experiment models useful H100 INT8 attention throughput for SCOPE, I-LLM, and IntAttention at 2K, 4K, 8K, and 16K context. The paper values are 1130.86/1526.51/1672.82/1713.89 TFLOP/s for SCOPE, 641.65/746.36/772.04/782.08 for I-LLM, and 888.73/1093.30/1170.41/1104.60 for IntAttention.

Run:

```bash
make run
```

`extend_attention_fixed_tiles.py --int-softmax-comparison` writes the latency and speedup CSVs. The validator converts modeled latency to useful attention TFLOP/s. `actual-results/` contains the latest run; `expected-results/` includes the paper-matched attention/full-model data and the other integer-softmax values used for audit.
