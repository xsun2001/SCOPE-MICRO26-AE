# Figure 13: Prefill-Attention Performance

This experiment reproduces FP16 and INT8 prefill-attention results for B200, AWSv4, and TPUv6e at 2K--32K context. At 32K, the paper reports FP16 speedups of 1.34x, 1.34x, and 1.70x and INT8 speedups of 3.05x, 2.51x, and 2.81x, respectively.

Run:

```bash
make run
make run RUN_ID=my-run JOBS=4
make run FULL_BASELINE=1
```

`run_condition_parallel.py` and `simulate_prefill_attention.py` execute the simulator; the JSON files in `configs/` select only paper devices; `generate_paper_figures.py` creates PNG/PDF panels. By default the expensive unfused baseline is omitted because it is not used in the reported SCOPE/FlashAttention speedups.

Exact archived paper inputs are in `expected-results/`. Fresh outputs are written to the ignored `actual-results/<RUN_ID>/` tree, and new console logs are written under the ignored root `runs/` directory.
