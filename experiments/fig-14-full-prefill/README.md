# Figure 14: Llama 3 8B Full Prefill

This experiment derives end-to-end Llama 3 8B prefill latency for B200, AWSv4, and TPUv6e from 2K through 512K context. The paper reports 32K FP16 speedups of 1.183x, 1.207x, and 1.473x and 512K FP16 speedups of 1.341x, 1.329x, and 1.681x; 512K INT8 speedups are 2.69x, 1.28x, and 1.91x.

From the bundle root, `make fig-14` first ensures a matching Figure 13 run. Direct invocation accepts an explicit source:

```bash
make run FIG13_RUN_DIR=../fig-13-prefill-attention/actual-results/2026-07-13_ae-validation
```

`extend_attention_fixed_tiles.py` uses fresh Figure 13 rows through 32K and models 64K--512K. `derive_full_model_from_attention.py` adds the remaining Llama 3 8B prefill operations. The two plotting scripts generate the intermediate speedup CSV and paper panel.

Actual results and expected paper-matched results are under `actual-results/` and `expected-results/`; new console logs go under the ignored root `runs/` directory. The final comparison file is `paper_figures/paper_main_e2e_speedups.csv` inside each run.
