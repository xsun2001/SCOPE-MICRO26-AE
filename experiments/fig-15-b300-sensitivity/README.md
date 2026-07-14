# Figure 15: B300 Sensitivity

This experiment models the paper's B300 sensitivity configuration with doubled nonlinear/SFU throughput. At 512K, the paper reports FP16 attention/full-prefill gains of 1.09x/1.08x and INT8 gains of 1.94x/1.90x.

Run:

```bash
make run
```

`extend_attention_fixed_tiles.py` models B300 attention through 512K, `derive_full_model_from_attention.py` constructs the full-prefill result, and `plot_paper_speedup_figures.py --figures b300` creates the CSV, PNG, and PDF. Actual, expected, and log directories are included beside the scripts.
