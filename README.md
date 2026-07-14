# SCOPE Artifact-Evaluation Bundle

This is a standalone CPU artifact-evaluation bundle for `SCOPE-revision.pdf`. It contains the simulator snapshots, paper-matched experiment scripts, actual and expected outputs, plotting code, minimal Chisel RTL project, generated SystemVerilog, and archived Synopsys Design Compiler reports needed to audit the experiments executed on this host.

The CPU scope is Figures 13, 14, 15, 18, 19, and 21 plus Table 3. Model accuracy, approximation precision, training, perplexity, and quantization experiments are intentionally excluded from execution because they were performed on GPU/model-evaluation hosts.

## Quick start

```bash
make setup
make validate-packaged
make all
```

`make validate-packaged` checks the included `2026-07-13_ae-validation` outputs without rerunning the simulators. `make all` creates a new human-readable timestamp under every experiment's `actual-results/` directory, regenerates the four N=8 RTL configurations, and validates the new run. Use `RUN_ID=name`, `JOBS=n`, or `PYTHON=/path/to/python` to override defaults.

The default Figure 13 run recomputes the FlashAttention and SCOPE rows used by the paper. Set `FULL_BASELINE=1` to include the complete unfused baseline; its AWSv4 32K row can take more than 35 CPU-minutes and does not affect the reported SCOPE-over-FlashAttention claims.

## Make targets

```bash
make help
make performance
make hardware
make fig-13
make fig-14
make fig-15
make tbl-3
make fig-18
make fig-19
make fig-21
make rtl
make package
```

Each experiment can also be run directly, for example `make -C experiments/fig-18-pe-area-power run`. Figure 14 and Figure 21 consume the matching Figure 13 run; the top-level Makefile passes that dependency automatically.

## Bundle layout

- `experiments/`: one `fig-X-description` or `tbl-X-description` directory per evaluated paper result. Every directory contains its scripts, Makefile, README, actual results, expected results, and execution logs.
- `LLMCompass/` and `SCALE-Sim/`: source snapshots required by the CPU performance experiments. `LLMCompass/ae` is excluded because it belongs to the LLMCompass paper.
- `hardware/rtl/`: minimal Chisel project, four freshly generated N=8 SCOPE/Pinnacle designs, and the 112 hash-indexed paper synthesis-input snapshots.
- `hardware/synthesis/`: one filtered, human-named tree containing the native area, power, and timing reports used by Figures 18 and 19.
- `validation/`: paper-value and all-row archive comparisons.
- `PAPER_RESULTS.md`: rounded claims extracted from the paper.
- `AE_SUBMISSION.md`: AE-form text covering reproducible results and hardware, software, and data dependencies.
- `MANIFEST.sha256`: checksums for every bundled file.

## Correct comparison semantics

Figure 13 compares FlashAttention with INT8 softmax conversion against SCOPE without conversion because SCOPE fuses the scale conversion. Figure 21 reports the conversion-fusion ablation separately. Figure 14 reuses freshly generated Figure 13 rows through 32K and applies the fixed-tile model from 64K through 512K; B300 in Figure 15 is modeled directly as the doubled nonlinear/SFU-throughput sensitivity configuration.

## Hardware evidence without Synopsys

Run `make rtl` to force a fresh elaboration of four current N=8 designs. The paper-matching synthesis evidence is stored once under `hardware/synthesis/reports/`: 112 filtered result sets containing native `area`, `power`, and `timing` reports. `make hardware` reparses the reports, reproduces the Figure 18 per-PE fit, calculates Figure 19 for a 32x32 baseline SA, verifies the clean CSVs, and redraws both figures.

The reports record Synopsys DC V-2023.12, TSMC 28 nm libraries, and the 1 GHz target. Service-side console logs had already expired, so the bundle relies on native reports rather than claiming unavailable logs. Figure 18 fits completed N=4--N=28 samples as available; Figure 19's x16/x32 values are explicitly labeled as 32x32 incremental-overhead calculations rather than complete large-mesh syntheses. See `hardware/rtl/RTL_VERSION.md` for exact-versus-retained RTL provenance.

## Included validation

The packaged validation covers paper claims, all reproduced Figure 13/21 latency rows, all 54 Figure 14 paper CSV rows, all 18 Figure 15 rows, rendered hardware plots, generated RTL, all selected native synthesis reports, report-derived paper cells, and RTL provenance classes.
