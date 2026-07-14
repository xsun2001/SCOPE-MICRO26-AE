# Figure 19: 32x32 Hardware Comparison

Yes: Figure 19 reports throughput-normalized incremental hardware overhead over a 32x32 baseline systolic array. It does not claim that every complete 32x32 design was synthesized. Large configurations are calculated from the Figure 18 per-PE fit because full synthesis is prohibitively slow.

Run both figures from the bundle root for the complete report-to-figure chain:

```bash
make hardware
```

Or run this directory alone with the bundled, verified Figure 18 fit:

```bash
make run
```

For area or power `R`, the local-design values are calculated from the paper-rounded Figure 18 CSV:

```text
SCNA-8  = 16 * (R_SCOPE   - R_Baseline)
SCNA-16 = 32 * (R_SCOPE   - R_Baseline)
OneSA   = 32 * (R_OneSA   - R_Baseline)
FuseMax = 32 * (R_FuseMax - R_Baseline)
FSA     = 32 * (R_FSA     - R_Baseline)
```

The multipliers are the paper's equal-throughput accounting for a 32x32 baseline SA: SCNA-8 uses half the augmented-PE budget of SCNA-16, and the comparison array-fused designs use the 32-PE budget. `reproduce.py` exposes the calculation directly. `literature.csv` contains only the standalone NN-LUT, T-LUT, and PICACHU values taken from their papers.

The generated `figure19.csv` has only four result columns: accumulation type, method, incremental area, and incremental power. The script verifies it against `expected-results/figure19.csv` and then draws `figure19.png` and `figure19.pdf`.

The local values trace to the filtered native reports under `../../hardware/synthesis/reports/` through Figure 18. The corresponding RTL snapshots are under `../../hardware/rtl/paper-mesh-snapshot/`.
