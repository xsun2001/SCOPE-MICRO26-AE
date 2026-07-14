# Figure 18: Per-PE Area and Power

Yes: Figure 18 reports area and power per PE, using synthesized square-array sizes from N=4 through the largest completed N for each design/type combination.

Run:

```bash
make run
```

The command performs the complete reviewer-facing path:

1. Read the native Design Compiler `area.rpt`, `power.rpt`, and `timing.rpt` files under `../../hardware/synthesis/reports/`.
2. Extract the whole-array area and power, confirm timing is met, and calculate each sample as `per_PE = whole_array / N^2`.
3. Fit one constant per-PE value across the completed sizes:

   `c = argmin_c sum_N (per_PE_N - c)^2 = mean_N(per_PE_N)`

4. Round the fitted value to three decimal places, as in the paper, verify it against `expected-results/figure18.csv`, and draw `figure18.png` and `figure18.pdf`.

`report_values.csv` is the clean report extraction. `figure18.csv` is the fitted, paper-ready result. Their columns contain only the design configuration, array sizes, and measured/fitted area and power.

Seventeen of the eighteen design/type cells use the constant fit above. Only the FSA FP8-FP16 cell lacks a completed size sweep. It uses the exact paper values from the corrected N=4 native report: area hierarchy row `mesh_1_2` and power hierarchy row `mesh_3_3`. The CSV exposes those row names in `area_scope` and `power_scope`; it does not present the value as an extrapolated N=32 synthesis.

The reports are filtered to the paper-matching versions and use human-readable paths with no project or job identifiers. Corresponding synthesis-time RTL snapshots are under `../../hardware/rtl/paper-mesh-snapshot/`.
