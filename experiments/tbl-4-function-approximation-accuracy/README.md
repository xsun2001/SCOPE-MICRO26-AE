# Table 4: nonlinear approximation accuracy

`expected-results/paper_table4.csv` contains the revised final-paper MSE and MAE entries for the primary SCNA-16 Table 4. `expected-results/scna32_reference_table4.csv` retains the SCNA-32 reference values. The execution command does not read either expected-results CSV; `scripts/audit.py` receives them only after raw predictions have been written. The other method columns are literature reference values.

`data/scna_parameters.json` embeds the fused trained weights and biases for both widths, so evaluation has no checkpoint-file or hash dependency. Each row keeps its original checkpoint path as provenance metadata, but the evaluator reads parameters directly from the manifest. SCNA-16 is the default variant. The SCNA-16 Softplus and GeLU runs are from `20260822-table4-softplus-gelu-scna16`; their SCNA-32 references are from the corresponding `scna32` run. For GeLU, SCNA approximates `erf(x/sqrt(2))+1`; the outer `x/2` multiply is exact. `data/REVISED_TABLE4_DATA.md` is the single data document containing the old table, primary SCNA-16 revision, and SCNA-32 reference.

Rsqrt is evaluated as `1 / sqrt(-x)` on `[-256, -1]`, independently checked point by point by the audit. Reciprocal remains a separate trainer target and is not substituted for the Table 4 Rsqrt row.

Both evaluations match all 22 rounded values in their respective expected-results CSVs within the unchanged symmetric ±10% audit.

The revised NN-LUT comparison uses all 11 functions, consistent with the final
paper wording; the T-LUT comparison uses the nine rows with numeric T-LUT
entries. Corrected SCNA-16 MSE values produce 360.83x and 14.87x improvements,
respectively. The SCNA-32 reference produces 835.53x and 31.48x.

Run the actual reproduction with:

```bash
make reproduce
```

It emits `scna_metrics.csv`, point-level `raw_predictions.csv`, and
`validation.json` under the `scna16/` and `scna32/` subdirectories, plus one
combined reviewer table under
`runs/<run-id>/tbl-4-function-approximation-accuracy/generated/`.

The evaluator selects SCNA-16 when `--variant` is omitted. Run SCNA-32 alone
with `scripts/reproduce_scna.py --variant scna32 --output-dir <dir>`.

Run the same CPU evaluation as part of the hardware-free evidence path with:

```bash
make evidence
```

Output: `runs/<run-id>/evidence/tbl-4-function-approximation-accuracy/generated/table4.md`. The ignored staging directory keeps `make evidence` worktree-clean.
