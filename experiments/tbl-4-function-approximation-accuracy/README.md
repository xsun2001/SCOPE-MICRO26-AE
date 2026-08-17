# Table 4: nonlinear approximation accuracy

`expected-results/paper_table4.csv` contains the MSE and MAE entries from Table 4. The artifact independently evaluates archived 32-unit SCNA checkpoints for all 11 nonlinear functions. The execution command does not read that CSV; `scripts/audit.py` receives it only after raw predictions have been written. The other method columns are literature reference values.

`data/scna_checkpoints.json` contains the 11 fixed configurations evaluated by the command. For GeLU, SCNA approximates `erf(x/sqrt(2))+1`; the outer `x/2` multiply is exact.

The 431x NN-LUT geomean follows the paper protocol over the nine general
nonlinearities (Exp and Exp2 are excluded); the 14.9x T-LUT geomean uses the
nine rows with numeric T-LUT entries. The audit recomputes both ratios from the
fresh SCNA MSE values and the literature columns.

Run the actual reproduction with:

```bash
make reproduce
```

It emits `scna_metrics.csv`, point-level `raw_predictions.csv`,
`validation.json`, and the reviewer table under
`runs/<run-id>/tbl-4-function-approximation-accuracy/`.

Run the same CPU evaluation as part of the hardware-free evidence path with:

```bash
make evidence
```

Output: `runs/<run-id>/evidence/tbl-4-function-approximation-accuracy/generated/table4.md`. The ignored staging directory keeps `make evidence` worktree-clean.
