# Table 4: nonlinear approximation accuracy

`expected-results/paper_table4.csv` contains the MSE and MAE entries from Table 4. The artifact reproduces our SCNA accuracy for 11 nonlinear functions. The other method columns are literature reference values; please refer to the corresponding baseline papers for their reproduction procedures.

The shared `../../train/` directory contains the SCNA trainer and the approximation implementations supplied with the artifact.

Generate the reviewer table with:

```bash
make evidence
```

Output: `runs/<run-id>/evidence/tbl-4-function-approximation-accuracy/generated/table4.md`. The ignored staging directory keeps `make evidence` worktree-clean.
