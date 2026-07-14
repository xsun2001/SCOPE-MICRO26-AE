# Table 4: nonlinear approximation accuracy

`expected-results/paper_table4.csv` contains the exact MSE and MAE entries extracted from Table 4. The shared `../../train/` directory preserves every related implementation found in the workspace: the paper-era SCNA trainer, GQA-LUT, NLI, NN-LUT, and a separate Taylor diagnostic.

The experiment is evidence-only. The repository has no common evaluation grid or raw results that combine Taylor, Frac-T, Interp, Frac-I, LinearLUT, NN-LUT, T-LUT, and SCNA under the Table 4 protocol. Therefore `make reproduce` records `not-reproducible` instead of fabricating a comparison.

Generate the reviewer table with:

```bash
make evidence
```

Output: `actual-results/2026-07-13_ae-validation/generated/table4.md`.
