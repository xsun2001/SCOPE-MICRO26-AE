# Response to Reviewer A3

Thank you for the careful follow-up and for identifying these remaining inconsistencies. We corrected the artifact paths that affect reproducibility and made the corresponding wording and documentation corrections in the revised paper. **The revised parts in our paper are highlighted in orange.** Our responses are below.

## 1. Table 4 parameter provenance

We agree that the nine `curvature_spline` entries in artifact v2 were misleadingly described as trained SCNA checkpoints. We removed these recipes and revised Table 4.
The corrected Table 4 path evaluates explicit fused weights and biases from trained SCNA models for all 11 functions. `experiments/tbl-4-function-approximation-accuracy/data/scna_parameters.json` directly embeds every trained parameter value for both the primary SCNA-16 table and a separate SCNA-32 reference. It also records the training seed, target semantics, input range, selection rule, and original source-checkpoint path for each row.

```bash
make -C experiments/tbl-4-function-approximation-accuracy evidence
```

now independently evaluates SCNA-16 and SCNA-32 from the embedded parameters, writes point-level predictions, and checks 22 MSE/MAE values for each width. The SCNA-16 result is the revised Table 4; SCNA-32 is retained as a reference. The updated paper now includes the revised Table 4.

## 2. Rsqrt semantics and affected reruns

We agree that the old trainer incorrectly used reciprocal semantics for the Rsqrt row. We separated the two functions in the shared trainer:

- `rsqrt`: `1 / sqrt(-x)` on `[-256, -1]`;
- `recip`: `-(1 / x)` on `[-16, -1]`.

We retrained all affected Rsqrt configurations: SCNA-4/8/16/32 for Figure 17, constrained and unconstrained SCNA-16 for Figure 20, and the SCNA-16/32 Table 4 rows. The revised primary Table 4 Rsqrt result is MSE `1.2212449e-6` and MAE `7.4120480e-4`; the SCNA-32 reference is MSE `1.7109673e-7` and MAE `2.7280003e-4`. The affected figures and table were regenerated with the corrected Rsqrt semantics and included in the revised paper.

## 3. Scope of the NN-LUT geomean

We agree that excluding Exp and Exp2 was inconsistent with the phrase “across all functions.” The revised NN-LUT geomean includes all 11 rows with numeric NN-LUT values. With the corrected primary SCNA-16 results, the MSE improvement is `360.83x`, reported as `361x` in the revised table. We replaced the paper's `431x` claim with `361x` in the abstract, contribution summary, table data, evaluation, and conclusion.

## 4. Figure 17 monotonicity wording

We agree. The experiment supports the overall capacity-scaling conclusion, but it does not support a universal monotonicity claim between every adjacent width. In particular, Exp can have higher MSE at width 32 than at width 16 even though width 32 remains substantially better than width 4. We corrected the wording in Section 7.3.4.

## 5. Final-submission documentation

Confirmed. BF16 is the executed full-precision protocol for the Figure 16 and Table 5 model-quality experiments. We changed the artifact's user-facing Table 5 label, summaries, provenance fields, and validation interface from FP16 to BF16. The revised paper uses `BF16 Baseline` in Table 5 and “BF16 and INT8” in the associated model-quality discussion. The DOI number is also updated.
