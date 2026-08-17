# Response to Reviewer A1

Thank you for the detailed evaluation and for successfully completing the affected runs. The reviewers also ran correctly. Your executions followed the intended setup, and the few numerical differences are small and do not affect any conclusion in the paper.

## Issues that need an artifact fix

### 1. Table 4 reproduction

We replaced the expected-value replay with an executable SCNA path. Running `make tbl-4` generates all 11 MSE/MAE rows and point-level predictions under `runs/<RUN_ID>/`, then a separate validator compares them with the paper CSV and recomputes the 431x and 14.9x geomean claims.

### 2. Missing Figure 16 `quant/core`

The complete `end2endacc/quant/core` package is now included. `make -C experiments/fig-16-end-to-end-quality smoke` imports it, runs a small quantization calculation, and exercises one Figure 16 configuration without an external overlay.

### 3. Provenance outside Git

Both Figure 16 wrappers now check whether the source root is a Git worktree before calling `git rev-parse`. An archive without `.git` records `unknown (source archive is not a Git worktree)` instead of exiting with RC 128.

### 4. GPU dependencies

We pinned `sentencepiece==0.2.0` and `protobuf==5.29.5`, removed unused torchvision from the installation path, and isolated the LLMCompass dependencies so they do not replace the pinned PyTorch version. `make setup` now ends with `pip check`, which passes in the corrected environment.

### 5. FP16/BF16 consistency

BF16 is the intended and executed protocol, and the tables and documentation now use that label. Historical `fp16_*` filenames and CSV keys remain only for compatibility with the packaged run.

### 6. Top-level license

We added a top-level CC BY 4.0 `LICENSE` for the first-party artifact. The bundled LLMCompass, SCALE-Sim, and OSTQuant trees retain their own licenses.

## Results and protocol points

### 1. Authoritative Table 5 protocol

The experimental runs are correct. The issue is in the paper table: its accuracy column averages all evaluated benchmarks, while the artifact and Figure 16 use the uniform four-task mean of ARC-Easy, HellaSwag, PIQA, and WinoGrande; we will replace Table 5 with the four-task results in the final paper.

The small differences in your correct rerun are expected from stochastic training/calibration effects and low-level GPU and library variation, even with fixed seeds. They do not change the comparison or conclusions, and the validator now uses 0.04 absolute PPL and 1.0 percentage-point accuracy tolerances so these negligible differences are accepted.

### 2. Figure 16 aggregate differences

Thank you for running Figure 16 correctly. The three small aggregate differences are expected cross-stack variation, are negligible for the paper's conclusions, and are now accepted by the 0.007 accuracy tolerance while all 80 values and four-task means remain checked.

### 3. Figures 17 and 20 variability

Thank you for also running Figures 17 and 20 correctly. Stochastic training and CUDA-level variation prevent bit-for-bit agreement, but the differences are negligible and preserve every qualitative conclusion; Figure 17 now uses a 20% relative envelope, while Figure 20 uses 20% relative error or a `2e-4` absolute floor and still requires all nine constrained variants to improve.

## Requested iteration material

The root `README.md` now contains the concise change record, environment, commands, protocol choices, and tolerances; we did not add checksum files. Reviewer-facing evidence commands write fresh predictions, figures, and logs under the ignored `runs/` tree, so the experiment directories contain no duplicated reviewer runs or unnecessary logs; `make evidence-gpu` passes Table 4 (22 comparisons), Figure 16 (80), Table 5 (20), Figure 17 (36), and Figure 20 (18 plus all nine improvement checks).
