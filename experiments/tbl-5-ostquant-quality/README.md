# Table 5: OSTQuant low-bit model quality

This directory is the complete corrected Table 5 workflow for Llama-2-7B and Llama-3-8B under W6A6 and W4A4. It trains or reuses four exact SDPA OST transforms, generates four GPTQ `qmodel.pt` files, and evaluates exact eager attention plus SCNA-8/16/32 with the corrected causal mask.

Table 5 reports WikiText-2 perplexity and the unweighted mean accuracy over ARC-Easy, HellaSwag, PIQA, and WinoGrande, matching Figure 16. The replacement table includes OSTQuant, SCNA-8/16/32, and the FP16 baseline for both models and bit-widths. Additional task outputs retained from the original sweep are diagnostics only and do not enter the reported average.

- `../../OSTQuant/`: shared OSTQuant source with the SCNA integration and causal-mask fix.
- `data/`: four checkpoint-generation cases and sixteen paper evaluation cases.
- `expected-results/four_task_table5.csv`: replacement Table 5 targets under the unified four-task protocol.
- `actual-results/2026-07-13_ae-validation/`: 16 fresh OSTQuant/SCNA evaluations, four FP16 baseline entries, 20/20 validated table entries, reports, logs, compact metric JSONs, and the generated table. Large `model.bin` and `qmodel.pt` intermediates are intentionally outside the portable archive.
- `scripts/`: exact training, GPTQ generation, evaluation, collection, validation, and table generation.

Use `make evidence` to verify the bundled result and regenerate the table under `runs/<run-id>/evidence/tbl-5-ostquant-quality/generated/table5.md` without a GPU. The ignored staging directory keeps the packaged evidence and worktree unchanged.

For a full fresh run:

```bash
make reproduce
```

For direct execution on an already allocated GPU:

```bash
make reproduce EXECUTOR=local WORKERS=1
```

A fresh run needs an NVIDIA CUDA GPU from the Ampere generation or newer with at least 80 GB of device memory. H100 80 GB (Hopper) is the validated reference, not a device lock; an A100 80 GB (Ampere), H100/H200 (Hopper), or B100/B200 (Blackwell) is suitable with a compatible PyTorch/CUDA installation. The included timestamps measure 14.6 total GPU-hours on H100: 7.6 hours for exact-checkpoint preparation, 0.7 hours for quantized-checkpoint generation, and 6.3 hours for the evaluations. The documented 15-worker Slurm execution (one H100 per worker) takes about 3 hours of wall time, excluding downloads and queue delay.

To reuse expensive intermediates, set `TABLE5_CHECKPOINT_SOURCE` to a directory containing the four `exact_*_sdpa` and four `qmodel_*_sdpa` folders. If it is empty, the Makefile regenerates them from the model weights before the sixteen evaluations.

Only results from the corrected `2026-06-12_140246_corrected_protocol` lineage and the fresh 2026-07-13/14 rerun are included. Older eager-attention runs had a causal-mask alignment bug and are deliberately excluded. The paper-aligned rows use full-precision SCNA inputs (`scna_input_quant_bits=0`) on top of the W6A6/W4A4 OSTQuant backbone.

Fresh evaluation commands pass the four task names explicitly. For the packaged rerun, `scripts/derive_four_task_metrics.py` promotes the retained four-task subset to the official result and preserves the wider raw sweep under `all_task_diagnostics` for provenance.
