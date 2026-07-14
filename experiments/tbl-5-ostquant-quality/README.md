# Table 5: OSTQuant low-bit model quality

This directory is the complete corrected Table 5 workflow for Llama-2-7B and Llama-3-8B under W6A6 and W4A4. It trains or reuses four exact SDPA OST transforms, generates four GPTQ `qmodel.pt` files, and evaluates exact eager attention plus SCNA-8/16/32 with the corrected causal mask.

- `../../OSTQuant/`: shared OSTQuant source with the SCNA integration and causal-mask fix.
- `data/`: four checkpoint-generation cases and sixteen paper evaluation cases.
- `expected-results/`: exact Table 5 targets and corrected-protocol provenance.
- `actual-results/2026-07-13_ae-validation/`: 16/16 fresh comparisons, reports, logs, compact raw metric JSONs, and generated table. Large `model.bin` and `qmodel.pt` intermediates are intentionally outside the portable archive.
- `scripts/`: exact training, GPTQ generation, evaluation, collection, validation, and table generation.

Use `make evidence` to verify the bundled result and regenerate `actual-results/2026-07-13_ae-validation/generated/table5.md` without a GPU.

For a full fresh run:

```bash
make reproduce
```

For direct execution on an already allocated GPU:

```bash
make reproduce EXECUTOR=local WORKERS=1
```

To reuse expensive intermediates, set `TABLE5_CHECKPOINT_SOURCE` to a directory containing the four `exact_*_sdpa` and four `qmodel_*_sdpa` folders. If it is empty, the Makefile regenerates them from the model weights before the sixteen evaluations.

Only results from the corrected `2026-06-12_140246_corrected_protocol` lineage and the fresh 2026-07-13/14 rerun are included. Older eager-attention runs had a causal-mask alignment bug and are deliberately excluded. The paper-aligned rows use full-precision SCNA inputs (`scna_input_quant_bits=0`) on top of the W6A6/W4A4 OSTQuant backbone.
