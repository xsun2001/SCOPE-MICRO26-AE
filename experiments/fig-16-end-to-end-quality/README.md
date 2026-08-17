# Figure 16: end-to-end model quality

This experiment reproduces the paper's WikiText-2 perplexity and average zero-shot accuracy (ARC-Easy, HellaSwag, PIQA, and WinoGrande) for OPT-6.7B, Llama-2-7B, Llama-3-8B, Qwen2.5-7B, and Qwen3-8B. It is an accuracy/precision experiment suitable for this GPU host; it does not measure end-to-end latency or hardware performance.

- `../../end2endacc/`: shared inference/evaluation harness and SCNA weights.
- `data/configs/` and `data/manifest.tsv`: the exact 80-run matrix.
- `expected-results/`: Figure 16 paper values and provenance maps.
- `actual-results/2026-07-13_ae-validation/`: compact results and generated figures from the fresh 2026-07-13/14 validation.
- `scripts/`: runner, collector, validator, and paper-figure generator.

Use `make evidence` to validate 80/80 bundled plotted values and regenerate the figure under `runs/<run-id>/evidence/fig-16-end-to-end-quality/generated/`. The ignored staging directory keeps the packaged evidence and worktree unchanged. Use `make reproduce` for Slurm, or `make reproduce EXECUTOR=local WORKERS=1` on an already allocated GPU.

A fresh run needs an NVIDIA CUDA GPU from the Ampere generation or newer with at least 80 GB of device memory. H100 80 GB (Hopper) is the validated reference, not a device lock; an A100 80 GB (Ampere), H100/H200 (Hopper), or B100/B200 (Blackwell) is suitable with a compatible PyTorch/CUDA installation. The validated 80-run execution is Slurm array job `410219`: Slurm accounting measures 18.80 H100 GPU-hours and 2:26:32 wall time with 16 one-GPU workers, excluding downloads and queue delay.

Set `MODEL_ROOT` to a directory with all five model folders; dataset downloads use Hugging Face caches. The configurations execute `torch.bfloat16`, so BF16 is the authoritative full-precision label. Historical `fp16_*` filenames and record keys are retained only to address packaged results. Validation permits 0.05 absolute PPL and 0.007 absolute accuracy drift and always reconstructs the four-task mean from raw task records. Run `make smoke` for the source-only import/config/wrapper check.
