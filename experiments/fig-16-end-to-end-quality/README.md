# Figure 16: end-to-end model quality

This experiment reproduces the paper's WikiText-2 perplexity and average zero-shot accuracy (ARC-Easy, HellaSwag, PIQA, and WinoGrande) for OPT-6.7B, Llama-2-7B, Llama-3-8B, Qwen2.5-7B, and Qwen3-8B. It is an accuracy/precision experiment suitable for this GPU host; it does not measure end-to-end latency or hardware performance.

- `../../end2endacc/`: shared inference/evaluation harness and SCNA weights.
- `data/configs/` and `data/manifest.tsv`: the exact 80-run matrix.
- `expected-results/`: Figure 16 paper values and provenance maps.
- `actual-results/2026-07-13_ae-validation/`: compact results and generated figures from the fresh 2026-07-13/14 validation.
- `scripts/`: runner, collector, validator, and paper-figure generator.

Use `make evidence` to validate 80/80 bundled plotted values and regenerate `actual-results/2026-07-13_ae-validation/generated/figure16.{png,pdf}`. Use `make reproduce` for Slurm, or `make reproduce EXECUTOR=local WORKERS=1` on an already allocated GPU.

A fresh run needs an NVIDIA CUDA GPU from the Ampere generation or newer with at least 80 GB of device memory. H100 80 GB (Hopper) is the validated reference, not a device lock; an A100 80 GB (Ampere), H100/H200 (Hopper), or B100/B200 (Blackwell) is suitable with a compatible PyTorch/CUDA installation. Budget about 16--24 GPU-hours on one H100, or about 2--4 hours of wall time with the documented 15-worker Slurm execution (one H100 per worker), excluding downloads and queue delay. This range is a conservative planning estimate from the 80-run matrix.

Set `MODEL_ROOT` to a directory with all five model folders. Dataset downloads use Hugging Face caches. The runtime uses BF16 even though the paper groups this condition under the full-precision/FP16 label. The static condition quantizes the backbone and SCNA W/A path; this checkout does not implement the paper's stated INT8 KV-cache quantization. These limitations are retained explicitly so the bundle matches the experiments that produced the paper values rather than a newer or hypothetical protocol.
