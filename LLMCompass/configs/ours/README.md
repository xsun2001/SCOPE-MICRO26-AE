Custom LLMCompass configs for this project live here.

The top-level `LLMCompass/configs/` files are reserved for the original upstream
LLMCompass configs and templates. New project-specific GPU configs should be
added under `LLMCompass/configs/ours/` and referenced explicitly by experiments.

This directory can also contain non-GPU configs, such as TPU variants, when the
JSON schema needs project-local extensions that are not part of upstream
LLMCompass templates.

See [REPORT.md](./REPORT.md) for a generated matrix of modeled compute and
memory characteristics across all configs in this directory.

TPU configs currently include:
* `TPUv3.json`, `TPUv4.json`, `TPUv5e.json`, `TPUv5p.json`, `TPUv6e.json`: per-chip models that serve both the BF16-path and INT8-path experiments

The TPU per-chip configs carry `int8_multiplier` plus INT8 tensor I/O dtype
overrides, so the same JSON can be reused for INT8 runs without duplicating the
full file.
