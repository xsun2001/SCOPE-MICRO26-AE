include config/common.mk

.PHONY: help setup evidence validate reproduce fig-16 fig-17 fig-20 tbl-4 tbl-5 archive clean-generated

help:
	@$(PYTHON) tools/show_help.py

setup:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv .venv; uv pip install --python .venv/bin/python -r requirements/accuracy.txt; \
	else \
		python3 -m venv .venv; .venv/bin/pip install -r requirements/accuracy.txt; \
	fi

evidence:
	@$(MAKE) -C experiments/tbl-4-function-approximation-accuracy evidence
	@$(MAKE) -C experiments/fig-16-end-to-end-quality evidence
	@$(MAKE) -C experiments/tbl-5-ostquant-quality evidence
	@$(MAKE) -C experiments/fig-17-neuron-scalability evidence
	@$(MAKE) -C experiments/fig-20-shape-constraints evidence

validate:
	@$(PYTHON) tools/validate_bundle.py --bundle-root "$(BUNDLE_ROOT)"

reproduce:
	@$(MAKE) tbl-4 EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)"
	@$(MAKE) fig-16 EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)"
	@$(MAKE) tbl-5 EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)"
	@$(MAKE) fig-17 EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)"
	@$(MAKE) fig-20 EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)"

tbl-4:
	@$(MAKE) -C experiments/tbl-4-function-approximation-accuracy reproduce RUN_ROOT="$(RUN_ROOT)"

fig-16:
	@$(MAKE) -C experiments/fig-16-end-to-end-quality reproduce EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)" WORKERS="$(WORKERS)"

tbl-5:
	@$(MAKE) -C experiments/tbl-5-ostquant-quality reproduce EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)" WORKERS="$(WORKERS)" TABLE5_CHECKPOINT_SOURCE="$(TABLE5_CHECKPOINT_SOURCE)"

fig-17:
	@$(MAKE) -C experiments/fig-17-neuron-scalability reproduce EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)" WORKERS="$(WORKERS)"

fig-20:
	@$(MAKE) -C experiments/fig-20-shape-constraints reproduce EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)" WORKERS="$(WORKERS)"

archive: evidence validate
	@$(PYTHON) tools/create_archive.py --bundle-root "$(BUNDLE_ROOT)" --output "$(ARCHIVE)"

clean-generated:
	@find experiments -type f -path '*/generated/*' -delete
