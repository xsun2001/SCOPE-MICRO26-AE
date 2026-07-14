.DEFAULT_GOAL := help

include mk/common.mk
include config/common.mk

FIG13 := experiments/fig-13-prefill-attention
FIG14 := experiments/fig-14-full-prefill
FIG15 := experiments/fig-15-b300-sensitivity
FIG16 := experiments/fig-16-end-to-end-quality
FIG17 := experiments/fig-17-neuron-scalability
FIG18 := experiments/fig-18-pe-area-power
FIG19 := experiments/fig-19-hardware-comparison
FIG20 := experiments/fig-20-shape-constraints
FIG21 := experiments/fig-21-scale-fusion
TBL3 := experiments/tbl-3-integer-softmax
TBL4 := experiments/tbl-4-function-approximation-accuracy
TBL5 := experiments/tbl-5-ostquant-quality
VALIDATION_OUT ?= validation/results/$(RUN_ID)

.PHONY: help setup all evidence evidence-cpu evidence-gpu validate validate-cpu validate-cpu-run validate-gpu validate-packaged reproduce reproduce-cpu reproduce-gpu performance hardware fig-13 fig-14 fig-15 fig-16 fig-17 fig-18 fig-19 fig-20 fig-21 tbl-3 tbl-4 tbl-5 rtl archive package

help:
	@echo "SCOPE unified artifact-evaluation bundle"
	@echo ""
	@echo "Fast, hardware-free review:"
	@echo "  make setup              install CPU and GPU experiment dependencies"
	@echo "  make evidence           regenerate bundled figures/tables and audit CPU evidence"
	@echo "  make validate           validate all packaged CPU and GPU results"
	@echo "  make all                evidence + validation (no experiment rerun)"
	@echo ""
	@echo "Fresh reproduction:"
	@echo "  make reproduce-cpu      Figures 13/14/15/18/19/21, Table 3, and RTL"
	@echo "  make reproduce-gpu      Figures 16/17/20 and Table 5 (Table 4 is evidence-only)"
	@echo "  make reproduce          run both suites"
	@echo "  make hardware           extract reports, fit, and draw Figures 18/19"
	@echo ""
	@echo "Individual targets: fig-13 ... fig-21, tbl-3, tbl-4, tbl-5, rtl"
	@echo "GPU variables: MODEL_ROOT, EXECUTOR=slurm|local, WORKERS, TABLE5_CHECKPOINT_SOURCE"
	@echo "CPU variables: RUN_ID, JOBS, FULL_BASELINE=1, PYTHON"

setup:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv .venv; \
		uv pip install --python .venv/bin/python -r requirements/accuracy.txt -r SCALE-Sim/requirements.txt; \
		uv pip install --python .venv/bin/python -e SCALE-Sim; \
		uv pip install --python .venv/bin/python -r LLMCompass/requirements.txt; \
	else \
		python3 -m venv .venv; \
		.venv/bin/pip install -r requirements/accuracy.txt -r SCALE-Sim/requirements.txt; \
		.venv/bin/pip install -e SCALE-Sim; \
		.venv/bin/pip install -r LLMCompass/requirements.txt; \
	fi

all: evidence validate

evidence: evidence-cpu evidence-gpu

evidence-cpu: validate-cpu

evidence-gpu:
	$(MAKE) -C $(TBL4) evidence
	$(MAKE) -C $(FIG16) evidence
	$(MAKE) -C $(TBL5) evidence
	$(MAKE) -C $(FIG17) evidence
	$(MAKE) -C $(FIG20) evidence

validate: validate-cpu validate-gpu

validate-cpu:
	$(PYTHON) validation/validate_results.py --run-id "$(PACKAGED_RUN_ID)" --output-dir "validation/results/$(PACKAGED_RUN_ID)"

validate-cpu-run:
	$(PYTHON) validation/validate_results.py --run-id "$(RUN_ID)" --output-dir "$(VALIDATION_OUT)"

validate-gpu:
	$(PYTHON) tools/validate_bundle.py --bundle-root "$(BUNDLE_ROOT)"

validate-packaged: validate

reproduce: reproduce-cpu reproduce-gpu

reproduce-cpu: performance hardware rtl
	$(MAKE) validate-cpu-run RUN_ID="$(RUN_ID)"

reproduce-gpu: tbl-4 fig-16 tbl-5 fig-17 fig-20

performance: fig-13 fig-14 fig-15 tbl-3 fig-21

hardware: fig-19

fig-13:
	$(MAKE) -C $(FIG13) run RUN_ID="$(RUN_ID)" JOBS="$(JOBS)" FULL_BASELINE="$(FULL_BASELINE)"

fig-14: fig-13
	$(MAKE) -C $(FIG14) run RUN_ID="$(RUN_ID)" JOBS="$(JOBS)" FIG13_RUN_DIR="$(BUNDLE_ROOT)/$(FIG13)/actual-results/$(RUN_ID)"

fig-15:
	$(MAKE) -C $(FIG15) run RUN_ID="$(RUN_ID)" JOBS="$(JOBS)"

fig-16:
	$(MAKE) -C $(FIG16) reproduce EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)" WORKERS="$(WORKERS)"

fig-17:
	$(MAKE) -C $(FIG17) reproduce EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)" WORKERS="$(WORKERS)"

fig-18:
	$(MAKE) -C $(FIG18) run RUN_ID="$(RUN_ID)"

fig-19: fig-18
	$(MAKE) -C $(FIG19) run RUN_ID="$(RUN_ID)" \
		FIG18_CSV="$(BUNDLE_ROOT)/$(FIG18)/actual-results/$(RUN_ID)/figure18.csv"

fig-20:
	$(MAKE) -C $(FIG20) reproduce EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)" WORKERS="$(WORKERS)"

fig-21: fig-13
	$(MAKE) -C $(FIG21) run RUN_ID="$(RUN_ID)" FIG13_RUN_DIR="$(BUNDLE_ROOT)/$(FIG13)/actual-results/$(RUN_ID)"

tbl-3:
	$(MAKE) -C $(TBL3) run RUN_ID="$(RUN_ID)" JOBS="$(JOBS)"

tbl-4:
	$(MAKE) -C $(TBL4) reproduce RUN_ROOT="$(RUN_ROOT)"

tbl-5:
	$(MAKE) -C $(TBL5) reproduce EXECUTOR="$(EXECUTOR)" RUN_ROOT="$(RUN_ROOT)" WORKERS="$(WORKERS)" TABLE5_CHECKPOINT_SOURCE="$(TABLE5_CHECKPOINT_SOURCE)"

rtl:
	mkdir -p hardware/rtl/logs validation/results/$(RUN_ID)
	(cd hardware/rtl && sbt "runMain pinn.common.GenerateMeshes --filter pinnacle/n8_ --force --verbose") 2>&1 | tee hardware/rtl/logs/$(RUN_ID).log
	find hardware/rtl/generated/meshes/pinnacle -type f \( -name '*.sv' -o -name '*.v' \) -print | sort > validation/results/$(RUN_ID)/generated_verilog_files.txt

archive: evidence validate
	$(PYTHON) tools/create_archive.py --bundle-root "$(BUNDLE_ROOT)" --output "$(ARCHIVE)"

package: archive
