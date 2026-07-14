.DEFAULT_GOAL := help

include mk/common.mk

FIG13 := experiments/fig-13-prefill-attention
FIG14 := experiments/fig-14-full-prefill
FIG15 := experiments/fig-15-b300-sensitivity
TBL3 := experiments/tbl-3-integer-softmax
FIG18 := experiments/fig-18-pe-area-power
FIG19 := experiments/fig-19-hardware-comparison
FIG21 := experiments/fig-21-scale-fusion
VALIDATION_OUT ?= validation/results/$(RUN_ID)
ARCHIVE ?= ../SCOPE-AE-EXP-2026-07-14.tar.gz

.PHONY: help setup all performance hardware fig-13 fig-14 fig-15 tbl-3 fig-18 fig-19 fig-21 rtl validate validate-packaged manifest check-manifest package

help:
	@echo "SCOPE AE bundle targets"
	@echo "  make setup              install the Python dependencies into .venv"
	@echo "  make all                reproduce all CPU experiments, RTL, and validate"
	@echo "  make performance        reproduce Figures 13/14/15/21 and Table 3"
	@echo "  make hardware           regenerate Figures 18 and 19"
	@echo "  make fig-13|fig-14|fig-15|fig-18|fig-19|fig-21|tbl-3"
	@echo "  make rtl                regenerate the four N=8 SystemVerilog designs"
	@echo "  make validate-packaged  validate the included 2026-07-13 results"
	@echo "  make package            verify and create the standalone tar.gz archive"
	@echo "Variables: RUN_ID=<timestamp>, JOBS=<n>, FULL_BASELINE=1, PYTHON=<path>"

setup:
	uv venv .venv
	uv pip install --python .venv/bin/python -r SCALE-Sim/requirements.txt
	uv pip install --python .venv/bin/python -e SCALE-Sim
	uv pip install --python .venv/bin/python -r LLMCompass/requirements.txt

all: performance hardware rtl
	$(MAKE) validate RUN_ID="$(RUN_ID)"

performance: fig-13 fig-14 fig-15 tbl-3 fig-21

hardware: fig-19

fig-13:
	$(MAKE) -C $(FIG13) run RUN_ID="$(RUN_ID)" JOBS="$(JOBS)" FULL_BASELINE="$(FULL_BASELINE)"

fig-14: fig-13
	$(MAKE) -C $(FIG14) run RUN_ID="$(RUN_ID)" JOBS="$(JOBS)" FIG13_RUN_DIR="$(BUNDLE_ROOT)/$(FIG13)/actual-results/$(RUN_ID)"

fig-15:
	$(MAKE) -C $(FIG15) run RUN_ID="$(RUN_ID)" JOBS="$(JOBS)"

tbl-3:
	$(MAKE) -C $(TBL3) run RUN_ID="$(RUN_ID)" JOBS="$(JOBS)"

fig-18:
	$(MAKE) -C $(FIG18) run RUN_ID="$(RUN_ID)"

fig-19: fig-18
	$(MAKE) -C $(FIG19) run RUN_ID="$(RUN_ID)" \
		FIG18_CSV="$(BUNDLE_ROOT)/$(FIG18)/actual-results/$(RUN_ID)/figure18.csv"

fig-21: fig-13
	$(MAKE) -C $(FIG21) run RUN_ID="$(RUN_ID)" FIG13_RUN_DIR="$(BUNDLE_ROOT)/$(FIG13)/actual-results/$(RUN_ID)"

rtl:
	mkdir -p hardware/rtl/logs validation/results/$(RUN_ID)
	(cd hardware/rtl && sbt "runMain pinn.common.GenerateMeshes --filter pinnacle/n8_ --force --verbose") 2>&1 | tee hardware/rtl/logs/$(RUN_ID).log
	find hardware/rtl/generated/meshes/pinnacle -type f \( -name '*.sv' -o -name '*.v' \) -print | sort > validation/results/$(RUN_ID)/generated_verilog_files.txt

validate:
	$(PYTHON) validation/validate_results.py --run-id "$(RUN_ID)" --output-dir "$(VALIDATION_OUT)"

validate-packaged:
	$(MAKE) validate RUN_ID="$(PACKAGED_RUN_ID)" VALIDATION_OUT="validation/results/$(PACKAGED_RUN_ID)"

manifest:
	(cd "$(BUNDLE_ROOT)" && find . \
		\( -path './.venv' -o -path './hardware/rtl/target' -o -path './hardware/rtl/project/target' -o -name __pycache__ \) -prune -o \
		-type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)

check-manifest:
	(cd "$(BUNDLE_ROOT)" && sha256sum --check MANIFEST.sha256)

package: validate-packaged manifest check-manifest
	tar --exclude='ae-exp/.venv' --exclude='*/__pycache__' --exclude='*/target' --exclude='*/target/**' -czf "$(ARCHIVE)" -C "$(dir $(BUNDLE_ROOT))" "$(notdir $(BUNDLE_ROOT))"
	sha256sum "$(ARCHIVE)" | sed 's#  .*/#  #' > "$(ARCHIVE).sha256"
	@echo "Archive: $(abspath $(ARCHIVE))"
