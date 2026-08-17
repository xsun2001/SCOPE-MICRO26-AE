include ../../config/common.mk
EXP_ROOT := $(CURDIR)
EXP_SLUG := fig-16-end-to-end-quality
RESULT_DIR := $(RUN_ROOT)/$(EXP_SLUG)
PACKAGED_DIR := actual-results/$(PACKAGED_GPU_RUN_ID)
EVIDENCE_DIR := $(RUN_ROOT)/evidence/$(EXP_SLUG)
TASKS := $(shell expr $$(wc -l < data/manifest.tsv) - 1)
SLOTS := $(shell n=$(WORKERS); t=$(TASKS); [[ $$n -gt $$t ]] && n=$$t; echo $$n)
ARRAY_MAX := $(shell expr $(SLOTS) - 1)

.PHONY: evidence smoke run collect generate reproduce
evidence:
	@$(PYTHON) scripts/validate_actual.py --actual $(PACKAGED_DIR)/analysis
	@$(PYTHON) scripts/comparison_to_plot.py --comparison $(PACKAGED_DIR)/analysis/comparison.csv --template expected-results/paper_figure16.csv --output $(EVIDENCE_DIR)/generated/actual_plot.csv
	@FIG16_INPUT="$(EVIDENCE_DIR)/generated/actual_plot.csv" FIG16_OUTPUT_DIR="$(EVIDENCE_DIR)/generated" $(PYTHON) scripts/generate_figure.py --merged-triptych-only --variants exact,pinn8,pinn16,pinn32 --stem figure16
smoke:
	@$(PYTHON) scripts/smoke_source.py
run:
	$(check-executor)
	@mkdir -p "$(RESULT_DIR)" "$(RUN_ROOT)/slurm"
	@if [[ "$(EXECUTOR)" == "slurm" ]]; then \
		sbatch --wait --partition="$(PARTITION)" --array="0-$(ARRAY_MAX)%$(SLOTS)" \
		--output="$(RUN_ROOT)/slurm/fig16_%A_%a.out" --error="$(RUN_ROOT)/slurm/fig16_%A_%a.err" \
		--export=ALL,BUNDLE_ROOT="$(BUNDLE_ROOT)",EXP_ROOT="$(EXP_ROOT)",RUN_ROOT="$(RUN_ROOT)",MODEL_ROOT="$(MODEL_ROOT)",WORKER_COUNT="$(SLOTS)",PYTHON_BIN="$(PYTHON)" scripts/worker.sbatch; \
	else BUNDLE_ROOT="$(BUNDLE_ROOT)" EXP_ROOT="$(EXP_ROOT)" RUN_ROOT="$(RUN_ROOT)" MODEL_ROOT="$(MODEL_ROOT)" WORKER_COUNT=1 PYTHON_BIN="$(PYTHON)" SLURM_ARRAY_TASK_ID=0 bash scripts/worker.sbatch; fi
collect:
	@$(PYTHON) scripts/collect.py --result-dir "$(RESULT_DIR)" --experiment-root "$(EXP_ROOT)" --ppl-tolerance $(PPL_TOLERANCE) --accuracy-tolerance $(ACCURACY_TOLERANCE)
generate:
	@$(PYTHON) scripts/comparison_to_plot.py --comparison "$(RESULT_DIR)/analysis/comparison.csv" --template expected-results/paper_figure16.csv --output "$(RESULT_DIR)/generated/actual_plot.csv"
	@FIG16_INPUT="$(RESULT_DIR)/generated/actual_plot.csv" FIG16_OUTPUT_DIR="$(RESULT_DIR)/generated" $(PYTHON) scripts/generate_figure.py --merged-triptych-only --variants exact,pinn8,pinn16,pinn32 --stem figure16
reproduce: run collect generate
