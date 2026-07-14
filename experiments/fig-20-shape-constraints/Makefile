include ../../config/common.mk
EXP_ROOT := $(CURDIR)
EXP_SLUG := fig-20-shape-constraints
RESULT_DIR := $(RUN_ROOT)/$(EXP_SLUG)
TASKS := $(shell expr $$(wc -l < data/manifest.tsv) - 1)
SLOTS := $(shell n=$(WORKERS); t=$(TASKS); [[ $$n -gt $$t ]] && n=$$t; echo $$n)
ARRAY_MAX := $(shell expr $(SLOTS) - 1)

.PHONY: evidence run collect generate reproduce
evidence:
	@$(PYTHON) scripts/collect.py --runs-dir actual/runs --analysis-dir actual/analysis --expected expected/paired_summary.csv --relative-tolerance $(FUNCTION_RELATIVE_TOLERANCE)
	@$(PYTHON) scripts/generate_figure.py --runs-dir actual/runs --output-dir generated
run:
	$(check-executor)
	@mkdir -p "$(RESULT_DIR)" "$(RUN_ROOT)/slurm"
	@if [[ "$(EXECUTOR)" == "slurm" ]]; then \
		sbatch --wait --partition="$(PARTITION)" --array="0-$(ARRAY_MAX)%$(SLOTS)" \
		--output="$(RUN_ROOT)/slurm/fig20_%A_%a.out" --error="$(RUN_ROOT)/slurm/fig20_%A_%a.err" \
		--export=ALL,BUNDLE_ROOT="$(BUNDLE_ROOT)",EXP_ROOT="$(EXP_ROOT)",RUN_ROOT="$(RUN_ROOT)",WORKER_COUNT="$(SLOTS)",PYTHON_BIN="$(PYTHON)" scripts/worker.sbatch; \
	else BUNDLE_ROOT="$(BUNDLE_ROOT)" EXP_ROOT="$(EXP_ROOT)" RUN_ROOT="$(RUN_ROOT)" WORKER_COUNT=1 PYTHON_BIN="$(PYTHON)" SLURM_ARRAY_TASK_ID=0 bash scripts/worker.sbatch; fi
collect:
	@$(PYTHON) scripts/collect.py --runs-dir "$(RESULT_DIR)/runs" --analysis-dir "$(RESULT_DIR)/analysis" --expected expected/paired_summary.csv --relative-tolerance $(FUNCTION_RELATIVE_TOLERANCE)
generate:
	@$(PYTHON) scripts/generate_figure.py --runs-dir "$(RESULT_DIR)/runs" --output-dir "$(RESULT_DIR)/generated"
reproduce: run collect generate
