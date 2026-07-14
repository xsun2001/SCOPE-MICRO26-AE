include ../../config/common.mk
EXP_ROOT := $(CURDIR)
EXP_SLUG := tbl-5-ostquant-quality
RESULT_DIR := $(RUN_ROOT)/$(EXP_SLUG)
PACKAGED_DIR := actual-results/$(PACKAGED_GPU_RUN_ID)

.PHONY: evidence dispatch checkpoints run collect generate reproduce

evidence:
	@$(PYTHON) scripts/validate_actual.py --actual $(PACKAGED_DIR)/analysis
	@$(PYTHON) scripts/generate_table.py --comparison $(PACKAGED_DIR)/analysis/paper_comparison.csv --output $(PACKAGED_DIR)/generated/table5.md

dispatch:
	$(check-executor)
	@mkdir -p "$(RESULT_DIR)" "$(RUN_ROOT)/slurm"
	@tasks="$(TASKS)"; slots="$(WORKERS)"; \
	if [[ $$slots -gt $$tasks ]]; then slots=$$tasks; fi; array_max=$$((slots - 1)); \
	if [[ "$(EXECUTOR)" == "slurm" ]]; then \
		sbatch --wait --partition="$(PARTITION)" --array="0-$$array_max%$$slots" \
		--output="$(RUN_ROOT)/slurm/tbl5_$(STAGE)_%A_%a.out" --error="$(RUN_ROOT)/slurm/tbl5_$(STAGE)_%A_%a.err" \
		--export=ALL,BUNDLE_ROOT="$(BUNDLE_ROOT)",EXP_ROOT="$(EXP_ROOT)",RUN_ROOT="$(RUN_ROOT)",MODEL_ROOT="$(MODEL_ROOT)",STAGE="$(STAGE)",WORKER_COUNT="$$slots",PYTHON_BIN="$(PYTHON)" scripts/stage_worker.sbatch; \
	else BUNDLE_ROOT="$(BUNDLE_ROOT)" EXP_ROOT="$(EXP_ROOT)" RUN_ROOT="$(RUN_ROOT)" MODEL_ROOT="$(MODEL_ROOT)" STAGE="$(STAGE)" WORKER_COUNT=1 PYTHON_BIN="$(PYTHON)" SLURM_ARRAY_TASK_ID=0 bash scripts/stage_worker.sbatch; fi

checkpoints:
	@if [[ -n "$(TABLE5_CHECKPOINT_SOURCE)" ]]; then \
		$(PYTHON) scripts/prepare_checkpoints.py --source "$(TABLE5_CHECKPOINT_SOURCE)" --result-dir "$(RESULT_DIR)"; \
	else \
		$(MAKE) dispatch STAGE=exact TASKS=4 EXECUTOR="$(EXECUTOR)" WORKERS="$(WORKERS)" RUN_ROOT="$(RUN_ROOT)"; \
		$(MAKE) dispatch STAGE=qmodel TASKS=4 EXECUTOR="$(EXECUTOR)" WORKERS="$(WORKERS)" RUN_ROOT="$(RUN_ROOT)"; \
	fi

run: checkpoints
	@$(MAKE) dispatch STAGE=eval TASKS=16 EXECUTOR="$(EXECUTOR)" WORKERS="$(WORKERS)" RUN_ROOT="$(RUN_ROOT)"

collect:
	@$(PYTHON) scripts/collect_results.py --run-root "$(RESULT_DIR)"
	@$(PYTHON) scripts/validate.py --result-dir "$(RESULT_DIR)" --expected expected-results/paper_table5.csv

generate:
	@$(PYTHON) scripts/generate_table.py --comparison "$(RESULT_DIR)/analysis/paper_comparison.csv" --output "$(RESULT_DIR)/generated/table5.md"

reproduce: run collect generate
