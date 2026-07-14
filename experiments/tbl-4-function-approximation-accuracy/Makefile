include ../../config/common.mk
EXP_ROOT := $(CURDIR)
RESULT_DIR := $(RUN_ROOT)/tbl-4-function-approximation-accuracy
PACKAGED_DIR := actual-results/$(PACKAGED_GPU_RUN_ID)

.PHONY: evidence reproduce generate

evidence generate:
	@mkdir -p $(PACKAGED_DIR)/generated
	@$(PYTHON) scripts/generate_table.py --expected expected-results/paper_table4.csv --output $(PACKAGED_DIR)/generated/table4.md

reproduce: evidence
	@mkdir -p "$(RESULT_DIR)"
	@$(PYTHON) scripts/audit.py --experiment-root "$(EXP_ROOT)" --output "$(RESULT_DIR)/validation.json"
	@echo "Table 4 is evidence-only: the shared baseline harness and raw result grid are absent."
