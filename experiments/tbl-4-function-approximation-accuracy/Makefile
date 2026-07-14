include ../../config/common.mk
EXP_ROOT := $(CURDIR)
RESULT_DIR := $(RUN_ROOT)/tbl-4-function-approximation-accuracy

.PHONY: evidence reproduce generate

evidence generate:
	@mkdir -p generated
	@$(PYTHON) scripts/generate_table.py --expected expected/paper_table4.csv --output generated/table4.md

reproduce: evidence
	@mkdir -p "$(RESULT_DIR)"
	@$(PYTHON) scripts/audit.py --experiment-root "$(EXP_ROOT)" --output "$(RESULT_DIR)/validation.json"
	@echo "Table 4 is evidence-only: the shared baseline harness and raw result grid are absent."
