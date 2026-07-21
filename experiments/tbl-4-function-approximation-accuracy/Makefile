include ../../config/common.mk
EXP_ROOT := $(CURDIR)
RESULT_DIR := $(RUN_ROOT)/tbl-4-function-approximation-accuracy
PACKAGED_DIR := actual-results/$(PACKAGED_GPU_RUN_ID)
EVIDENCE_DIR := $(RUN_ROOT)/evidence/tbl-4-function-approximation-accuracy

.PHONY: evidence reproduce generate

evidence:
	@mkdir -p $(EVIDENCE_DIR)/generated
	@$(PYTHON) scripts/generate_table.py --expected expected-results/paper_table4.csv --output $(EVIDENCE_DIR)/generated/table4.md

generate:
	@mkdir -p $(RESULT_DIR)/generated
	@$(PYTHON) scripts/generate_table.py --expected expected-results/paper_table4.csv --output $(RESULT_DIR)/generated/table4.md

reproduce: generate
	@mkdir -p "$(RESULT_DIR)"
	@$(PYTHON) scripts/audit.py --experiment-root "$(EXP_ROOT)" --output "$(RESULT_DIR)/validation.json"
	@echo "Table 4 SCNA accuracy reproduced; consult the baseline papers for other methods."
