include ../../config/common.mk
EXP_ROOT := $(CURDIR)
RESULT_DIR := $(RUN_ROOT)/tbl-4-function-approximation-accuracy
PACKAGED_DIR := actual-results/$(PACKAGED_GPU_RUN_ID)
EVIDENCE_DIR := $(RUN_ROOT)/evidence/tbl-4-function-approximation-accuracy

.PHONY: evidence reproduce generate

evidence:
	@mkdir -p $(EVIDENCE_DIR)/generated
	@$(PYTHON) scripts/reproduce_scna.py --checkpoints data/scna_checkpoints.json --output-dir "$(EVIDENCE_DIR)"
	@$(PYTHON) scripts/audit.py --generated "$(EVIDENCE_DIR)" --expected expected-results/paper_table4.csv --checkpoints data/scna_checkpoints.json --output "$(EVIDENCE_DIR)/validation.json"
	@$(PYTHON) scripts/generate_table.py --expected expected-results/paper_table4.csv --generated "$(EVIDENCE_DIR)/scna_metrics.csv" --output $(EVIDENCE_DIR)/generated/table4.md

generate:
	@mkdir -p $(RESULT_DIR)/generated
	@$(PYTHON) scripts/generate_table.py --expected expected-results/paper_table4.csv --generated "$(RESULT_DIR)/scna_metrics.csv" --output $(RESULT_DIR)/generated/table4.md

reproduce:
	@mkdir -p "$(RESULT_DIR)"
	@$(PYTHON) scripts/reproduce_scna.py --checkpoints data/scna_checkpoints.json --output-dir "$(RESULT_DIR)"
	@$(PYTHON) scripts/audit.py --generated "$(RESULT_DIR)" --expected expected-results/paper_table4.csv --checkpoints data/scna_checkpoints.json --output "$(RESULT_DIR)/validation.json"
	@$(MAKE) generate
	@echo "Table 4 SCNA accuracy reproduced from 11 independent raw prediction streams."
