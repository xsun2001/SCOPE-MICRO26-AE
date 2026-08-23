include ../../config/common.mk
EXP_ROOT := $(CURDIR)
RESULT_DIR := $(RUN_ROOT)/tbl-4-function-approximation-accuracy
EVIDENCE_DIR := $(RUN_ROOT)/evidence/tbl-4-function-approximation-accuracy
PARAMETERS := data/scna_parameters.json
PRIMARY_EXPECTED := expected-results/paper_table4.csv
REFERENCE_EXPECTED := expected-results/scna32_reference_table4.csv

.PHONY: evidence reproduce generate

evidence:
	@mkdir -p "$(EVIDENCE_DIR)/scna16" "$(EVIDENCE_DIR)/scna32" "$(EVIDENCE_DIR)/generated"
	@$(PYTHON) scripts/reproduce_scna.py --parameters $(PARAMETERS) --output-dir "$(EVIDENCE_DIR)/scna16"
	@$(PYTHON) scripts/reproduce_scna.py --parameters $(PARAMETERS) --variant scna32 --output-dir "$(EVIDENCE_DIR)/scna32"
	@$(PYTHON) scripts/audit.py --generated "$(EVIDENCE_DIR)/scna16" --expected $(PRIMARY_EXPECTED) --parameters $(PARAMETERS) --variant scna16 --expected-nnlut-geomean 361 --expected-tlut-geomean 14.9 --output "$(EVIDENCE_DIR)/scna16/validation.json"
	@$(PYTHON) scripts/audit.py --generated "$(EVIDENCE_DIR)/scna32" --expected $(REFERENCE_EXPECTED) --parameters $(PARAMETERS) --variant scna32 --expected-nnlut-geomean 836 --expected-tlut-geomean 31.5 --output "$(EVIDENCE_DIR)/scna32/validation.json"
	@$(PYTHON) scripts/generate_table.py --expected $(PRIMARY_EXPECTED) --generated "$(EVIDENCE_DIR)/scna16/scna_metrics.csv" --reference-generated "$(EVIDENCE_DIR)/scna32/scna_metrics.csv" --output "$(EVIDENCE_DIR)/generated/table4.md"

generate:
	@mkdir -p $(RESULT_DIR)/generated
	@$(PYTHON) scripts/generate_table.py --expected $(PRIMARY_EXPECTED) --generated "$(RESULT_DIR)/scna16/scna_metrics.csv" --reference-generated "$(RESULT_DIR)/scna32/scna_metrics.csv" --output "$(RESULT_DIR)/generated/table4.md"

reproduce:
	@mkdir -p "$(RESULT_DIR)/scna16" "$(RESULT_DIR)/scna32"
	@$(PYTHON) scripts/reproduce_scna.py --parameters $(PARAMETERS) --output-dir "$(RESULT_DIR)/scna16"
	@$(PYTHON) scripts/reproduce_scna.py --parameters $(PARAMETERS) --variant scna32 --output-dir "$(RESULT_DIR)/scna32"
	@$(PYTHON) scripts/audit.py --generated "$(RESULT_DIR)/scna16" --expected $(PRIMARY_EXPECTED) --parameters $(PARAMETERS) --variant scna16 --expected-nnlut-geomean 361 --expected-tlut-geomean 14.9 --output "$(RESULT_DIR)/scna16/validation.json"
	@$(PYTHON) scripts/audit.py --generated "$(RESULT_DIR)/scna32" --expected $(REFERENCE_EXPECTED) --parameters $(PARAMETERS) --variant scna32 --expected-nnlut-geomean 836 --expected-tlut-geomean 31.5 --output "$(RESULT_DIR)/scna32/validation.json"
	@$(MAKE) generate
	@echo "Table 4 evaluated from embedded SCNA-16 parameters (primary) and SCNA-32 parameters (reference)."
