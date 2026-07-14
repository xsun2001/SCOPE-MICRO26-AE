.DEFAULT_GOAL := help

include ../../mk/common.mk

OUT_DIR ?= $(CURDIR)/actual-results/$(RUN_ID)
REPORTS ?= $(CURDIR)/../../hardware/synthesis/reports

.PHONY: help run extract-fit plot

help:
	@echo "make run [RUN_ID=<timestamp>]"
	@echo "Extract 112 native report sets, fit per-PE values, verify the CSVs, and draw Figure 18."

run: extract-fit

extract-fit:
	$(PYTHON) reproduce.py --reports "$(REPORTS)" --output-dir "$(OUT_DIR)"

plot:
	$(PYTHON) plot.py --input "$(OUT_DIR)/figure18.csv" --output-dir "$(OUT_DIR)"
