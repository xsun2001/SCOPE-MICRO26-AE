.DEFAULT_GOAL := help

include ../../mk/common.mk

OUT_DIR ?= $(CURDIR)/actual-results/$(RUN_ID)
FIG18_CSV ?= $(CURDIR)/../fig-18-pe-area-power/expected-results/figure18.csv

.PHONY: help run calculate plot

help:
	@echo "make run [RUN_ID=<timestamp>] [FIG18_CSV=<path>]"
	@echo "Calculate 32x32 incremental overhead, verify the CSV, and draw Figure 19."

run: calculate

calculate:
	$(PYTHON) reproduce.py --figure18 "$(FIG18_CSV)" --output-dir "$(OUT_DIR)"

plot:
	$(PYTHON) plot.py --input "$(OUT_DIR)/figure19.csv" --output-dir "$(OUT_DIR)"
