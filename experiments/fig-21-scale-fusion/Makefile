.DEFAULT_GOAL := help

include ../../mk/common.mk

OUT_DIR ?= $(CURDIR)/actual-results/$(RUN_ID)
LOG_DIR ?= $(BUNDLE_ROOT)/runs/$(RUN_ID)/fig-21-scale-fusion/logs
FIG13_RUN_DIR ?= $(BUNDLE_ROOT)/experiments/fig-13-prefill-attention/expected-results

.PHONY: help run inputs

help:
	@echo "make run [RUN_ID=<timestamp>] [FIG13_RUN_DIR=<path>]"
	@echo "Uses the paired conversion/no-conversion Figure 13 conditions."

inputs:
	test -f "$(FIG13_RUN_DIR)/b200/gpu_no_conv/attention_latency.csv"
	test -f "$(FIG13_RUN_DIR)/awsv4/aws_no_conv/attention_latency.csv"
	test -f "$(FIG13_RUN_DIR)/tpuv6e/tpu_no_conv/attention_latency.csv"

run: inputs
	mkdir -p "$(OUT_DIR)" "$(LOG_DIR)"
	for device in b200 awsv4 tpuv6e; do mkdir -p "$(OUT_DIR)/$$device"; cp -a "$(FIG13_RUN_DIR)/$$device/." "$(OUT_DIR)/$$device/"; done
	$(PYTHON) generate_paper_figures.py \
		--gpu-suite-dir "$(OUT_DIR)/b200" --aws-suite-dir "$(OUT_DIR)/awsv4" \
		--tpu-suite-dir "$(OUT_DIR)/tpuv6e" --output-dir "$(OUT_DIR)/paper_figures" \
		--requested-only --devices b200,awsv4,tpuv6e 2>&1 | tee "$(LOG_DIR)/figure21.log"
