.DEFAULT_GOAL := help

include ../../mk/common.mk

OUT_DIR ?= $(CURDIR)/actual-results/$(RUN_ID)
LOG_DIR ?= $(CURDIR)/logs/$(RUN_ID)

.PHONY: help run

help:
	@echo "make run [RUN_ID=<timestamp>]"
	@echo "Produces H100 INT8 latency CSVs for SCOPE, I-LLM, and IntAttention."

run:
	mkdir -p "$(OUT_DIR)" "$(LOG_DIR)"
	$(PYTHON) extend_attention_fixed_tiles.py --devices h100_int8 \
		--lengths 2048,4096,8192,16384 --int-softmax-comparison --no-source-rows \
		--output-dir "$(OUT_DIR)" 2>&1 | tee "$(LOG_DIR)/table3_h100.log"
