.DEFAULT_GOAL := help

include ../../mk/common.mk

OUT_DIR ?= $(CURDIR)/actual-results/$(RUN_ID)
LOG_DIR ?= $(BUNDLE_ROOT)/runs/$(RUN_ID)/tbl-3-integer-softmax/logs
REFERENCE_MICROBENCHMARK ?= $(CURDIR)/data/h100-softmax-microbenchmark

.PHONY: help run microbenchmark validate-microbenchmark

help:
	@echo "make run [RUN_ID=<timestamp>]"
	@echo "Produces H100 INT8 latency CSVs for SCOPE, I-LLM, and IntAttention."
	@echo "make microbenchmark       run the physical CUDA/Triton softmax benchmark on an allocated H100"
	@echo "make validate-microbenchmark  recompute all statistics from packaged raw repetitions"

run:
	mkdir -p "$(OUT_DIR)" "$(LOG_DIR)"
	$(PYTHON) extend_attention_fixed_tiles.py --devices h100_int8 \
		--lengths 2048,4096,8192,16384 --int-softmax-comparison --no-source-rows \
		--output-dir "$(OUT_DIR)" 2>&1 | tee "$(LOG_DIR)/table3_h100.log"

microbenchmark:
	mkdir -p "$(OUT_DIR)/softmax-microbenchmark" "$(LOG_DIR)"
	$(PYTHON) bench_softmax_h100.py --check --check-max-elems 67108864 \
		--output-dir "$(OUT_DIR)/softmax-microbenchmark" \
		2>&1 | tee "$(LOG_DIR)/table3_softmax_h100.log"

validate-microbenchmark:
	$(PYTHON) scripts/validate_microbenchmark.py --result-dir "$(REFERENCE_MICROBENCHMARK)"
