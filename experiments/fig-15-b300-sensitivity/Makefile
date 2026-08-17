.DEFAULT_GOAL := help

include ../../mk/common.mk

OUT_DIR ?= $(CURDIR)/actual-results/$(RUN_ID)
LOG_DIR ?= $(BUNDLE_ROOT)/runs/$(RUN_ID)/fig-15-b300-sensitivity/logs

.PHONY: help run fixed-tiles full-model figure

help:
	@echo "make run [RUN_ID=<timestamp>]"
	@echo "Models the doubled-SFU B300 sensitivity configuration through 512K."

run: figure

fixed-tiles:
	mkdir -p "$(OUT_DIR)" "$(LOG_DIR)"
	if [[ ! -f "$(OUT_DIR)/fixed_tiles_fp16/b300/attention_latency.csv" ]]; then $(PYTHON) extend_attention_fixed_tiles.py --devices b300 --lengths "$(LENGTHS)" --output-dir "$(OUT_DIR)/fixed_tiles_fp16" 2>&1 | tee "$(LOG_DIR)/fixed_tiles_fp16.log"; fi
	if [[ ! -f "$(OUT_DIR)/fixed_tiles_int8/b300_int8/attention_latency.csv" ]]; then LLMCOMPASS_INT8_SOFTMAX_CONVERSION=1 $(PYTHON) extend_attention_fixed_tiles.py --devices b300_int8 --lengths "$(LENGTHS)" --output-dir "$(OUT_DIR)/fixed_tiles_int8" 2>&1 | tee "$(LOG_DIR)/fixed_tiles_int8.log"; fi

full-model: fixed-tiles
	if [[ ! -f "$(OUT_DIR)/full_model_fp16/b300/full_model_speedup.csv" ]]; then $(PYTHON) derive_full_model_from_attention.py --attention-dir "$(OUT_DIR)/fixed_tiles_fp16/b300" --output-dir "$(OUT_DIR)/full_model_fp16/b300" --experiment-config configs/default.json --case-name b300_fp16 --system-label "B300 FP16" --system-config LLMCompass/configs/ours/B300.json 2>&1 | tee "$(LOG_DIR)/full_model_fp16_b300.log"; fi
	if [[ ! -f "$(OUT_DIR)/full_model_int8/b300_int8/full_model_speedup.csv" ]]; then $(PYTHON) derive_full_model_from_attention.py --attention-dir "$(OUT_DIR)/fixed_tiles_int8/b300_int8" --output-dir "$(OUT_DIR)/full_model_int8/b300_int8" --experiment-config configs/default_int8.json --case-name b300_int8 --system-label "B300 INT8" --system-config LLMCompass/configs/ours/B300.json 2>&1 | tee "$(LOG_DIR)/full_model_int8_b300.log"; fi

figure: full-model
	mkdir -p "$(OUT_DIR)/paper_figures"
	$(PYTHON) plot_paper_speedup_figures.py --figures b300 --b300-fp16-csv "$(OUT_DIR)/full_model_fp16/b300/full_model_speedup.csv" --b300-int8-csv "$(OUT_DIR)/full_model_int8/b300_int8/full_model_speedup.csv" --output-dir "$(OUT_DIR)/paper_figures" 2>&1 | tee "$(LOG_DIR)/paper_figure.log"
