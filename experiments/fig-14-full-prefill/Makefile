.DEFAULT_GOAL := help

include ../../mk/common.mk

OUT_DIR ?= $(CURDIR)/actual-results/$(RUN_ID)
LOG_DIR ?= $(BUNDLE_ROOT)/runs/$(RUN_ID)/fig-14-full-prefill/logs
FIG13_RUN_DIR ?= $(BUNDLE_ROOT)/experiments/fig-13-prefill-attention/expected-results

.PHONY: help run inputs fixed-tiles full-model figures

help:
	@echo "make run [RUN_ID=<timestamp>] [FIG13_RUN_DIR=<path>]"
	@echo "Uses fresh Figure 13 rows through 32K and models 64K--512K."

run: figures

inputs:
	test -f "$(FIG13_RUN_DIR)/b200/gpu_conv/attention_latency.csv"
	test -f "$(FIG13_RUN_DIR)/awsv4/aws_conv/attention_latency.csv"
	test -f "$(FIG13_RUN_DIR)/tpuv6e/tpu_conv/attention_latency.csv"

fixed-tiles: inputs
	mkdir -p "$(OUT_DIR)" "$(LOG_DIR)"
	if [[ ! -f "$(OUT_DIR)/fixed_tiles_fp16/metadata.json" ]]; then \
		$(PYTHON) extend_attention_fixed_tiles.py --devices b200,awsv4,tpuv6e --lengths "$(LENGTHS)" \
			--source-dir "b200=$(FIG13_RUN_DIR)/b200/gpu_conv" \
			--source-dir "awsv4=$(FIG13_RUN_DIR)/awsv4/aws_conv" \
			--source-dir "tpuv6e=$(FIG13_RUN_DIR)/tpuv6e/tpu_conv" \
			--output-dir "$(OUT_DIR)/fixed_tiles_fp16" 2>&1 | tee "$(LOG_DIR)/fixed_tiles_fp16.log"; \
	fi
	if [[ ! -f "$(OUT_DIR)/fixed_tiles_int8/metadata.json" ]]; then \
		LLMCOMPASS_INT8_SOFTMAX_CONVERSION=1 $(PYTHON) extend_attention_fixed_tiles.py \
			--devices b200_int8,awsv4_int8,tpuv6e_int8 --lengths "$(LENGTHS)" \
			--source-dir "b200_int8=$(FIG13_RUN_DIR)/b200/gpu_conv" \
			--source-dir "awsv4_int8=$(FIG13_RUN_DIR)/awsv4/aws_conv" \
			--source-dir "tpuv6e_int8=$(FIG13_RUN_DIR)/tpuv6e/tpu_conv" \
			--output-dir "$(OUT_DIR)/fixed_tiles_int8" 2>&1 | tee "$(LOG_DIR)/fixed_tiles_int8.log"; \
	fi

full-model: fixed-tiles
	if [[ ! -f "$(OUT_DIR)/full_model_fp16/b200/full_model_speedup.csv" ]]; then $(PYTHON) derive_full_model_from_attention.py --attention-dir "$(OUT_DIR)/fixed_tiles_fp16/b200" --output-dir "$(OUT_DIR)/full_model_fp16/b200" --experiment-config configs/default.json --case-name b200_fp16 --system-label "B200 FP16" --system-config LLMCompass/configs/ours/B200.json 2>&1 | tee "$(LOG_DIR)/full_model_fp16_b200.log"; fi
	if [[ ! -f "$(OUT_DIR)/full_model_fp16/awsv4/full_model_speedup.csv" ]]; then $(PYTHON) derive_full_model_from_attention.py --attention-dir "$(OUT_DIR)/fixed_tiles_fp16/awsv4" --output-dir "$(OUT_DIR)/full_model_fp16/awsv4" --experiment-config configs/default.json --case-name awsv4_fp16 --system-label "AWSv4 FP16" --system-config LLMCompass/configs/ours/AWSv4.json --adjust-global-buffer-to-memory-capacity 2>&1 | tee "$(LOG_DIR)/full_model_fp16_awsv4.log"; fi
	if [[ ! -f "$(OUT_DIR)/full_model_fp16/tpuv6e/full_model_speedup.csv" ]]; then $(PYTHON) derive_full_model_from_attention.py --attention-dir "$(OUT_DIR)/fixed_tiles_fp16/tpuv6e" --output-dir "$(OUT_DIR)/full_model_fp16/tpuv6e" --experiment-config configs/default.json --case-name tpuv6e_fp16 --system-label "TPUv6e FP16" --system-config LLMCompass/configs/ours/TPUv6e.json --adjust-global-buffer-to-memory-capacity 2>&1 | tee "$(LOG_DIR)/full_model_fp16_tpuv6e.log"; fi
	if [[ ! -f "$(OUT_DIR)/full_model_int8/b200_int8/full_model_speedup.csv" ]]; then $(PYTHON) derive_full_model_from_attention.py --attention-dir "$(OUT_DIR)/fixed_tiles_int8/b200_int8" --output-dir "$(OUT_DIR)/full_model_int8/b200_int8" --experiment-config configs/default_int8.json --case-name b200_int8 --system-label "B200 INT8" --system-config LLMCompass/configs/ours/B200.json 2>&1 | tee "$(LOG_DIR)/full_model_int8_b200.log"; fi
	if [[ ! -f "$(OUT_DIR)/full_model_int8/awsv4_int8/full_model_speedup.csv" ]]; then $(PYTHON) derive_full_model_from_attention.py --attention-dir "$(OUT_DIR)/fixed_tiles_int8/awsv4_int8" --output-dir "$(OUT_DIR)/full_model_int8/awsv4_int8" --experiment-config configs/default_int8.json --case-name awsv4_int8 --system-label "AWSv4 INT8" --system-config LLMCompass/configs/ours/AWSv4.json --adjust-global-buffer-to-memory-capacity 2>&1 | tee "$(LOG_DIR)/full_model_int8_awsv4.log"; fi
	if [[ ! -f "$(OUT_DIR)/full_model_int8/tpuv6e_int8/full_model_speedup.csv" ]]; then $(PYTHON) derive_full_model_from_attention.py --attention-dir "$(OUT_DIR)/fixed_tiles_int8/tpuv6e_int8" --output-dir "$(OUT_DIR)/full_model_int8/tpuv6e_int8" --experiment-config configs/default_int8.json --case-name tpuv6e_int8 --system-label "TPUv6e INT8" --system-config LLMCompass/configs/ours/TPUv6e.json --adjust-global-buffer-to-memory-capacity 2>&1 | tee "$(LOG_DIR)/full_model_int8_tpuv6e.log"; fi

figures: full-model
	mkdir -p "$(OUT_DIR)/combined_fp16" "$(OUT_DIR)/combined_int8" "$(OUT_DIR)/paper_figures"
	$(PYTHON) plot_customsa_flash_speedups.py --b200-dir "$(OUT_DIR)/full_model_fp16/b200" --awsv4-dir "$(OUT_DIR)/full_model_fp16/awsv4" --tpuv6e-dir "$(OUT_DIR)/full_model_fp16/tpuv6e" --output-dir "$(OUT_DIR)/combined_fp16" --label-mode endpoints 2>&1 | tee "$(LOG_DIR)/combined_fp16.log"
	$(PYTHON) plot_customsa_flash_speedups.py --b200-dir "$(OUT_DIR)/full_model_int8/b200_int8" --awsv4-dir "$(OUT_DIR)/full_model_int8/awsv4_int8" --tpuv6e-dir "$(OUT_DIR)/full_model_int8/tpuv6e_int8" --output-dir "$(OUT_DIR)/combined_int8" --label-mode endpoints 2>&1 | tee "$(LOG_DIR)/combined_int8.log"
	$(PYTHON) plot_paper_speedup_figures.py --figures main --fp16-main-csv "$(OUT_DIR)/combined_fp16/customsa_flash_speedups.csv" --int8-main-csv "$(OUT_DIR)/combined_int8/customsa_flash_speedups.csv" --output-dir "$(OUT_DIR)/paper_figures" 2>&1 | tee "$(LOG_DIR)/paper_figure.log"
