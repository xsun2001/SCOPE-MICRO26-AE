.DEFAULT_GOAL := help

include ../../mk/common.mk

OUT_DIR ?= $(CURDIR)/actual-results/$(RUN_ID)
LOG_DIR ?= $(CURDIR)/logs/$(RUN_ID)

ifeq ($(FULL_BASELINE),1)
CONV_VARIANTS := baseline,flashattention,customsa
else
CONV_VARIANTS := flashattention,customsa
endif

.PHONY: help run b200 awsv4 tpuv6e figures

help:
	@echo "make run [RUN_ID=<timestamp>] [JOBS=<n>] [FULL_BASELINE=1]"
	@echo "Outputs: $(OUT_DIR)"

run: figures

b200:
	mkdir -p "$(OUT_DIR)/b200" "$(LOG_DIR)"
	if [[ ! -f "$(OUT_DIR)/b200/gpu_conv/attention_latency.csv" ]]; then \
		$(PYTHON) run_condition_parallel.py --experiment-config configs/paper_b200.json \
			--output-dir "$(OUT_DIR)/b200/gpu_conv" --softmax-conversion 1 --jobs "$(JOBS)" \
			--variants "$(CONV_VARIANTS)" --ignore-hbm-bottleneck --ignore-onchip-io-bottleneck \
			2>&1 | tee "$(LOG_DIR)/b200_conv.log"; \
	fi
	if [[ ! -f "$(OUT_DIR)/b200/gpu_no_conv/attention_latency.csv" ]]; then \
		$(PYTHON) run_condition_parallel.py --experiment-config configs/paper_b200.json \
			--output-dir "$(OUT_DIR)/b200/gpu_no_conv" --softmax-conversion 0 --jobs "$(JOBS)" \
			--variants customsa --ignore-hbm-bottleneck --ignore-onchip-io-bottleneck \
			2>&1 | tee "$(LOG_DIR)/b200_no_conv.log"; \
	fi

awsv4:
	mkdir -p "$(OUT_DIR)/awsv4" "$(LOG_DIR)"
	if [[ ! -f "$(OUT_DIR)/awsv4/aws_conv/attention_latency.csv" ]]; then \
		$(PYTHON) run_condition_parallel.py --experiment-config configs/paper_awsv4.json \
			--output-dir "$(OUT_DIR)/awsv4/aws_conv" --softmax-conversion 1 --jobs "$(JOBS)" \
			--variants "$(CONV_VARIANTS)" --ignore-hbm-bottleneck \
			2>&1 | tee "$(LOG_DIR)/awsv4_conv.log"; \
	fi
	if [[ ! -f "$(OUT_DIR)/awsv4/aws_no_conv/attention_latency.csv" ]]; then \
		$(PYTHON) run_condition_parallel.py --experiment-config configs/paper_awsv4.json \
			--output-dir "$(OUT_DIR)/awsv4/aws_no_conv" --softmax-conversion 0 --jobs "$(JOBS)" \
			--variants customsa --ignore-hbm-bottleneck \
			2>&1 | tee "$(LOG_DIR)/awsv4_no_conv.log"; \
	fi

tpuv6e:
	mkdir -p "$(OUT_DIR)/tpuv6e" "$(LOG_DIR)"
	if [[ ! -f "$(OUT_DIR)/tpuv6e/tpu_conv/attention_latency.csv" ]]; then \
		$(PYTHON) run_condition_parallel.py --experiment-config configs/paper_tpuv6e.json \
			--output-dir "$(OUT_DIR)/tpuv6e/tpu_conv" --softmax-conversion 1 --jobs "$(JOBS)" \
			--variants "$(CONV_VARIANTS)" --ignore-hbm-bottleneck \
			2>&1 | tee "$(LOG_DIR)/tpuv6e_conv.log"; \
	fi
	if [[ ! -f "$(OUT_DIR)/tpuv6e/tpu_no_conv/attention_latency.csv" ]]; then \
		$(PYTHON) run_condition_parallel.py --experiment-config configs/paper_tpuv6e.json \
			--output-dir "$(OUT_DIR)/tpuv6e/tpu_no_conv" --softmax-conversion 0 --jobs "$(JOBS)" \
			--variants customsa --ignore-hbm-bottleneck \
			2>&1 | tee "$(LOG_DIR)/tpuv6e_no_conv.log"; \
	fi

figures: b200 awsv4 tpuv6e
	mkdir -p "$(OUT_DIR)/paper_figures" "$(LOG_DIR)"
	$(PYTHON) generate_paper_figures.py \
		--gpu-suite-dir "$(OUT_DIR)/b200" --aws-suite-dir "$(OUT_DIR)/awsv4" \
		--tpu-suite-dir "$(OUT_DIR)/tpuv6e" --output-dir "$(OUT_DIR)/paper_figures" \
		--requested-only --devices b200,awsv4,tpuv6e 2>&1 | tee "$(LOG_DIR)/figures.log"
