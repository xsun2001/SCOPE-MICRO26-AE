BUNDLE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(BUNDLE_ROOT)/config/default.env
-include $(BUNDLE_ROOT)/config/local.env

export BUNDLE_ROOT RUN_ROOT MODEL_ROOT PYTHON
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1
export TOKENIZERS_PARALLELISM := false

define check-executor
	@if [[ "$(EXECUTOR)" != "slurm" && "$(EXECUTOR)" != "local" ]]; then \
		echo "EXECUTOR must be 'slurm' or 'local' (got '$(EXECUTOR)')" >&2; exit 2; \
	fi
endef
