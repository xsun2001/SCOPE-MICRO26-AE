SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

BUNDLE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PACKAGED_RUN_ID := 2026-07-13_ae-validation
RUN_ID ?= $(shell date +%Y-%m-%d_%H-%M-%S)

ifeq ($(wildcard $(BUNDLE_ROOT)/.venv/bin/python),)
  ifeq ($(wildcard $(BUNDLE_ROOT)/../.venv/bin/python),)
    PYTHON ?= python3
  else
    PYTHON ?= $(BUNDLE_ROOT)/../.venv/bin/python
  endif
else
  PYTHON ?= $(BUNDLE_ROOT)/.venv/bin/python
endif

CPU_COUNT := $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
JOBS ?= $(shell n=$(CPU_COUNT); if (( n > 8 )); then echo 8; else echo $$n; fi)
LENGTHS ?= 2048,4096,8192,16384,32768,65536,131072,262144,524288

ORIGINAL_PYTHONPATH := $(PYTHONPATH)
export PYTHONPATH := $(BUNDLE_ROOT)/LLMCompass:$(BUNDLE_ROOT)/SCALE-Sim$(if $(ORIGINAL_PYTHONPATH),:$(ORIGINAL_PYTHONPATH))
export MPLBACKEND := Agg
export LLMCOMPASS_CUSTOMSA_STAGE_OVERHEAD_CYCLES := 8
export RUN_ID JOBS PYTHON
