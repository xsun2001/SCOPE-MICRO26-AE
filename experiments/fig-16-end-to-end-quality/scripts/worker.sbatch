#!/usr/bin/env bash
#SBATCH -J scope_ae_fig16
#SBATCH -p acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

set -euo pipefail
EXP_ROOT=${EXP_ROOT:?EXP_ROOT is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:-python3}
WORKER_COUNT=${WORKER_COUNT:-1}
TASK_SLOT=${SLURM_ARRAY_TASK_ID:-0}
MANIFEST="$EXP_ROOT/data/manifest.tsv"
TASK_COUNT=$(( $(wc -l < "$MANIFEST") - 1 ))
export END2ENDACC_DIRECT=1
run_status=0

for task_index in $(seq "$TASK_SLOT" "$WORKER_COUNT" "$((TASK_COUNT - 1))"); do
  line=$(sed -n "$((task_index + 2))p" "$MANIFEST")
  line=${line%$'\r'}
  IFS=$'\t' read -r kind model variant config_rel <<<"$line"
  model_slug=$(printf '%s' "$model" | tr '[:upper:].-' '[:lower:]__')
  out_dir="$RUN_ROOT/fig-16-end-to-end-quality/$kind/$model_slug/$variant"
  mkdir -p "$out_dir"
  echo "[$(date --iso-8601=seconds)] task=$task_index kind=$kind model=$model variant=$variant"
  if ! "$PYTHON_BIN" "$EXP_ROOT/scripts/end2endacc_runner.py" --kind "$kind" \
    --config "$EXP_ROOT/$config_rel" --output-dir "$out_dir"; then
    run_status=1
  fi
done
exit "$run_status"
