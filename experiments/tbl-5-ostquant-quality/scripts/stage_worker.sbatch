#!/usr/bin/env bash
#SBATCH -J scope_ae_tbl5
#SBATCH -p acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail
EXP_ROOT=${EXP_ROOT:?EXP_ROOT is required}
BUNDLE_ROOT=${BUNDLE_ROOT:?BUNDLE_ROOT is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
STAGE=${STAGE:?STAGE is required}
PYTHON_BIN=${PYTHON_BIN:-python3}
WORKER_COUNT=${WORKER_COUNT:-1}
TASK_SLOT=${SLURM_ARRAY_TASK_ID:-0}
RESULT_DIR="$RUN_ROOT/tbl-5-ostquant-quality"

case "$STAGE" in
  exact|qmodel) MANIFEST="$EXP_ROOT/data/manifest-exact.tsv" ;;
  eval) MANIFEST="$EXP_ROOT/data/manifest-eval.tsv" ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
TASK_COUNT=$(( $(wc -l < "$MANIFEST") - 1 ))
run_status=0

for task_index in $(seq "$TASK_SLOT" "$WORKER_COUNT" "$((TASK_COUNT - 1))"); do
  line=$(sed -n "$((task_index + 2))p" "$MANIFEST")
  line=${line%$'\r'}
  if [[ "$STAGE" == "eval" ]]; then
    IFS=$'\t' read -r MODEL_KEY QUANT_KEY MODE <<<"$line"
    export MODEL_KEY QUANT_KEY MODE
    export EXACT_SUFFIX=sdpa QMODEL_SUFFIX=sdpa OUT_SUFFIX=maskfix_acc LM_EVAL=True
    script="$EXP_ROOT/scripts/eval_resume.sbatch"
  else
    IFS=$'\t' read -r MODEL_KEY QUANT_KEY <<<"$line"
    export MODEL_KEY QUANT_KEY
    script="$EXP_ROOT/scripts/$([[ "$STAGE" == exact ]] && echo train_exact || echo save_qmodel).sbatch"
  fi
  echo "[$(date --iso-8601=seconds)] stage=$STAGE task=$task_index model=$MODEL_KEY quant=$QUANT_KEY mode=${MODE:-}"
  if ! BUNDLE_ROOT="$BUNDLE_ROOT" EXP_ROOT="$EXP_ROOT" MODEL_ROOT="$MODEL_ROOT" PYTHON_BIN="$PYTHON_BIN" RUN_ROOT="$RESULT_DIR" bash "$script"; then
    run_status=1
  fi
done
exit "$run_status"
