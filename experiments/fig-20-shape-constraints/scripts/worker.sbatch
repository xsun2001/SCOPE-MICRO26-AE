#!/usr/bin/env bash
#SBATCH -J scope_ae_fig20
#SBATCH -p acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

set -euo pipefail
EXP_ROOT=${EXP_ROOT:?EXP_ROOT is required}
BUNDLE_ROOT=${BUNDLE_ROOT:?BUNDLE_ROOT is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:-python3}
WORKER_COUNT=${WORKER_COUNT:-1}
TASK_SLOT=${SLURM_ARRAY_TASK_ID:-0}
MANIFEST="$EXP_ROOT/data/manifest.tsv"
TASK_COUNT=$(( $(wc -l < "$MANIFEST") - 1 ))
run_status=0

for task_index in $(seq "$TASK_SLOT" "$WORKER_COUNT" "$((TASK_COUNT - 1))"); do
  line=$(sed -n "$((task_index + 2))p" "$MANIFEST")
  line=${line%$'\r'}
  IFS=$'\t' read -r func width reparam run_name <<<"$line"
  out_dir="$RUN_ROOT/fig-20-shape-constraints/runs/$run_name"
  range_args=()
  if [[ "$func" == "rsqrt" ]]; then
    range_args=(--l-range -256 --r-range -1)
  fi
  mkdir -p "$out_dir"
  echo "[$(date --iso-8601=seconds)] task=$task_index func=$func width=$width reparam=$reparam"
  if ! "$PYTHON_BIN" "$BUNDLE_ROOT/train/train.py" \
    --func "$func" --num-units "$width" --reparam "$reparam" --l-bound 0.15 "${range_args[@]}" \
    --batch-size 128 --optim Adam --lr 0.05 --lr-scheduler ReduceLROnPlateau \
    --num-epochs 30000 --train-data-shape 4096 4096 --eval-points 1000 \
    --seed 42 --device cuda --pin-memory --early-stop-patience 0 --output-dir "$out_dir"; then
    run_status=1
  fi
done
exit "$run_status"
