#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o experiments/ostquant_scna_investigation/slurm/%x_%j.out
#SBATCH -e experiments/ostquant_scna_investigation/slurm/%x_%j.err

set -euo pipefail

EXP_ROOT=${EXP_ROOT:?EXP_ROOT must be set}
BUNDLE_ROOT=${BUNDLE_ROOT:?BUNDLE_ROOT must be set}
MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT must be set}
PYTHON_BIN=${PYTHON_BIN:-python3}
CODE_ROOT="$BUNDLE_ROOT/OSTQuant"
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be set}
MODEL_KEY=${MODEL_KEY:?MODEL_KEY must be set}
QUANT_KEY=${QUANT_KEY:?QUANT_KEY must be set}
EXACT_SUFFIX=${EXACT_SUFFIX:-sdpa}
USE_SDPA=${USE_SDPA:-True}

case "$MODEL_KEY" in
  llama2_7b) MODEL_PATH="$MODEL_ROOT/Llama-2-7b-hf" ;;
  llama3_8b) MODEL_PATH="$MODEL_ROOT/Llama-3-8b" ;;
  *) echo "Unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac

case "$QUANT_KEY" in
  w4a4kv4) QUANT_ARGS=(--w_bits=4 --a_bits=4 --v_bits=4 --k_bits=4 --down_bits=4) ;;
  w6a6kv6) QUANT_ARGS=(--w_bits=6 --a_bits=6 --v_bits=6 --k_bits=6 --down_bits=6) ;;
  *) echo "Unknown QUANT_KEY=$QUANT_KEY" >&2; exit 2 ;;
esac

EXACT_DIR="$RUN_ROOT/exact_${MODEL_KEY}_${QUANT_KEY}_${EXACT_SUFFIX}"
RESUME_PATH=${RESUME_PATH:-"$EXACT_DIR/model.bin"}
if [[ ! -f "$RESUME_PATH" ]]; then
  echo "Missing resume checkpoint: $RESUME_PATH" >&2
  exit 3
fi

QMODEL_SUFFIX=${QMODEL_SUFFIX:-$EXACT_SUFFIX}
OUT_DIR="$RUN_ROOT/qmodel_${MODEL_KEY}_${QUANT_KEY}_${QMODEL_SUFFIX}"
QMODEL_PATH="$OUT_DIR/qmodel.pt"
mkdir -p "$OUT_DIR"
cd "$CODE_ROOT"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

COMMON_ARGS=(
  --output_dir="$OUT_DIR"
  --model="$MODEL_PATH"
  --loss_type=kl_top
  --post_attn=True
  --rotate_ov=True
  --rotate_post_rope=False
  --online_qk_hadamard=True
  --smooth_qk=True
  --smooth_ov=True
  --smooth_up_down=True
  --smooth_norm_linear=True
  --bf16=True
  --lm_eval=False
  --use_sdpa="$USE_SDPA"
  --per_device_train_batch_size=4
  --max_steps=100
  --train_enable_wquant=False
  --sub_mean=False
  --use_klt=False
  --train_rotate=False
  --resume_path="$RESUME_PATH"
  --save_qmodel_path="$QMODEL_PATH"
  --report_to=none
)

CMD=(
  "$PYTHON_BIN"
  main.py
  "${COMMON_ARGS[@]}"
  "${QUANT_ARGS[@]}"
)

printf '%q ' "${CMD[@]}" > "$OUT_DIR/command.sh"
printf '\n' >> "$OUT_DIR/command.sh"
cat > "$OUT_DIR/slurm_env.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-}",
  "job_name": "${SLURM_JOB_NAME:-}",
  "model_key": "$MODEL_KEY",
  "quant_key": "$QUANT_KEY",
  "mode": "qmodel_${QMODEL_SUFFIX}_gptq",
  "use_sdpa": "$USE_SDPA",
  "run_root": "$RUN_ROOT",
  "resume_path": "$RESUME_PATH",
  "qmodel_path": "$QMODEL_PATH",
  "node_list": "${SLURM_NODELIST:-}",
  "started_at": "$(date --iso-8601=seconds)"
}
EOF

set +e
"${CMD[@]}"
RC=$?
set -e

cat > "$OUT_DIR/status.json" <<EOF
{
  "returncode": $RC,
  "finished_at": "$(date --iso-8601=seconds)"
}
EOF
exit "$RC"
