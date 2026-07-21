#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
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
MODE=${MODE:?MODE must be set}

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

SCNA_ARGS=()
case "$MODE" in
  exact_eager)
    ;;
  scna_d8)
    SCNA_ARGS=(--scna=True --scna_dim=8)
    ;;
  scna_d16)
    SCNA_ARGS=(--scna=True --scna_dim=16)
    ;;
  scna_d32)
    SCNA_ARGS=(--scna=True --scna_dim=32)
    ;;
  scna_d16_fit*)
    SCNA_ARGS=(--scna=True --scna_dim=16)
    ;;
  scna_d32_fit*)
    SCNA_ARGS=(--scna=True --scna_dim=32)
    ;;
  scna_d16_clip8)
    SCNA_ARGS=(--scna=True --scna_dim=16 --scna_input_clip_min=-8)
    ;;
  scna_d32_clip8)
    SCNA_ARGS=(--scna=True --scna_dim=32 --scna_input_clip_min=-8)
    ;;
  scna_d32_floor12)
    SCNA_ARGS=(--scna=True --scna_dim=32 --scna_output_floor_log=-12)
    ;;
  scna_d32_floor16)
    SCNA_ARGS=(--scna=True --scna_dim=32 --scna_output_floor_log=-16)
    ;;
  scna_d16_floor12)
    SCNA_ARGS=(--scna=True --scna_dim=16 --scna_output_floor_log=-12)
    ;;
  scna_d16_floor16)
    SCNA_ARGS=(--scna=True --scna_dim=16 --scna_output_floor_log=-16)
    ;;
  scna_d16_clip6)
    SCNA_ARGS=(--scna=True --scna_dim=16 --scna_input_clip_min=-6)
    ;;
  scna_d32_clip6)
    SCNA_ARGS=(--scna=True --scna_dim=32 --scna_input_clip_min=-6)
    ;;
  scna_d16_q6in)
    SCNA_ARGS=(--scna=True --scna_dim=16 --scna_input_quant_bits=6)
    ;;
  scna_d16_q4in)
    SCNA_ARGS=(--scna=True --scna_dim=16 --scna_input_quant_bits=4)
    ;;
  scna_d16_clip8_q6in)
    SCNA_ARGS=(--scna=True --scna_dim=16 --scna_input_clip_min=-8 --scna_input_quant_bits=6)
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

if [[ -n "${SCNA_ARTIFACT_ROOT:-}" ]]; then
  SCNA_ARGS+=(--scna_artifact_root="$SCNA_ARTIFACT_ROOT")
fi

EXACT_SUFFIX=${EXACT_SUFFIX:-sdpa}
EXACT_DIR="$RUN_ROOT/exact_${MODEL_KEY}_${QUANT_KEY}_${EXACT_SUFFIX}"
RESUME_PATH=${RESUME_PATH:-"$EXACT_DIR/model.bin"}
if [[ ! -f "$RESUME_PATH" ]]; then
  echo "Missing resume checkpoint: $RESUME_PATH" >&2
  exit 3
fi

QMODEL_SUFFIX=${QMODEL_SUFFIX:-$EXACT_SUFFIX}
DEFAULT_QMODEL_PATH="$RUN_ROOT/qmodel_${MODEL_KEY}_${QUANT_KEY}_${QMODEL_SUFFIX}/qmodel.pt"
LOAD_QMODEL_PATH=${LOAD_QMODEL_PATH:-}
if [[ -z "$LOAD_QMODEL_PATH" && -f "$DEFAULT_QMODEL_PATH" ]]; then
  LOAD_QMODEL_PATH="$DEFAULT_QMODEL_PATH"
fi
QMODEL_ARGS=()
if [[ -n "$LOAD_QMODEL_PATH" ]]; then
  if [[ ! -f "$LOAD_QMODEL_PATH" ]]; then
    echo "Missing quantized model: $LOAD_QMODEL_PATH" >&2
    exit 4
  fi
  QMODEL_ARGS=(--load_qmodel_path="$LOAD_QMODEL_PATH")
fi

OUT_SUFFIX=${OUT_SUFFIX:-}
if [[ -n "$OUT_SUFFIX" ]]; then
  OUT_DIR="$RUN_ROOT/eval_${MODEL_KEY}_${QUANT_KEY}_${MODE}_${OUT_SUFFIX}"
else
  OUT_DIR="$RUN_ROOT/eval_${MODEL_KEY}_${QUANT_KEY}_${MODE}"
fi
mkdir -p "$OUT_DIR"
cd "$CODE_ROOT"
LM_EVAL=${LM_EVAL:-True}
TABLE5_TASKS=(arc_easy hellaswag piqa winogrande)

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
  --lm_eval="$LM_EVAL"
  --tasks "${TABLE5_TASKS[@]}"
  --use_sdpa=False
  --per_device_train_batch_size=4
  --max_steps=100
  --train_enable_wquant=False
  --sub_mean=False
  --use_klt=False
  --train_rotate=False
  --resume_path="$RESUME_PATH"
  --report_to=none
)

CMD=(
  "$PYTHON_BIN"
  main.py
  "${COMMON_ARGS[@]}"
  "${QUANT_ARGS[@]}"
  "${SCNA_ARGS[@]}"
  "${QMODEL_ARGS[@]}"
)

printf '%q ' "${CMD[@]}" > "$OUT_DIR/command.sh"
printf '\n' >> "$OUT_DIR/command.sh"
cat > "$OUT_DIR/slurm_env.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-}",
  "job_name": "${SLURM_JOB_NAME:-}",
  "model_key": "$MODEL_KEY",
  "quant_key": "$QUANT_KEY",
  "mode": "$MODE",
  "exact_suffix": "$EXACT_SUFFIX",
  "qmodel_suffix": "$QMODEL_SUFFIX",
  "out_suffix": "$OUT_SUFFIX",
  "run_root": "$RUN_ROOT",
  "resume_path": "$RESUME_PATH",
  "load_qmodel_path": "$LOAD_QMODEL_PATH",
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
