#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
project_root="$(cd -- "${repo_dir}/.." && pwd)"
cd "${repo_dir}"

timestamp="$(date +%F_%H%M%S)"
result_dir="${OUTPUT_DIR:-${project_root}/experiments/end2endacc_llama2_7b_int8_ppl/results/${timestamp}}"
mkdir -p "${result_dir}"

model="${MODEL:-${project_root}/models/Llama-2-7b-hf}"
torch_dtype="${TORCH_DTYPE:-bfloat16}"
dataset_name="${DATASET_NAME:-wikitext}"
dataset_config="${DATASET_CONFIG:-wikitext-2-raw-v1}"
dataset_split="${DATASET_SPLIT:-test}"
dataset_text_column="${DATASET_TEXT_COLUMN:-text}"
dataset_joiner="${DATASET_JOINER:-$'\n\n'}"
sequence_length="${SEQUENCE_LENGTH:-2048}"
batch_size="${BATCH_SIZE:-1}"
num_samples="${NUM_SAMPLES:-}"
max_blocks="${MAX_BLOCKS:-}"
max_chunks="${MAX_CHUNKS:-}"
max_tokens="${MAX_TOKENS:-}"

cmd=(
    "${PYTHON_BIN:-python3}" evaluation/wikitext/evaluate_hf.py
    --model "${model}"
    --dtype "${torch_dtype}"
    --dataset_name "${dataset_name}"
    --dataset_config "${dataset_config}"
    --dataset_split "${dataset_split}"
    --dataset_text_column "${dataset_text_column}"
    --dataset_joiner "${dataset_joiner}"
    --sequence_length "${sequence_length}"
    --batch_size "${batch_size}"
    --output_dir "${result_dir}"
)

if [[ -n "${num_samples}" ]]; then
    cmd+=(--num_samples "${num_samples}")
fi
if [[ -n "${max_blocks}" ]]; then
    cmd+=(--max_blocks "${max_blocks}")
fi
if [[ -n "${max_chunks}" ]]; then
    cmd+=(--max_chunks "${max_chunks}")
fi
if [[ -n "${max_tokens}" ]]; then
    cmd+=(--max_tokens "${max_tokens}")
fi
if [[ -n "${BACKBONE_CALIBRATION_SAMPLES:-}" ]]; then
    cmd+=(--backbone_calibration_samples "${BACKBONE_CALIBRATION_SAMPLES}")
fi
if [[ -n "${BACKBONE_CALIBRATION_SEQ_LEN:-}" ]]; then
    cmd+=(--backbone_calibration_seq_len "${BACKBONE_CALIBRATION_SEQ_LEN}")
fi
if [[ -n "${APPROX_BACKEND:-}" && "${APPROX_BACKEND}" != "none" ]]; then
    if [[ -z "${APPROX_SCOPE:-}" ]]; then
        echo "APPROX_SCOPE must be set explicitly when APPROX_BACKEND is enabled." >&2
        exit 1
    fi
    cmd+=(--approx_backend "${APPROX_BACKEND}" --approx_scope "${APPROX_SCOPE}")
    if [[ -n "${APPROX_EXP_LUT_PATH:-}" ]]; then
        cmd+=(--approx_exp_lut_path "${APPROX_EXP_LUT_PATH}")
    fi
    if [[ -n "${APPROX_EXP_LUT_BITS:-}" ]]; then
        cmd+=(--approx_exp_lut_bits "${APPROX_EXP_LUT_BITS}")
    fi
fi
if [[ "${QUANT_APPROX_WEIGHTS:-0}" == "1" ]]; then
    cmd+=(--quant_approx_weights --w_bits "${W_BITS:-8}" --w_mantissa_bit "${W_MANTISSA_BIT:-2}")
    if [[ "${W_PER_TENSOR:-0}" == "1" ]]; then
        cmd+=(--w_per_tensor)
    fi
fi
if [[ "${QUANT_APPROX_ACTIVATIONS:-0}" == "1" ]]; then
    cmd+=(--quant_approx_activations --a_bits "${A_BITS:-8}" --a_mantissa_bit "${A_MANTISSA_BIT:-2}" --calibrate_static_act)
    if [[ "${A_PER_TENSOR:-1}" == "1" ]]; then
        cmd+=(--a_per_tensor)
    fi
fi
if [[ "${PINN:-0}" == "1" ]]; then
    cmd+=(--pinn --pinn_dim "${PINN_DIM:-16}")
fi
if [[ "${QUANT_PINN_WEIGHTS:-0}" == "1" ]]; then
    cmd+=(--quant_pinn_weights --w_bits "${W_BITS:-8}" --w_mantissa_bit "${W_MANTISSA_BIT:-2}")
    if [[ "${W_PER_TENSOR:-0}" == "1" ]]; then
        cmd+=(--w_per_tensor)
    fi
fi
if [[ "${QUANT_PINN_ACTIVATIONS:-0}" == "1" ]]; then
    cmd+=(--quant_pinn_activations --a_bits "${A_BITS:-8}" --a_mantissa_bit "${A_MANTISSA_BIT:-2}" --calibrate_static_act)
    if [[ "${A_PER_TENSOR:-1}" == "1" ]]; then
        cmd+=(--a_per_tensor)
    fi
fi
if [[ "${FPQ:-0}" == "1" ]]; then
    cmd+=(--fpq)
fi
if [[ "${QUANT_BACKBONE:-0}" == "1" ]]; then
    cmd+=(
        --quant_backbone
        --backbone_w_bits "${BACKBONE_W_BITS:-8}"
        --backbone_a_bits "${BACKBONE_A_BITS:-8}"
        --backbone_weight_dtype "${BACKBONE_WEIGHT_DTYPE:-int8}"
        --backbone_activation_dtype "${BACKBONE_ACTIVATION_DTYPE:-int8}"
        --backbone_weight_scheme "${BACKBONE_WEIGHT_SCHEME:-per_channel}"
        --backbone_act_scheme "${BACKBONE_ACT_SCHEME:-per_tensor}"
        --backbone_calibration "${BACKBONE_CALIBRATION:-static}"
    )
    if [[ "${BACKBONE_SMOOTHQUANT:-0}" == "1" ]]; then
        cmd+=(--backbone_smoothquant --backbone_smoothquant_alpha "${BACKBONE_SMOOTHQUANT_ALPHA:-0.85}")
    fi
fi
if [[ "${QUANTIZE_LM_HEAD:-0}" == "1" ]]; then
    cmd+=(--quantize_lm_head)
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    cmd+=(--trust_remote_code)
fi
if [[ "${USE_FAST_TOKENIZER:-0}" == "1" ]]; then
    cmd+=(--use_fast_tokenizer)
fi

printf '%q ' "${cmd[@]}" > "${result_dir}/command.txt"
printf '\n' >> "${result_dir}/command.txt"
if command -v git >/dev/null 2>&1 && git -C "${project_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${project_root}" rev-parse HEAD > "${result_dir}/git_commit.txt"
else
    printf 'unknown (source archive is not a Git worktree)\n' > "${result_dir}/git_commit.txt"
fi
printf -v quoted_cmd '%q ' "${cmd[@]}"

set +e
if [[ "${END2ENDACC_DIRECT:-0}" == "1" ]]; then
    "${cmd[@]}" 2>&1 | tee "${result_dir}/stdout.log"
    run_status=${PIPESTATUS[0]}
else
    srun -J "${JOB_NAME:-end2endacc_wikitext}" -p acd_u -n 1 --cpus-per-task=8 --mem=16G --gres=gpu:1 \
        bash -lc "source '${project_root}/.venv/bin/activate' && cd '${repo_dir}' && ${quoted_cmd}" \
        2>&1 | tee "${result_dir}/stdout.log"
    run_status=${PIPESTATUS[0]}
fi
set -e

if [[ -f "${result_dir}/metrics.json" && -f "${result_dir}/config.json" ]]; then
    "${PYTHON_BIN:-python3}" evaluation/write_run_summary.py --result_dir "${result_dir}"
fi

exit "${run_status}"
