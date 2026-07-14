device=1

# model="Qwen/Qwen2.5-0.5B-Instruct"
# model="Qwen/Qwen2.5-1.5B-Instruct"
# model="Qwen/Qwen2.5-3B-Instruct"
# model="Qwen/Qwen2.5-7B-Instruct"
# model="Qwen/Qwen2.5-14B-Instruct"
# model="Qwen/Qwen2.5-32B-Instruct"

# model="facebook/opt-125m"
# model="facebook/opt-350m"
# model="facebook/opt-1.3b"
# model="facebook/opt-2.7b"
# model="facebook/opt-6.7b"
# model="facebook/opt-13b"


model="../models/Llama-2-7b-hf"
# model="/local-ssd/jiaxiang/models/llama/llama2_70b_hf"


torch_dtype="float16"

exp_method="plf"


srun -J end2endacc_finetune -p acd_u -n 1 --cpus-per-task=8 --mem=16G --gres=gpu:1 python finetune/finetune.py \
    --model ${model} \
    --dtype ${torch_dtype} \
    --use_lora \
    --num_epochs 10 \
    --learning_rate 2e-4 \
    --custom_attention \
    --exp_method ${exp_method} \
