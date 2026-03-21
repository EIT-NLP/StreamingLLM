#!/bin/bash

# Distributed training configuration
NPROC_PER_NODE=4
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


PY_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PY_ROOT:$PYTHONPATH"
export VIDEO_MIN_PIXELS=78400 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=60 # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=2408448 # 24576*28*28. the maximum overall video tokens sent to llm is 24k (leave 8k for language)

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
if [ -d "../../../../model/Qwen2.5-VL-7B-Instruct" ]; then
    llm=../../../../model/Qwen2.5-VL-7B-Instruct  # Using local model
else
    llm=Qwen/Qwen2.5-VL-7B-Instruct  # Using HuggingFace model ID
fi

# Training hyperparameters
lr=2e-5
batch_size=1
grad_accum_steps=16

# Training entry point
entry_file=qwenvl/train/train_livecc.py

# Dataset configuration (replace with public dataset names)
datasets=../../../../data/VideoEspresso/videoespresso_train_video_5s-30s_final_with_skip.jsonl

# Output configuration
run_name="qwen2_5vl-group-CoT"
output_dir=./output/qwen2_5vl-group-CoT-7B-$(date +"%Y%m%d_%H%M")

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --use_liger_kernel True \
    --run_name ${run_name} \
    --report_to none \
    --randK False \
    --randF False \
    "

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}