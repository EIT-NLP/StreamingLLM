#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4


JOB_NAME="vicostream-finetune-debug"
OUTPUT_DIR="Path/to/your/output/dir"
MODEL_PATH="Path/to/your/model"  # e.g. "ViCoStream/Qwen2.5-VL-3B-Instruct"

# temporal
SIM_THRESHOLD=0.25
DROP_METHOD="chunk-intra"
CHUNK_SIZE=4
# spatial
VISION_POOL_TYPE="none"
SPATIAL_KEEP_RATIO=1.
# attention
NEAREST_WINDOW_SIZE=0
ATTEND_CHUNK_NUM=4
USER_QUERY_RETRIEVAL=16



export MAX_PIXELS=90000
export VIDEO_MAX_PIXELS=90000   # about 300x300 area


MASTER_PORT=$((20000 + RANDOM % 20000))
export MASTER_PORT=$MASTER_PORT

swift sft \
    --model_type qwen2_5_vl \
    --save_strategy steps \
    --model "${MODEL_PATH}" \
    --model_revision main \
    --enable_cache true \
    --freeze_vit true \
    --freeze_aligner false \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --num_train_epochs 1 \
    --attn_impl eager \
    --train_type full \
    --eval_steps 200 \
    --save_steps 200 \
    --torch_dtype bfloat16 \
    --deepspeed zero2 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --truncation_strategy delete \
    --report_to none \
    --sim_threshold ${SIM_THRESHOLD} \
    --vision_pool_type ${VISION_POOL_TYPE} \
    --spatial_keep_ratio ${SPATIAL_KEEP_RATIO} \
    --chunk_size ${CHUNK_SIZE} \
    --drop_method ${DROP_METHOD} \
    --nearest_window_size ${NEAREST_WINDOW_SIZE} \
    --attend_chunk_num ${ATTEND_CHUNK_NUM} \
    --user_query_retrieval ${USER_QUERY_RETRIEVAL} \
    --dataset "Path/to/your/dataset-1" "Path/to/your/dataset-2" 

