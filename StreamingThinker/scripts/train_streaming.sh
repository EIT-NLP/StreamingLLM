#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_DATASETS_DISABLE_CACHE="${HF_DATASETS_DISABLE_CACHE:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-StreamingThinker}"

MASTER_PORT="${MASTER_PORT:-29511}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_ID="${RDZV_ID:-$RANDOM}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/checkpoints/streaming}"
LOGGING_DIR="${LOGGING_DIR:-$REPO_ROOT/outputs/logs/streaming}"
DATA_CONFIG="${DATA_CONFIG:-$REPO_ROOT/configs/params_qwen_D2.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$REPO_ROOT/configs/deepspeed_config.json}"

torchrun \
  --nnodes="$NUM_NODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank "$NODE_RANK" \
  --rdzv_id "$RDZV_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  "$REPO_ROOT/finetune_streaming.py" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$LOGGING_DIR" \
  --data_config "$DATA_CONFIG" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --batch_from_position_zero \
  --warmup_steps 300 \
  --lr_scheduler_type linear \
  --model_backbone Qwen3 \
  --training_mode streaming \
  --gradient_checkpointing \
  --deepspeed "$DEEPSPEED_CONFIG"
