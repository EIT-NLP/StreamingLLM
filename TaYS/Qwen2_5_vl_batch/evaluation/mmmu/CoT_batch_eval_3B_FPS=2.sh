#!/bin/bash
export FPS=2


# MODEL_PATH="/code/model/Qwen2.5-VL-3B-Instruct"
MODEL_PATH="../../qwen-vl-finetune/output/qwen2_5vl-batch-CoT-3B-20251027_1545"
DATASET="VideoEspresso"
OUTPUT_PREFIX="../../../infer_results/FPS/qwen2_5vl-CoT-batch-3B-1027-FPS=2"

python run_mmmu.py \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-file "${OUTPUT_PREFIX}.jsonl" \
    --CoT