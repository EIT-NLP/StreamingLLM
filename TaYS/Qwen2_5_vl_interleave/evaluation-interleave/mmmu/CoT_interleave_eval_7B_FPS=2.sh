
#!/bin/bash

MODEL_PATH="../../qwen-vl-finetune/output/qwen2_5vl-interleave-CoT-7B-20251029_0304"
DATASET="VideoEspresso"
OUTPUT_PREFIX="../../../infer_results/CoT/qwen2_5vl-CoT-interleave-7B-1029-FPS=2"

python run_mmmu.py \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-file "${OUTPUT_PREFIX}.jsonl" \
    --CoT