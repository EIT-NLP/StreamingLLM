
#!/bin/bash
export FPS=2

# MODEL_PATH="/code/model/Qwen2.5-VL-3B-Instruct"
MODEL_PATH="../../qwen-vl-finetune/output/qwen2_5vl-group-CoT-7B-20251029_0453"
DATASET="VideoEspresso"
OUTPUT_PREFIX="../../../infer_results/CoT/qwen2_5vl-CoT-group-7B-1029-FPS=2"

python run_mmmu.py \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-file "${OUTPUT_PREFIX}.jsonl" \
    --CoT