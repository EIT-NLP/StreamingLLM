
#!/bin/bash
export FPS=1

MODEL_PATH="../../qwen-vl-finetune/output/qwen2_5vl-interleave-CoT-3B-20251027_1824"
DATASET="VideoEspresso"
OUTPUT_PREFIX="../../../infer_results/CoT/qwen2_5vl-CoT-interleave-3B-1027-FPS=1"

python run_mmmu.py \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-file "${OUTPUT_PREFIX}.jsonl" \
    --CoT