#!/bin/bash


# MODEL_PATH="/code/model/Qwen2.5-VL-3B-Instruct"
MODEL_PATH="../../qwen-vl-finetune/output/qwen2_5vl-batch-CoT-7B-20251025_1703"
DATASET="VideoEspresso"
OUTPUT_PREFIX="../../../infer_results/CoT/qwen2_5vl-CoT-batchQA-7B-1025"

# ========================================

# ========================================


TIME_LOG="${OUTPUT_PREFIX}-timing.log"


get_timestamp() {
    date +%s
}


get_formatted_time() {
    date '+%Y-%m-%d %H:%M:%S'
}


format_duration() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%02d:%02d:%02d" $hours $minutes $seconds
    elif [ $minutes -gt 0 ]; then
        printf "%02d:%02d" $minutes $seconds
    else
        printf "%ds" $seconds
    fi
}


log_time() {
    local message="$1"
    local timestamp="$(get_formatted_time)"
    echo "[$timestamp] $message" | tee -a "$TIME_LOG"
}


start_stage() {
    local stage_name="$1"
    local var_name="$2"
    local start_time=$(get_timestamp)
    eval "${var_name}=$start_time"
    log_time "🚀 开始 $stage_name"
    echo "⏰ 开始时间: $(get_formatted_time)"
}


end_stage() {
    local stage_name="$1"
    local start_var="$2"
    local end_time=$(get_timestamp)
    local start_time=$(eval echo \$$start_var)
    local duration=$((end_time - start_time))
    local formatted_duration=$(format_duration $duration)
    
    log_time "✅ 完成 $stage_name - 耗时: $formatted_duration"
    echo "⏰ 结束时间: $(get_formatted_time)"
    echo "⏱️  阶段耗时: $formatted_duration"
    echo ""
}

# ========================================

# ========================================


echo "========================================" > "$TIME_LOG"
echo "CoT Group Evaluation - 执行时间记录" >> "$TIME_LOG"
echo "开始时间: $(get_formatted_time)" >> "$TIME_LOG"
echo "========================================" >> "$TIME_LOG"


TOTAL_START_TIME=$(get_timestamp)
log_time "🚀 开始三阶段评估流程"
echo "🚀 开始三阶段评估流程..."
echo "⏰ 总体开始时间: $(get_formatted_time)"
echo ""

# ----------------------------------------

# ----------------------------------------
start_stage "第一阶段：运行模型生成推理过程 (CoT)" "STAGE1_START"
echo "🔍 第一阶段：运行模型生成推理过程 (CoT)..."

python run_mmmu.py \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-file "${OUTPUT_PREFIX}.jsonl" \
    --CoT

if [ $? -ne 0 ]; then
    log_time "❌ 第一阶段失败！"
    echo "❌ 第一阶段失败！"
    exit 1
fi

end_stage "第一阶段" "STAGE1_START"

# ----------------------------------------

# ----------------------------------------
start_stage "第二阶段：让模型从选项中选择答案" "STAGE2_START"
echo "📝 第二阶段：让模型从选项中选择答案..."

python CoT_choice.py \
    --input_jsonl "${OUTPUT_PREFIX}.jsonl" \
    --output_jsonl "${OUTPUT_PREFIX}-choice.jsonl" \
    --model_path "$MODEL_PATH"

if [ $? -ne 0 ]; then
    log_time "❌ 第二阶段失败！"
    echo "❌ 第二阶段失败！"
    exit 1
fi

end_stage "第二阶段" "STAGE2_START"

# ----------------------------------------

# ----------------------------------------
start_stage "第三阶段：计算准确率" "STAGE3_START"
echo "📊 第三阶段：计算准确率..."

python CoT_score.py \
    --input_jsonl "${OUTPUT_PREFIX}-choice.jsonl" \
    --output_result "${OUTPUT_PREFIX}-score.json"

if [ $? -ne 0 ]; then
    log_time "❌ 第三阶段失败！"
    echo "❌ 第三阶段失败！"
    exit 1
fi

end_stage "第三阶段" "STAGE3_START"

# ========================================

# ========================================
TOTAL_END_TIME=$(get_timestamp)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_FORMATTED_DURATION=$(format_duration $TOTAL_DURATION)

log_time "🎉 全部流程完成！总耗时: $TOTAL_FORMATTED_DURATION"
echo "✅ 全部流程完成！结果已生成。"
echo ""
echo "=========================================="
echo "📊 执行时间汇总报告"
echo "=========================================="
echo "⏰ 总体开始时间: $(date -d @$TOTAL_START_TIME '+%Y-%m-%d %H:%M:%S')"
echo "⏰ 总体结束时间: $(date -d @$TOTAL_END_TIME '+%Y-%m-%d %H:%M:%S')"
echo "⏱️  总体执行时间: $TOTAL_FORMATTED_DURATION"
echo "📄 详细时间日志: $TIME_LOG"
echo "📁 输出文件前缀: $OUTPUT_PREFIX"
echo "=========================================="


echo "" >> "$TIME_LOG"
echo "========================================" >> "$TIME_LOG"
echo "执行时间汇总" >> "$TIME_LOG"
echo "总体开始时间: $(date -d @$TOTAL_START_TIME '+%Y-%m-%d %H:%M:%S')" >> "$TIME_LOG"
echo "总体结束时间: $(date -d @$TOTAL_END_TIME '+%Y-%m-%d %H:%M:%S')" >> "$TIME_LOG"
echo "总体执行时间: $TOTAL_FORMATTED_DURATION" >> "$TIME_LOG"
echo "========================================" >> "$TIME_LOG"