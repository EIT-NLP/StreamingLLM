export MAX_PIXELS=90000
export VIDEO_MAX_PIXELS=90000
export PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-4}
export NPROC_PER_NODE=1

cd "$(dirname "$0")/../../.."

CKPT_PATH="Qwen2.5-VL-3B-Instruct"
RUN_NAME="timechat-none-baseline-s0_15-eager-chunk-intra-cs4-acn4-ur64-qwen7B-streambench-v0_3-300x300"
DATA_ROOT="datasets/StreamBench_v0.3"
TASK_JSON="${DATA_ROOT}/streaming_bench_v0.3_incremental_qa.json"
SOURCE_JSON="${DATA_ROOT}/streaming_bench_v0.3.json"
RESULT_DIR="outputs/eval/streambench_v0_3/${RUN_NAME}"

${PYTHON_BIN} eval/scripts/increnmental_stream.py \
   --run_name ${RUN_NAME} \
   --drop_method "chunk-intra" \
   --drop_threshold 0.15 \
   --max_frames 2048 \
   --ckpt_path ${CKPT_PATH} \
   --data_root ${DATA_ROOT} \
   --task_json ${TASK_JSON} \
   --source_json ${SOURCE_JSON} \
   --result_dir ${RESULT_DIR} \
   --min_pixels 90000 \
   --max_pixels 90000 \
   --spatial_keep_ratio 0. \
   --vision_pool_type none \
   --input_step 8 \
   --chunk_size 4 \
   --nearest_window_size 0 \
   --attend_chunk_num 4 \
   --user_query_retrieval 64 \
   --drop_relative
