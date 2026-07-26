export MAX_PIXELS=200704
export VIDEO_MAX_PIXELS=200704
export PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-4}
export NPROC_PER_NODE=1

cd "$(dirname "$0")/../../.."

CKPT_PATH="Qwen2.5-VL-3B-Instruct"
RUN_NAME="timechat-none-baseline-s0_5-eager-chunk-intra-cs4-acn4-ur64-qwen7B-vstream-ego4d-300x300"
TASK_JSON="datasets/VStream-QA/vstream-realtime/test_qa_ego4d.json"
VIDEO_DIR="datasets/VStream-QA/vstream-realtime/ego4d_frames"
RESULT_DIR="outputs/eval/vstream-ego4d/${RUN_NAME}"

${PYTHON_BIN} eval/scripts/incremental_vstream_ego4d.py \
   --run_name ${RUN_NAME} \
   --drop_method "chunk-intra" \
   --drop_threshold 0.5 \
   --max_frames 2048 \
   --ckpt_path ${CKPT_PATH} \
   --task_json ${TASK_JSON} \
   --video_dir ${VIDEO_DIR} \
   --result_dir ${RESULT_DIR} \
   --min_pixels 90000 \
   --max_pixels 90000 \
   --spatial_keep_ratio 0. \
   --vision_pool_type none \
   --input_step 8 \
   --chunk_size 4 \
   --nearest_window_size 0 \
   --attend_chunk_num 16 \
   --user_query_retrieval 64 \
   --drop_relative
