from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import os.path as osp
from tqdm import tqdm
from datetime import datetime
import re
import logging
import time
import argparse
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from qwen2_5_vl import Qwen2_5_VLForConditionaIncrementalGeneration

from evaluate_open_ended_qa import evaluate_open_stream

# Parameters
RUN_NAME = "timechat-baseline-sim0_5-vstream_ego4d"  # timechat-baseline-sim0_7, timechat-baseline-simfull
DROP_METHOD = 'chunk-intra'  # 'feature', 'chunk-intra', None
DROP_THRESHOLD = 0.1
DROP_ABSOLUTE = True
SPATIAL_KEEP_RATIO = 1.0
VISION_POOL_TYPE = 'none'
## chunk-level temporal token drop
CHUNK_SIZE = 16
NEAREST_WINDOW_SIZE = 0
ATTEND_CHUNK_NUM = 4
INPUT_STEP = 32
USER_QUERY_RETRIEVAL = 16
EVAL_API_KEY = os.getenv("VSTREAM_EVAL_API_KEY", "sk-...")
EVAL_API_BASE = os.getenv("VSTREAM_EVAL_API_BASE", "your_base_url")
EVAL_MODEL = os.getenv("VSTREAM_EVAL_MODEL", "gpt-3.5-turbo")
EVAL_NUM_TASKS = 1
EVAL_SLEEP_PER_CALL = 0.5

CKPT_PATH = "Qwen2.5-VL-3B-Instruct"

TASK_JSON = "datasets/VStream-QA/vstream-realtime/temp_qa.json"

VIDEO_DIR = "datasets/VStream-QA/vstream-realtime/ego4d_frames"
RESULT_ROOT = "outputs/eval/vstream-ego4d"
RESULT_DIR = osp.join(RESULT_ROOT, RUN_NAME)
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "pred/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"

SAVE_DROP = True
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 4
MAX_FRAMES = 1016


prompt = """You are an advanced video understanding AI assistant. You are given several frames from a video and a question about the video. 
Your goal is to answer the question in a short, factual sentence based on visual evidence. 
Answer style: {}

Question: {}

Answer:"""

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)


def get_video_num_frames(video):
    if hasattr(video, "shape"):
        return video.shape[0]
    return len(video)


def frame_sort_key(path):
    name = osp.splitext(osp.basename(path))[0]
    if name.isdigit():
        return int(name)
    match = re.search(r"shot_(\d+)_img_(\d+)", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return name


def select_frame_paths(video_path, start_frame, end_frame):
    frame_paths = sorted(
        [osp.join(video_path, fn) for fn in os.listdir(video_path) if fn.endswith('.jpg')],
        key=frame_sort_key,
    )
    if isinstance(start_frame, int) and isinstance(end_frame, int):
        return frame_paths[start_frame:end_frame + 1]

    basename_to_idx = {osp.basename(path): idx for idx, path in enumerate(frame_paths)}
    if start_frame not in basename_to_idx or end_frame not in basename_to_idx:
        raise ValueError(
            f"Frame range not found in {video_path}: start={start_frame}, end={end_frame}"
        )
    start_idx = basename_to_idx[start_frame]
    end_idx = basename_to_idx[end_frame]
    if start_idx > end_idx:
        raise ValueError(
            f"Invalid frame range in {video_path}: start index {start_idx} > end index {end_idx}"
        )
    return frame_paths[start_idx:end_idx + 1]


### Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")  # Default is absolute
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--spatial_keep_ratio", type=float, default=SPATIAL_KEEP_RATIO)
    parser.add_argument("--vision_pool_type", type=str, default=VISION_POOL_TYPE)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--nearest_window_size", type=int, default=NEAREST_WINDOW_SIZE)
    parser.add_argument("--attend_chunk_num", type=int, default=ATTEND_CHUNK_NUM)
    parser.add_argument("--input_step", type=int, default=INPUT_STEP)
    parser.add_argument("--user_query_retrieval", type=int, default=USER_QUERY_RETRIEVAL)
    parser.add_argument("--eval_api_key", type=str, default=EVAL_API_KEY)
    parser.add_argument("--eval_api_base", type=str, default=EVAL_API_BASE)
    parser.add_argument("--eval_model", type=str, default=EVAL_MODEL)
    parser.add_argument("--eval_num_tasks", type=int, default=EVAL_NUM_TASKS)
    parser.add_argument("--eval_sleep_per_call", type=float, default=EVAL_SLEEP_PER_CALL)

    args = parser.parse_args()

    # Update global variables
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    DROP_METHOD = args.drop_method
    DROP_THRESHOLD = args.drop_threshold
    DROP_ABSOLUTE = not args.drop_relative
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir or osp.join(RESULT_ROOT, RUN_NAME)
    TASK_JSON = args.task_json
    VIDEO_DIR = args.video_dir
    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    DR_SAVE_PATH = osp.join(RESULT_DIR, DR_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    SPATIAL_KEEP_RATIO = args.spatial_keep_ratio
    VISION_POOL_TYPE = args.vision_pool_type
    NEAREST_WINDOW_SIZE = args.nearest_window_size
    ATTEND_CHUNK_NUM = args.attend_chunk_num
    INPUT_STEP = args.input_step
    USER_QUERY_RETRIEVAL = args.user_query_retrieval
    EVAL_API_KEY = args.eval_api_key
    EVAL_API_BASE = args.eval_api_base
    EVAL_MODEL = args.eval_model
    EVAL_NUM_TASKS = args.eval_num_tasks
    EVAL_SLEEP_PER_CALL = args.eval_sleep_per_call

    # chunk-level incremental token drop
    CHUNK_SIZE = args.chunk_size

    assert INPUT_STEP == CHUNK_SIZE * 2, (
        f"Error input_step / chunk_size = ({INPUT_STEP} / {CHUNK_SIZE}), which should be 2!"
    )

    # Create result directory
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'pred'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'drop'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'eval'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'tmp'), exist_ok=True)

    # Add file handler
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Print run info
    logger.info(f"Running {RUN_NAME} on VStream")
    logger.info(f"Drop method: {DROP_METHOD}")
    logger.info(f"Drop threshold: {DROP_THRESHOLD}")
    logger.info("Drop absolute" if DROP_ABSOLUTE else "Drop relative")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task json: {TASK_JSON}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Drop ratio info save path: {DR_SAVE_PATH}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    logger.info(f"Min frames: {MIN_FRAMES}")
    logger.info(f"Spatial keep ratio: {SPATIAL_KEEP_RATIO}")
    # logger.info(f"CT drop dict path: {CT_DROP_DICT_PATH}")
    logger.info(f"Chunk size: {CHUNK_SIZE}")
    logger.info(f"Vision pool type: {VISION_POOL_TYPE}")
    logger.info(f"Nearest window size: {NEAREST_WINDOW_SIZE}")
    logger.info(f"Attend chunk num: {ATTEND_CHUNK_NUM}")
    logger.info(f"Input step: {INPUT_STEP}")
    logger.info(f"User query retrieval: {USER_QUERY_RETRIEVAL}")
    logger.info(f"Eval API base: {EVAL_API_BASE}")
    logger.info(f"Eval model: {EVAL_MODEL}")
    logger.info(f"Eval num tasks: {EVAL_NUM_TASKS}")
    logger.info(f"Eval sleep per call: {EVAL_SLEEP_PER_CALL}")

    # Load model and processor
    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    ## Use Qwen2.5-VL
    model = Qwen2_5_VLForConditionaIncrementalGeneration.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info(f"Load model and processor from {CKPT_PATH}")

    # Load task info
    with open(TASK_JSON, 'r') as f:
        task_jsons = json.load(f)
        f.close()

    # Inference
    start_time = time.time()

    for task in tqdm(task_jsons, desc="Inference", total=len(task_jsons)):
        try:
            with torch.no_grad():
                video_id = task['video_id']
                question = task['question']
                question_id = task['id']
                answer = task['answer']
                answer_type = task['answer_type']
                video_name = task['video_name']
                video_start = task.get('start_time', 0)
                video_end = task.get('end_time', task.get('duration', task.get('gt_duration', 0)))
                gt_duration = task.get('gt_duration', 0)
                duration = task.get('duration', 0)
                video_path = osp.join(VIDEO_DIR, video_id)
                frame_paths = select_frame_paths(video_path, video_start, video_end)
                if not frame_paths:
                    raise ValueError(f"No frames found for {video_id} from {video_start} to {video_end}")

                this_question = {
                    'id': question_id,
                    'question': question,
                    'answer_type': answer_type,
                    'answer': answer,
                }

                vision_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": frame_paths,
                                "min_pixels": MIN_PIXELS,
                                "max_pixels": MAX_PIXELS,
                                "max_frames": MAX_FRAMES,
                                "min_frames": MIN_FRAMES,
                                "fps": 0.5,
                            },
                            {
                                "type": "text",
                                "text": "",
                            },
                        ],
                    }
                ]

                system_text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
                vision_only_text = "<|vision_start|><|video_pad|><|vision_end|>"
                end_text = (
                    "<|im_start|>user\n"
                    + prompt.format(answer_type, question)
                    + "<|im_end|>\n<|im_start|>assistant\n"
                )

                _, video_inputs = process_vision_info(vision_messages)
                if video_inputs is None or len(video_inputs) == 0:
                    raise ValueError(f"Failed to load video frames for {video_id}")
                video_inputs = video_inputs[0]

                inputs0 = processor(
                    text=[system_text],
                    images=None,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                out = model(
                    **inputs0,
                    past_key_values=None,
                    use_cache=True,
                    drop_method=DROP_METHOD,
                    drop_threshold=DROP_THRESHOLD,
                    drop_absolute=DROP_ABSOLUTE,
                    dr_save_path=DR_SAVE_PATH,
                    spatial_keep_ratio=SPATIAL_KEEP_RATIO,
                     vision_pool_type=VISION_POOL_TYPE,
                     chunk_size=CHUNK_SIZE,
                     attend_chunk_num=ATTEND_CHUNK_NUM,
                )
                past_key_values = out.past_key_values

                is_first_frame = True
                num_frames = get_video_num_frames(video_inputs)
                for start in range(0, num_frames, INPUT_STEP):
                    end = min(start + INPUT_STEP, num_frames)
                    chunk = video_inputs[start:end]
                    chunk_num_frames = get_video_num_frames(chunk)

                    vision_process_start = time.time()
                    chunk_inputs = processor(
                        text=[vision_only_text],
                        images=None,
                        videos=chunk,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")
                    vision_process_time = time.time() - vision_process_start

                    out = model(
                        **chunk_inputs,
                        past_key_values=past_key_values,
                        use_cache=True,
                        drop_method=DROP_METHOD,
                        drop_threshold=DROP_THRESHOLD,
                        drop_absolute=DROP_ABSOLUTE,
                        dr_save_path=DR_SAVE_PATH,
                        spatial_keep_ratio=SPATIAL_KEEP_RATIO,
                         vision_pool_type=VISION_POOL_TYPE,
                         chunk_size=CHUNK_SIZE,
                         nearest_window_size=NEAREST_WINDOW_SIZE,
                        is_first_frame=is_first_frame,
                        attend_chunk_num=ATTEND_CHUNK_NUM,
                    )
                    is_first_frame = False
                    past_key_values = out.past_key_values

                end_inputs = processor(
                    text=[end_text],
                    images=None,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                generated_ids = model.generate(
                    **end_inputs,
                    past_key_values=past_key_values,
                    max_new_tokens=128,
                    drop_method=DROP_METHOD,
                    drop_threshold=DROP_THRESHOLD,
                    drop_absolute=DROP_ABSOLUTE,
                    dr_save_path=DR_SAVE_PATH,
                    spatial_keep_ratio=SPATIAL_KEEP_RATIO,
                    vision_pool_type=VISION_POOL_TYPE,
                    chunk_size=CHUNK_SIZE,
                    nearest_window_size=NEAREST_WINDOW_SIZE,
                    is_user_query=True,
                    attend_chunk_num=ATTEND_CHUNK_NUM,
                    user_query_retrieval=USER_QUERY_RETRIEVAL,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(end_inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response = output_text[0]
                this_question['response'] = response

                del generated_ids
                del end_inputs
                del past_key_values
                torch.cuda.empty_cache()

                with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
                    f.write(json.dumps(this_question) + '\n')
        except Exception as e:
            logger.exception(f"Error in processing question id {task['id']}: {e}")
        # break
    end_time = time.time()
    cost_time = int(end_time - start_time)

    # evaluate results with gpt
    results = {}
    EVAL_TMP_DIR = osp.join(RESULT_DIR, 'tmp', f"{RUN_NAME}_{curr_time}")
    EVAL_JSON = osp.join(RESULT_DIR, 'eval', f"{RUN_NAME}_{curr_time}.json")
    EVAL_SUMMARY_JSON = osp.join(RESULT_DIR, 'eval', f"{RUN_NAME}_{curr_time}_summary.json")

    if osp.exists(OUTPUT_JSONL) and osp.getsize(OUTPUT_JSONL) > 0:
        logger.info(f"Evaluate generated answers from {OUTPUT_JSONL}")
        logger.info(f"Eval detail json: {EVAL_JSON}")
        results = evaluate_open_stream(pred_path=OUTPUT_JSONL, output_dir=EVAL_TMP_DIR,
                                       output_json=EVAL_JSON,
                                       num_tasks=EVAL_NUM_TASKS, num_chunks=1,
                                       api_key=EVAL_API_KEY,
                                       api_base=EVAL_API_BASE,
                                       model=EVAL_MODEL, sleep_per_call=EVAL_SLEEP_PER_CALL)
    else:
        logger.info("No prediction output found; skip GPT evaluation")

    task_types = ['Scene Summary', 'Action Caption', 'Whether Something Happened(Y/N)', 'Order Judging(Y/N)',
                  'What event order']

    task_metrics = {t: {"correct": 0, "total": 0, "score_sum": 0} for t in task_types}
    overall = {"correct": 0, "total": 0, "score_sum": 0}

    for k, v in results.items():
        if len(v) < 2 or "a_type" not in v[1]:
            continue

        pred_item, qa_item = v[0], v[1]
        task = qa_item["a_type"]

        if task not in task_metrics:
            continue

        pred, ans = qa_item["pred"].strip().lower(), qa_item["a"].strip().lower()
        judge = str(pred_item.get("pred", "")).strip().lower()
        score = pred_item.get("score", 0)

        correct = judge == "yes"

        task_metrics[task]["total"] += 1
        task_metrics[task]["score_sum"] += score
        task_metrics[task]["correct"] += int(correct)

        overall["total"] += 1
        overall["score_sum"] += score
        overall["correct"] += int(correct)

    summary = {
        "pred_path": OUTPUT_JSONL,
        "eval_json": EVAL_JSON,
        "details": {},
        "overall": {},
    }

    for t in task_metrics:
        m = task_metrics[t]
        if m["total"] > 0:
            acc = m["correct"] / m["total"]
            avg_score = m["score_sum"] / m["total"]
            summary["details"][t] = {
                "accuracy": acc,
                "correct": m["correct"],
                "total": m["total"],
                "score": avg_score,
            }
            logger.info(f"- {t}: acc={100 * acc:.2f}% ({m['correct']}/{m['total']}), score={avg_score:.3f}")
        else:
            summary["details"][t] = {
                "accuracy": None,
                "correct": 0,
                "total": 0,
                "score": None,
            }
            logger.info(f"- {t}: No question processed")

    overall_acc = overall["correct"] / overall["total"] if overall["total"] else 0
    overall_score = overall["score_sum"] / overall["total"] if overall["total"] else 0
    summary["overall"] = {
        "accuracy": overall_acc if overall["total"] else None,
        "correct": overall["correct"],
        "total": overall["total"],
        "score": overall_score if overall["total"] else None,
    }
    with open(EVAL_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Eval summary json: {EVAL_SUMMARY_JSON}")
    logger.info(f"Overall: acc={100 * overall_acc:.2f}% ({overall['correct']}/{overall['total']}), score={overall_score:.3f}")

    # Collect drop ratio info
    if DROP_METHOD is not None and DR_SAVE_PATH is not None and osp.exists(DR_SAVE_PATH):
        s_drop_list, s_total_list, s_ratio_list = [], [], []
        t_drop_list, t_total_list, t_ratio_list = [], [], []
        st_drop_list, st_total_list, st_ratio_list = [], [], []

        with open(DR_SAVE_PATH, 'r') as f:
            lines = f.readlines()
        for line in lines:
            drop_ratio_info = json.loads(line)

            s_drop_list.append(drop_ratio_info['s-drop'])
            s_total_list.append(drop_ratio_info['s-total'])
            s_ratio_list.append(drop_ratio_info['s-ratio'])

            t_drop_list.append(drop_ratio_info['t-drop'])
            t_total_list.append(drop_ratio_info['t-total'])
            t_ratio_list.append(drop_ratio_info['t-ratio'])

            st_drop_list.append(drop_ratio_info['s-t-drop'])
            st_total_list.append(drop_ratio_info['s-t-total'])
            st_ratio_list.append(drop_ratio_info['s-t-ratio'])

        s_total_dr = sum(s_drop_list) / max(sum(s_total_list), 1)
        t_total_dr = sum(t_drop_list) / max(sum(t_total_list), 1)
        st_total_dr = sum(st_drop_list) / max(sum(st_total_list), 1)

        s_avg_dr = sum(s_ratio_list) / max(len(s_ratio_list), 1)
        t_avg_dr = sum(t_ratio_list) / max(len(t_ratio_list), 1)
        st_avg_dr = sum(st_ratio_list) / max(len(st_ratio_list), 1)

        logger.info(
            f"[Spatial]   Total drop ratio (weighted): {100 * s_total_dr:.1f}% | Average (unweighted): {100 * s_avg_dr:.1f}%")
        logger.info(
            f"[Temporal] Total drop ratio (weighted): {100 * t_total_dr:.1f}% | Average (unweighted): {100 * t_avg_dr:.1f}%")
        logger.info(
            f"[S+T]      Total drop ratio (weighted): {100 * st_total_dr:.1f}% | Average (unweighted): {100 * st_avg_dr:.1f}%")

    # Print time
    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
