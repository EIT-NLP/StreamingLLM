from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import os.path as osp
from tqdm import tqdm
from datetime import datetime
import logging
import time
from collections import defaultdict
import argparse
from moviepy.editor import VideoFileClip
import math
import sys
import decord

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
sys.path.append(osp.dirname(osp.abspath(osp.join(osp.dirname(__file__), '..'))))
from qwen2_5_vl import Qwen2_5_VLForConditionaIncrementalGeneration


# Parameters
RUN_NAME = "incremental_ovobench"
DROP_METHOD = "chunk-intra"
DROP_THRESHOLD = 0.25
DROP_ABSOLUTE = True
CKPT_PATH = "Qwen2.5-VL-3B-Instruct"

TASK_JSON = "datasets/OVO-Bench/ovo_bench_new.json"
VIDEO_DIR = "datasets/OVO-Bench/src_videos"
RESULT_DIR = "outputs/eval/ovobench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"

MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 4
MAX_FRAMES = 720
FPS = 1

# spatial token drop
SPATIAL_KEEP_RATIO = 1.0
VISION_POOL_TYPE = "none"

# chunk-level temporal token drop
CHUNK_SIZE = 8
NEAREST_WINDOW_SIZE = 0
ATTEND_CHUNK_NUM = 1
INPUT_STEP = 16
USER_QUERY_RETRIEVAL = 16
MAX_NEW_TOKENS = 128

backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def build_prompt(task, question, options, _anno_, index):
    if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        formatted_options = "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options)) + ";"
        prompt = f"""
            Question: {question}
            Options:
            {formatted_options}
            Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
            Do not include any additional text or explanation in your response.
        """
    elif task == "REC":
        activity = _anno_["activity"]
        question = "How many times did they " + activity + "?"
        prompt = f""" 
            You're watching a video in which people may perform a certain type of action repetively. 
            The person performing this kind of action are referred to as 'they' in the following statement.
            You're task is to count how many times have different people in the video perform this kind of action in total.
            One complete motion counts as one. 
            Now, answer the following question: {question}
            Provide your answer as a single number (e.g., 0, 1, 2, 3...) indicating the total count.
            Do not include any additional text or explanation in your response.
        """
    elif task == "SSR":
        step = _anno_["test_info"][index]["step"]
        prompt = f"""
            You're watching a tutorial video which contain a sequential of steps. 
            The following is one step from the whole procedures: 
            {step}
            Your task is to determine if the man or woman in the video is currently performing this step.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    elif task == "CRR":
        question = _anno_["question"]
        prompt = f"""
            You're responsible of answering questions based on the video content. 
            The following question are relevant to the latest frames, i.e. the end of the video.
            {question}
            Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    else:
        raise ValueError(f"Unsupported task: {task}")
    return prompt


def chunk_video(video_path, end_time, start_time=0):
    end_time = math.ceil(end_time)
    video_name = osp.splitext(osp.basename(video_path))[0]
    output_dir = osp.join(VIDEO_DIR, "chunked")
    os.makedirs(output_dir, exist_ok=True)
    output_file = osp.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if osp.exists(output_file):
        logger.debug(f"Chunked video {output_file} already exists")
        return output_file

    tmp_dir = osp.join(VIDEO_DIR, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    video = VideoFileClip(video_path)
    if end_time > video.duration:
        end_time = video.duration
    clip = video.subclip(start_time, end_time)
    temp_audiofile = osp.join(tmp_dir, f"temp_audio_{video_name}_{start_time}_{end_time}.mp3")
    clip.write_videofile(output_file, logger=None, temp_audiofile=temp_audiofile)
    logger.debug(f"Chunked video {output_file} saved")
    return output_file


def get_nframes(video_path):
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    min_frames = ceil_by_factor(MIN_FRAMES, factor=2)
    max_frames = floor_by_factor(MAX_FRAMES, factor=2)
    nframes = int(total_frames // video_fps * FPS)
    nframes = min(max(nframes, min_frames), max_frames)
    if nframes % 2 == 1 and (nframes - 1) % 4 != 0:
        nframes -= 1
    return max(nframes, min_frames)


def get_past_length(past):
    if past is None:
        return 0
    return past[0][0].shape[2]


def get_response_incremental(prompt, video_path, model, processor):
    nframes = get_nframes(video_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "min_frames": MIN_FRAMES,
                    "max_frames": MAX_FRAMES,
                    "nframes": nframes,
                },
                {"type": "text", "text": ""},
            ],
        }
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    del image_inputs
    video_inputs = video_inputs[0]

    prefix_text = "<|im_start|>user\n"
    vision_only_text = "<|vision_start|><|video_pad|><|vision_end|>"
    end_text = f"{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs0 = processor(
        text=[prefix_text],
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
    past_len = get_past_length(past_key_values)

    is_first_frame = True
    for start in range(0, video_inputs.shape[0], INPUT_STEP):
        end = min(start + INPUT_STEP, video_inputs.shape[0])
        chunk = video_inputs[start:end]
        chunk_inputs = processor(
            text=[vision_only_text],
            images=None,
            videos=chunk,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

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
        past_len = get_past_length(past_key_values)

        del chunk_inputs

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
        max_new_tokens=MAX_NEW_TOKENS,
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
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = output_text[0]

    del generated_ids, end_inputs, past_key_values, inputs0
    torch.cuda.empty_cache()
    return response


def score(results):
    def calculate_score_backward_realtime(items):
        def get_score(response, gt):
            if response is None:
                return 0
            return int(gt in response)

        scores = {}
        for item in items:
            item["score"] = get_score(item["response"], item["ground_truth"])
            scores.setdefault(item["task"], []).append(item["score"])
        return items, scores

    def calculate_score_forward(items):
        def get_score_rec(response, gt):
            if response is None:
                return 0
            import re
            response = "".join(re.findall(r"\d+", response))
            return response == str(gt)

        def get_score_yes_no(response, gt):
            if response is None:
                return 0
            return int(gt in response)

        scores = {task: [] for task in set(item["task"] for item in items)}
        for item in items:
            if item["task"] == "REC":
                cnt_correct = 0
                for test_info in item["test_info"]:
                    cnt_correct += get_score_rec(test_info["response"], test_info["count"])
                scores["REC"].append(cnt_correct / len(item["test_info"]))
            elif item["task"] == "SSR":
                cnt_correct = 0
                for test_info in item["test_info"]:
                    if (test_info["response"] == "N" and test_info["type"] == 0) or (
                        test_info["response"] == "Y" and test_info["type"] == 1
                    ):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info["type"] == 0 else "Yes"
                    cnt_correct += get_score_yes_no(test_info["response"], gt)
                scores["SSR"].append(cnt_correct / len(item["test_info"]))
            elif item["task"] == "CRR":
                cnt_correct = 0
                for test_info in item["test_info"]:
                    if (test_info["response"] == "N" and test_info["type"] == 0) or (
                        test_info["response"] == "Y" and test_info["type"] == 1
                    ):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info["type"] == 0 else "Yes"
                    cnt_correct += get_score_yes_no(test_info["response"], gt)
                scores["CRR"].append(cnt_correct / len(item["test_info"]))
        return items, scores

    avg_scores = {"backward": [], "realtime": [], "forward": []}

    if results["backward"]:
        _, backward_scores = calculate_score_backward_realtime(results["backward"])
        for task, values in backward_scores.items():
            logger.info(f"Task: {task}, Acc: {100 * sum(values) / len(values):.2f}")
            avg_scores["backward"].append(sum(values) / len(values))
        logger.info(f"Backward Avg.: {100 * sum(avg_scores['backward']) / len(avg_scores['backward']):.2f}\n")

    if results["realtime"]:
        _, realtime_scores = calculate_score_backward_realtime(results["realtime"])
        for task, values in realtime_scores.items():
            logger.info(f"Task: {task}, Acc: {100 * sum(values) / len(values):.2f}")
            avg_scores["realtime"].append(sum(values) / len(values))
        logger.info(f"Realtime Avg.: {100 * sum(avg_scores['realtime']) / len(avg_scores['realtime']):.2f}\n")

    if results["forward"]:
        _, forward_scores = calculate_score_forward(results["forward"])
        for task, values in forward_scores.items():
            logger.info(f"Task: {task}, Acc: {100 * sum(values) / len(values):.2f}")
            avg_scores["forward"].append(sum(values) / len(values))
        logger.info(f"Forward Avg.: {100 * sum(avg_scores['forward']) / len(avg_scores['forward']):.2f}\n")


def log_drop_stats():
    if DROP_METHOD is None or DR_SAVE_PATH is None or not osp.exists(DR_SAVE_PATH):
        return

    s_drop_list, s_total_list, s_ratio_list = [], [], []
    t_drop_list, t_total_list, t_ratio_list = [], [], []
    st_drop_list, st_total_list, st_ratio_list = [], [], []

    with open(DR_SAVE_PATH, "r") as f:
        for line in f:
            drop_ratio_info = json.loads(line)
            if "s-drop" not in drop_ratio_info:
                continue
            s_drop_list.append(drop_ratio_info["s-drop"])
            s_total_list.append(drop_ratio_info["s-total"])
            s_ratio_list.append(drop_ratio_info["s-ratio"])
            t_drop_list.append(drop_ratio_info["t-drop"])
            t_total_list.append(drop_ratio_info["t-total"])
            t_ratio_list.append(drop_ratio_info["t-ratio"])
            st_drop_list.append(drop_ratio_info["s-t-drop"])
            st_total_list.append(drop_ratio_info["s-t-total"])
            st_ratio_list.append(drop_ratio_info["s-t-ratio"])

    if not st_total_list:
        return

    s_total_dr = sum(s_drop_list) / max(sum(s_total_list), 1)
    t_total_dr = sum(t_drop_list) / max(sum(t_total_list), 1)
    st_total_dr = sum(st_drop_list) / max(sum(st_total_list), 1)
    s_avg_dr = sum(s_ratio_list) / max(len(s_ratio_list), 1)
    t_avg_dr = sum(t_ratio_list) / max(len(t_ratio_list), 1)
    st_avg_dr = sum(st_ratio_list) / max(len(st_ratio_list), 1)

    logger.info(f"[Spatial]   Total drop ratio (weighted): {100 * s_total_dr:.1f}% | Average (unweighted): {100 * s_avg_dr:.1f}%")
    logger.info(f"[Temporal] Total drop ratio (weighted): {100 * t_total_dr:.1f}% | Average (unweighted): {100 * t_avg_dr:.1f}%")
    logger.info(f"[S+T]      Total drop ratio (weighted): {100 * st_total_dr:.1f}% | Average (unweighted): {100 * st_avg_dr:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--spatial_keep_ratio", type=float, default=SPATIAL_KEEP_RATIO)
    parser.add_argument("--vision_pool_type", type=str, default=VISION_POOL_TYPE)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--nearest_window_size", type=int, default=NEAREST_WINDOW_SIZE)
    parser.add_argument("--attend_chunk_num", type=int, default=ATTEND_CHUNK_NUM)
    parser.add_argument("--input_step", type=int, default=INPUT_STEP)
    parser.add_argument("--user_query_retrieval", type=int, default=USER_QUERY_RETRIEVAL)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--skip_ids_file", type=str, default=None)
    args = parser.parse_args()

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    DROP_METHOD = args.drop_method
    DROP_THRESHOLD = args.drop_threshold
    DROP_ABSOLUTE = not args.drop_relative
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_JSON = args.task_json
    VIDEO_DIR = args.video_dir
    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    DR_SAVE_PATH = osp.join(RESULT_DIR, DR_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    FPS = args.fps
    SPATIAL_KEEP_RATIO = args.spatial_keep_ratio
    VISION_POOL_TYPE = args.vision_pool_type
    CHUNK_SIZE = args.chunk_size
    NEAREST_WINDOW_SIZE = args.nearest_window_size
    ATTEND_CHUNK_NUM = args.attend_chunk_num
    INPUT_STEP = args.input_step
    USER_QUERY_RETRIEVAL = args.user_query_retrieval
    MAX_NEW_TOKENS = args.max_new_tokens

    assert INPUT_STEP / CHUNK_SIZE == 2, (
        f"Error input_step / chunk_size = ({INPUT_STEP} / {CHUNK_SIZE}), which should be 2!"
    )

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, "output"), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, "drop"), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, "log"), exist_ok=True)

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info(f"Running {RUN_NAME} on OVO-Bench")
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
    logger.info(f"FPS: {FPS}")
    logger.info(f"Spatial keep ratio: {SPATIAL_KEEP_RATIO}")
    logger.info(f"Chunk size: {CHUNK_SIZE}")
    logger.info(f"Vision pool type: {VISION_POOL_TYPE}")
    logger.info(f"Nearest window size: {NEAREST_WINDOW_SIZE}")
    logger.info(f"Attend chunk num: {ATTEND_CHUNK_NUM}")
    logger.info(f"Input step: {INPUT_STEP}")
    logger.info(f"User query retrieval: {USER_QUERY_RETRIEVAL}")
    logger.info(f"Max new tokens: {MAX_NEW_TOKENS}")

    torch.manual_seed(1234)
    logger.info("Set manual seed to 1234")
    model = Qwen2_5_VLForConditionaIncrementalGeneration.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info(f"Load model and processor from {CKPT_PATH}")

    with open(TASK_JSON, "r") as f:
        task_list = json.load(f)

    start_time = time.time()
    skip_ids = set()
    if args.skip_ids_file and osp.exists(args.skip_ids_file):
        with open(args.skip_ids_file) as f:
            skip_ids = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(skip_ids)} skip IDs from {args.skip_ids_file}")
    for item in tqdm(task_list):
        if str(item.get("id", "")) in skip_ids:
            continue
        with torch.no_grad():
            try:
                if item["task"] in backward_tasks or item["task"] in realtime_tasks:
                    item_id, video, task, question, options, realtime, gt = (
                        item["id"],
                        item["video"],
                        item["task"],
                        item["question"],
                        item["options"],
                        item["realtime"],
                        item["gt"],
                    )
                    prompt = build_prompt(task, question, options, None, None)
                    video_path = osp.join(VIDEO_DIR, video)
                    chunk_video_path = chunk_video(video_path=video_path, end_time=realtime)
                    response = get_response_incremental(prompt, chunk_video_path, model, processor)
                    output_dict = {
                        "id": item_id,
                        "video": video,
                        "task": task,
                        "question": question,
                        "response": response,
                        "ground_truth": chr(65 + gt),
                    }
                elif item["task"] in forward_tasks:
                    video = item["video"]
                    task = item["task"]
                    video_path = osp.join(VIDEO_DIR, video)
                    for i, test_info in enumerate(item["test_info"]):
                        prompt = build_prompt(task, None, None, item, i)
                        chunk_video_path = chunk_video(video_path=video_path, end_time=test_info["realtime"])
                        item["test_info"][i]["response"] = get_response_incremental(
                            prompt,
                            chunk_video_path,
                            model,
                            processor,
                        )
                    output_dict = item
                else:
                    logger.warning(f"Skip unsupported task item: {item}")
                    continue

                with open(OUTPUT_JSONL, "a" if osp.exists(OUTPUT_JSONL) else "w") as f:
                    f.write(json.dumps(output_dict) + "\n")
            except Exception as e:
                logger.error(f"Error in processing {item}: {e}")

    cost_time = int(time.time() - start_time)

    results = defaultdict(list)
    if osp.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r") as f:
            lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            if item["task"] in backward_tasks:
                results["backward"].append(item)
            elif item["task"] in realtime_tasks:
                results["realtime"].append(item)
            elif item["task"] in forward_tasks:
                results["forward"].append(item)
        score(results)
    else:
        logger.info("No output file generated")

    log_drop_stats()
    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
