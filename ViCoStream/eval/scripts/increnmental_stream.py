import argparse
import json
import logging
import math
import os
import os.path as osp
import sys
import time
from collections import Counter
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from qwen2_5_vl import Qwen2_5_VLForConditionaIncrementalGeneration

from evaluate_open_ended_qa import evaluate_open_stream


RUN_NAME = "streambench_v0_3_incremental"
DROP_METHOD = "chunk-intra"
DROP_THRESHOLD = 0.1
DROP_ABSOLUTE = True
SPATIAL_KEEP_RATIO = 1.0
VISION_POOL_TYPE = "none"
CHUNK_SIZE = 16
NEAREST_WINDOW_SIZE = 0
ATTEND_CHUNK_NUM = 4
INPUT_STEP = 32
USER_QUERY_RETRIEVAL = 16

CKPT_PATH = "Qwen2.5-VL-3B-Instruct"
DATA_ROOT = "datasets/StreamBench_v0.3"
TASK_JSON = osp.join(DATA_ROOT, "streaming_bench_v0.3_incremental_qa.json")
SOURCE_JSON = osp.join(DATA_ROOT, "streaming_bench_v0.3.json")
RESULT_ROOT = "outputs/eval/streambench_v0_3"

LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "pred/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"

MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 4
MAX_FRAMES = 1016
DECODE_FPS = 1.0

EVAL_API_KEY = os.getenv("STREAMBENCH_EVAL_API_KEY", os.getenv("VSTREAM_EVAL_API_KEY", "sk-..."))
EVAL_API_BASE = os.getenv("STREAMBENCH_EVAL_API_BASE", os.getenv("VSTREAM_EVAL_API_BASE", "your_base_url"))
EVAL_MODEL = os.getenv("STREAMBENCH_EVAL_MODEL", os.getenv("VSTREAM_EVAL_MODEL", "gpt-3.5-turbo"))
EVAL_NUM_TASKS = 1
EVAL_SLEEP_PER_CALL = 0.5

prompt = """You are an advanced streaming video understanding AI assistant. You are given frames decoded from the beginning of a video up to the current timestamp and a question about the video.
Answer the question in a short, factual sentence based on the visual evidence and useful commonsense when needed.
Question type: {}

Question: {}

Answer:"""

BAD_VIDEOS = {
    "YL5Vhd-WbB8.mp4",
    "YxXrLCWFxTg.mp4",
    "_tOrWlrdkIA.mp4",
    "vwNhd4q--RQ.mp4",
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)7s | %(message)s")


def resolve_video_path(data_root, info):
    video_path = info.get("video_path", "")
    class_1 = info.get("class_1", "")
    candidates = []
    if osp.isabs(video_path):
        candidates.append(video_path)
    candidates.extend([
        osp.join(data_root, class_1, video_path),
        osp.join(data_root, class_1, class_1, video_path),
        osp.join(data_root, "Movie", "Movie", video_path),
        osp.join(data_root, "WebVideo", video_path),
        osp.join(data_root, "Ego", video_path),
    ])
    for path in candidates:
        if osp.exists(path) and not osp.basename(path).startswith("._"):
            return path
    raise FileNotFoundError(f"Cannot resolve video path: {video_path}")


def normalize_streambench_json(source_json, output_json, data_root):
    with open(source_json, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    normalized = []
    for video_idx, item in enumerate(raw_items):
        info = item.get("info", {})
        video_path = resolve_video_path(data_root, info)
        video_name = info.get("video_name") or osp.splitext(osp.basename(info.get("video_path", "")))[0]
        for bp_idx, bp in enumerate(item.get("breakpoint", [])):
            normalized.append({
                "id": f"{video_idx:06d}_{bp_idx:02d}",
                "video_id": video_name,
                "video_path": video_path,
                "source_video_path": info.get("video_path", ""),
                "question": bp.get("question", ""),
                "answer": bp.get("answer", ""),
                "answer_type": bp.get("class", ""),
                "time": bp.get("time"),
                "video_index": video_idx,
                "breakpoint_index": bp_idx,
                "class_1": info.get("class_1", ""),
                "class_2": info.get("class_2", ""),
            })

    output_parent = osp.dirname(output_json)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return normalized


def sample_indices(total_frames, native_fps, end_time, decode_fps, max_frames):
    end_time = max(float(end_time), 0.0)
    end_frame = min(total_frames, max(1, int(math.ceil(end_time * native_fps))))
    step = max(native_fps / decode_fps, 1.0)
    indices = torch.arange(0, end_frame, step).round().long().unique()
    indices = indices[(indices >= 0) & (indices < end_frame)]
    if indices.numel() == 0:
        indices = torch.tensor([0], dtype=torch.long)
    if max_frames and indices.numel() > max_frames:
        pick = torch.linspace(0, indices.numel() - 1, max_frames).round().long()
        indices = indices[pick].unique()
    return indices.tolist()


def decode_video_prefix_1fps(video_path, end_time, decode_fps=1.0, min_frames=4, max_frames=1016):
    try:
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        native_fps = vr.get_avg_fps() or decode_fps
        indices = sample_indices(len(vr), native_fps, end_time, decode_fps, max_frames)
        video = torch.from_numpy(vr.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
    except Exception:
        from torchvision.io import read_video
        video, _, info = read_video(video_path, start_pts=0, end_pts=float(end_time), pts_unit="sec")
        if video.numel() == 0:
            raise ValueError(f"No frames decoded from {video_path} before {end_time}s")
        native_fps = info.get("video_fps", decode_fps) or decode_fps
        indices = sample_indices(video.shape[0], native_fps, end_time, decode_fps, max_frames)
        video = video[indices].permute(0, 3, 1, 2)

    if video.shape[0] < min_frames:
        repeat = min_frames - video.shape[0]
        video = torch.cat([video, video[-1:].repeat(repeat, 1, 1, 1)], dim=0)
    return video


def get_video_num_frames(video):
    return video.shape[0] if hasattr(video, "shape") else len(video)


def run_incremental_generate(model, processor, video_inputs, question, answer_type, args, dr_save_path):
    system_text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
    vision_only_text = "<|vision_start|><|video_pad|><|vision_end|>"
    end_text = (
        "<|im_start|>user\n"
        + prompt.format(answer_type, question)
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

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
        drop_method=args.drop_method,
        drop_threshold=args.drop_threshold,
        drop_absolute=not args.drop_relative,
        dr_save_path=dr_save_path,
        spatial_keep_ratio=args.spatial_keep_ratio,
        vision_pool_type=args.vision_pool_type,
        chunk_size=args.chunk_size,
        attend_chunk_num=args.attend_chunk_num,
    )
    past_key_values = out.past_key_values

    is_first_frame = True
    num_frames = get_video_num_frames(video_inputs)
    for start in range(0, num_frames, args.input_step):
        end = min(start + args.input_step, num_frames)
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
            drop_method=args.drop_method,
            drop_threshold=args.drop_threshold,
            drop_absolute=not args.drop_relative,
            dr_save_path=dr_save_path,
            spatial_keep_ratio=args.spatial_keep_ratio,
            vision_pool_type=args.vision_pool_type,
            chunk_size=args.chunk_size,
            nearest_window_size=args.nearest_window_size,
            is_first_frame=is_first_frame,
            attend_chunk_num=args.attend_chunk_num,
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
        max_new_tokens=args.max_new_tokens,
        drop_method=args.drop_method,
        drop_threshold=args.drop_threshold,
        drop_absolute=not args.drop_relative,
        dr_save_path=dr_save_path,
        spatial_keep_ratio=args.spatial_keep_ratio,
        vision_pool_type=args.vision_pool_type,
        chunk_size=args.chunk_size,
        nearest_window_size=args.nearest_window_size,
        is_user_query=True,
        attend_chunk_num=args.attend_chunk_num,
        user_query_retrieval=args.user_query_retrieval,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(end_inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del generated_ids
    del end_inputs
    del past_key_values
    torch.cuda.empty_cache()
    return response


def summarize_results(results, task_types, pred_path, eval_json):
    task_metrics = {t: {"correct": 0, "total": 0, "score_sum": 0} for t in task_types}
    overall = {"correct": 0, "total": 0, "score_sum": 0}

    for value in results.values():
        if len(value) < 2 or "a_type" not in value[1]:
            continue
        pred_item, qa_item = value[0], value[1]
        task = qa_item["a_type"]
        if task not in task_metrics:
            task_metrics[task] = {"correct": 0, "total": 0, "score_sum": 0}
        score = pred_item.get("score", 0)
        correct = str(pred_item.get("pred", "")).strip().lower() == "yes"
        task_metrics[task]["total"] += 1
        task_metrics[task]["score_sum"] += score
        task_metrics[task]["correct"] += int(correct)
        overall["total"] += 1
        overall["score_sum"] += score
        overall["correct"] += int(correct)

    summary = {"pred_path": pred_path, "eval_json": eval_json, "details": {}, "overall": {}}
    for task, metrics in task_metrics.items():
        if metrics["total"]:
            summary["details"][task] = {
                "accuracy": metrics["correct"] / metrics["total"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "score": metrics["score_sum"] / metrics["total"],
            }
        else:
            summary["details"][task] = {"accuracy": None, "correct": 0, "total": 0, "score": None}
    summary["overall"] = {
        "accuracy": overall["correct"] / overall["total"] if overall["total"] else None,
        "correct": overall["correct"],
        "total": overall["total"],
        "score": overall["score_sum"] / overall["total"] if overall["total"] else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--source_json", type=str, default=SOURCE_JSON)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--normalize_only", action="store_true")
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--decode_fps", type=float, default=DECODE_FPS)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")
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
    parser.add_argument("--skip_ids_file", type=str, default=None)
    args = parser.parse_args()

    if args.input_step != args.chunk_size * 2:
        raise ValueError(f"input_step / chunk_size should be 2, got {args.input_step}/{args.chunk_size}")

    if args.normalize_only or not osp.exists(args.task_json):
        items = normalize_streambench_json(args.source_json, args.task_json, args.data_root)
        print(f"Normalized {len(items)} QA items to {args.task_json}")
        if args.normalize_only:
            return

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.result_dir or osp.join(RESULT_ROOT, args.run_name)
    log_path = osp.join(result_dir, LOG_PATH.format(run_name=args.run_name, curr_time=curr_time))
    output_jsonl = osp.join(result_dir, OUTPUT_JSONL.format(run_name=args.run_name, curr_time=curr_time))
    dr_save_path = osp.join(result_dir, DR_SAVE_PATH.format(run_name=args.run_name, curr_time=curr_time))
    eval_tmp_dir = osp.join(result_dir, "tmp", f"{args.run_name}_{curr_time}")
    eval_json = osp.join(result_dir, "eval", f"{args.run_name}_{curr_time}.json")
    eval_summary_json = osp.join(result_dir, "eval", f"{args.run_name}_{curr_time}_summary.json")

    for subdir in ["pred", "drop", "log", "eval", "tmp"]:
        os.makedirs(osp.join(result_dir, subdir), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info(f"Running {args.run_name} on StreamBench v0.3")
    logger.info(f"Task json: {args.task_json}")
    logger.info(f"Result dir: {result_dir}")
    logger.info(f"Output jsonl: {output_jsonl}")
    logger.info(f"Eval model: {args.eval_model}")
    logger.info(f"Decode fps: {args.decode_fps}")
    logger.info(f"Drop method: {args.drop_method}")
    logger.info(f"Chunk size: {args.chunk_size}, input step: {args.input_step}")

    with open(args.task_json, "r", encoding="utf-8") as f:
        task_items = json.load(f)
    if args.limit is not None:
        task_items = task_items[:args.limit]
        logger.info(f"Limit enabled: {len(task_items)} samples")

    torch.manual_seed(1234)
    model = Qwen2_5_VLForConditionaIncrementalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    skip_ids = set()
    if args.skip_ids_file and osp.exists(args.skip_ids_file):
        with open(args.skip_ids_file) as f:
            skip_ids = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(skip_ids)} skip IDs from {args.skip_ids_file}")
    start_time = time.time()
    for task in tqdm(task_items, desc="Inference", total=len(task_items)):
        if str(task.get("question_id", task.get("id", ""))) in skip_ids:
            continue
        video_name = os.path.basename(task.get("video_path", ""))
        if video_name in BAD_VIDEOS:
            continue
        try:
            with torch.no_grad():
                video_inputs = decode_video_prefix_1fps(
                    task["video_path"],
                    task["time"],
                    decode_fps=args.decode_fps,
                    min_frames=args.min_frames,
                    max_frames=args.max_frames,
                )
                response = run_incremental_generate(
                    model=model,
                    processor=processor,
                    video_inputs=video_inputs,
                    question=task["question"],
                    answer_type=task.get("answer_type", ""),
                    args=args,
                    dr_save_path=dr_save_path,
                )
                output = {
                    "id": task["id"],
                    "question": task["question"],
                    "answer": task["answer"],
                    "answer_type": task.get("answer_type", ""),
                    "response": response,
                    "video_id": task.get("video_id", ""),
                    "video_path": task.get("video_path", ""),
                    "time": task.get("time"),
                    "class_1": task.get("class_1", ""),
                    "class_2": task.get("class_2", ""),
                }
                with open(output_jsonl, "a" if osp.exists(output_jsonl) else "w", encoding="utf-8") as f:
                    f.write(json.dumps(output, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.exception(f"Error in processing question id {task.get('id')}: {exc}")

    cost_time = int(time.time() - start_time)

    results = {}
    if not args.skip_eval and osp.exists(output_jsonl) and osp.getsize(output_jsonl) > 0:
        logger.info(f"Evaluate generated answers from {output_jsonl}")
        results = evaluate_open_stream(
            pred_path=output_jsonl,
            output_dir=eval_tmp_dir,
            output_json=eval_json,
            num_tasks=args.eval_num_tasks,
            num_chunks=1,
            api_key=args.eval_api_key,
            api_base=args.eval_api_base,
            model=args.eval_model,
            sleep_per_call=args.eval_sleep_per_call,
        )
    elif args.skip_eval:
        logger.info("Skip GPT evaluation")
    else:
        logger.info("No prediction output found; skip GPT evaluation")

    task_types = sorted(Counter(item.get("answer_type", "") for item in task_items))
    summary = summarize_results(results, task_types, output_jsonl, eval_json)
    with open(eval_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for task_type, metrics in summary["details"].items():
        if metrics["total"]:
            logger.info(
                f"- {task_type}: acc={100 * metrics['accuracy']:.2f}% "
                f"({metrics['correct']}/{metrics['total']}), score={metrics['score']:.3f}"
            )
        else:
            logger.info(f"- {task_type}: No question processed")
    overall = summary["overall"]
    if overall["total"]:
        logger.info(
            f"Overall: acc={100 * overall['accuracy']:.2f}% "
            f"({overall['correct']}/{overall['total']}), score={overall['score']:.3f}"
        )
    else:
        logger.info("Overall: No question processed")
    logger.info(f"Eval summary json: {eval_summary_json}")
    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")


if __name__ == "__main__":
    main()
