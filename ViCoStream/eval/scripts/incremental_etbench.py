import argparse
import json
import logging
import math
import os
import os.path as osp
import subprocess
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


RUN_NAME = "etbench_charades_sta_incremental"
DROP_METHOD = "chunk-intra"
DROP_THRESHOLD = 0.1
SPATIAL_KEEP_RATIO = 1.0
VISION_POOL_TYPE = "none"
CHUNK_SIZE = 16
NEAREST_WINDOW_SIZE = 0
ATTEND_CHUNK_NUM = 4
INPUT_STEP = 32
USER_QUERY_RETRIEVAL = 16

CKPT_PATH = "Qwen2.5-VL-3B-Instruct"
DATA_ROOT = "datasets/ETBench"
SOURCE_JSON = osp.join(DATA_ROOT, "annotations/etbench_txt_v1.0.json")
TASK_JSON = osp.join(DATA_ROOT, "annotations/txt/charades_sta_incremental.json")
RESULT_ROOT = "outputs/eval/etbench"
METRIC_SCRIPT = osp.join(DATA_ROOT, "evaluation/compute_metrics.py")

LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSON = "pred/{run_name}_{curr_time}.json"
OUTPUT_JSONL = "pred/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"

MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 4
MAX_FRAMES = 1016
DECODE_FPS = 1.0

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)7s | %(message)s")


def resolve_video_path(data_root, video):
    candidates = []
    if osp.isabs(video):
        candidates.append(video)
    candidates.extend([
        osp.join(data_root, "videos", video),
        osp.join(data_root, "videos_compressed", video),
    ])
    for path in candidates:
        if osp.exists(path) and not osp.basename(path).startswith("._"):
            return path
    raise FileNotFoundError(f"Cannot resolve ETBench video: {video}")


def normalize_etbench_json(source_json, output_json, data_root, source_filter="charades_sta"):
    with open(source_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    normalized = []
    for sample in samples:
        if source_filter and sample.get("source") != source_filter:
            continue
        item = dict(sample)
        item["video_path"] = resolve_video_path(data_root, item["video"])
        normalized.append(item)

    output_parent = osp.dirname(output_json)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return normalized


def sample_indices(total_frames, native_fps, duration, decode_fps, max_frames):
    duration = max(float(duration), 0.0)
    end_frame = min(total_frames, max(1, int(math.ceil(duration * native_fps))))
    step = max(native_fps / decode_fps, 1.0)
    indices = torch.arange(0, end_frame, step).round().long().unique()
    indices = indices[(indices >= 0) & (indices < end_frame)]
    if indices.numel() == 0:
        indices = torch.tensor([0], dtype=torch.long)
    if max_frames and indices.numel() > max_frames:
        pick = torch.linspace(0, indices.numel() - 1, max_frames).round().long()
        indices = indices[pick].unique()
    return indices.tolist()


def decode_video_1fps(video_path, duration, decode_fps=1.0, min_frames=4, max_frames=1016):
    try:
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        native_fps = vr.get_avg_fps() or decode_fps
        indices = sample_indices(len(vr), native_fps, duration, decode_fps, max_frames)
        video = torch.from_numpy(vr.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
    except Exception:
        from torchvision.io import read_video
        video, _, info = read_video(video_path, start_pts=0, end_pts=float(duration), pts_unit="sec")
        if video.numel() == 0:
            raise ValueError(f"No frames decoded from {video_path}")
        native_fps = info.get("video_fps", decode_fps) or decode_fps
        indices = sample_indices(video.shape[0], native_fps, duration, decode_fps, max_frames)
        video = video[indices].permute(0, 3, 1, 2)

    if video.shape[0] < min_frames:
        video = torch.cat([video, video[-1:].repeat(min_frames - video.shape[0], 1, 1, 1)], dim=0)
    return video


def get_video_num_frames(video):
    return video.shape[0] if hasattr(video, "shape") else len(video)


def run_incremental_generate(model, processor, video_inputs, question, args, dr_save_path):
    system_text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
    vision_only_text = "<|vision_start|><|video_pad|><|vision_end|>"
    end_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

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


def run_official_metrics(metric_script, pred_json, log_path):
    cmd = [sys.executable, metric_script, pred_json]
    with open(log_path, "a", encoding="utf-8") as log_file:
        return subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--source_json", type=str, default=SOURCE_JSON)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--source_filter", type=str, default="charades_sta")
    parser.add_argument("--normalize_only", action="store_true")
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_metrics", action="store_true")
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
    parser.add_argument("--metric_script", type=str, default=METRIC_SCRIPT)
    args = parser.parse_args()

    if args.input_step != args.chunk_size * 2:
        raise ValueError(f"input_step / chunk_size should be 2, got {args.input_step}/{args.chunk_size}")

    if args.normalize_only or not osp.exists(args.task_json):
        items = normalize_etbench_json(args.source_json, args.task_json, args.data_root, args.source_filter)
        print(f"Normalized {len(items)} ETBench items to {args.task_json}")
        if args.normalize_only:
            return

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.result_dir or osp.join(RESULT_ROOT, args.run_name)
    log_path = osp.join(result_dir, LOG_PATH.format(run_name=args.run_name, curr_time=curr_time))
    output_json = osp.join(result_dir, OUTPUT_JSON.format(run_name=args.run_name, curr_time=curr_time))
    output_jsonl = osp.join(result_dir, OUTPUT_JSONL.format(run_name=args.run_name, curr_time=curr_time))
    dr_save_path = osp.join(result_dir, DR_SAVE_PATH.format(run_name=args.run_name, curr_time=curr_time))

    for subdir in ["pred", "drop", "log"]:
        os.makedirs(osp.join(result_dir, subdir), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    with open(args.task_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.limit is not None:
        samples = samples[:args.limit]

    logger.info(f"Running {args.run_name} on ETBench")
    logger.info(f"Task json: {args.task_json}")
    logger.info(f"Samples: {len(samples)}")
    logger.info(f"Task/source counts: {Counter((s.get('task'), s.get('source')) for s in samples)}")
    logger.info(f"Output json: {output_json}")
    logger.info(f"Checkpoint path: {args.ckpt_path}")
    logger.info(f"Result dir: {result_dir}")
    logger.info(f"Drop method: {args.drop_method}")
    logger.info(f"Drop threshold: {args.drop_threshold}")
    logger.info("Drop absolute" if not args.drop_relative else "Drop relative")
    logger.info(f"Decode fps: {args.decode_fps}")
    logger.info(f"Min pixels: {args.min_pixels}")
    logger.info(f"Max pixels: {args.max_pixels}")
    logger.info(f"Max frames: {args.max_frames}")
    logger.info(f"Min frames: {args.min_frames}")
    logger.info(f"Spatial keep ratio: {args.spatial_keep_ratio}")
    logger.info(f"Vision pool type: {args.vision_pool_type}")

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

    predictions = []
    start_time = time.time()
    for sample in tqdm(samples, desc="Inference", total=len(samples)):
        try:
            with torch.no_grad():
                video_path = sample.get("video_path") or resolve_video_path(args.data_root, sample["video"])
                video_inputs = decode_video_1fps(
                    video_path,
                    sample.get("duration", 0),
                    decode_fps=args.decode_fps,
                    min_frames=args.min_frames,
                    max_frames=args.max_frames,
                )
                response = run_incremental_generate(
                    model=model,
                    processor=processor,
                    video_inputs=video_inputs,
                    question=sample["q"],
                    args=args,
                    dr_save_path=dr_save_path,
                )
                pred = dict(sample)
                pred["a"] = response
                pred.pop("video_path", None)
                predictions.append(pred)
                with open(output_jsonl, "a" if osp.exists(output_jsonl) else "w", encoding="utf-8") as f:
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.exception(f"Error in processing idx {sample.get('idx')}: {exc}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    cost_time = int(time.time() - start_time)
    logger.info(f"Saved {len(predictions)} predictions to {output_json}")

    if not args.skip_metrics and predictions:
        logger.info(f"Run official ETBench metrics: {args.metric_script} {output_json}")
        result = run_official_metrics(args.metric_script, output_json, log_path)
        logger.info(f"Official metrics exit code: {result.returncode}")
    elif args.skip_metrics:
        logger.info("Skip official ETBench metrics")
    else:
        logger.info("No predictions found; skip official ETBench metrics")

    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")


if __name__ == "__main__":
    main()
