from sympy import deg

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import re
import logging
import time
from collections import defaultdict
import argparse
import ffmpeg
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
sys.path.append(osp.dirname(osp.abspath(osp.join(osp.dirname(__file__), '..'))))
# from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLForConditionaIncrementalGeneration
from qwen2_5_vl import Qwen2_5_VLForConditionaIncrementalGeneration

# Parameters
RUN_NAME = "test"  # timechat-baseline-sim0_7, timechat-baseline-simfull
DROP_METHOD = 'inc-feature'  # 'feature', None
DROP_THRESHOLD = 0.1
DROP_ABSOLUTE = True
SPATIAL_KEEP_RATIO = 0.25
VISION_POOL_TYPE = 'none'
## chunk-level temporal token drop
CHUNK_SIZE = 8
NEAREST_WINDOW_SIZE = 0
ATTEND_CHUNK_NUM = 1
INPUT_STEP = 16
USER_QUERY_RETRIEVAL = 16


CKPT_PATH = "Qwen2.5-VL-3B-Instruct"
TASK_CSV = "datasets/StreamingBench/Real_Time_Visual_Understanding.csv"
VIDEO_DIR = "datasets/StreamingBench/Real-Time Visual Understanding"
RESULT_DIR = f"outputs/eval/streamingbench/{RUN_NAME}"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"
CT_DROP_DICT_PATH = None
TO_SAVE_DROP_CT = False

SAVE_DROP = True
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 4
MAX_FRAMES = 1016

# Prompt template
system_prompt = """You are an advanced video question-answering AI assistant. """

user_prompt = """You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

The best option is:"""

prompt = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

The best option is:"""

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)


# helper functions
def time_to_seconds(time_str):
    if len(time_str) == 5:
        time_obj = datetime.strptime(time_str, '%M:%S')
    else:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def split_video(video_file, start_time, end_time):
    """
    Split video into prefix part based on timestamp.
    video_file: path to video file
    start_time: start time in seconds
    end_time: end time in seconds
    """
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if os.path.exists(output_file):
        logger.debug(f"Video file {output_file} already exists.")
        return output_file
    try:
        (
            ffmpeg
            .input(video_file, ss=int(start_time))
            .output(output_file, t=(int(end_time) - int(start_time)), vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        logger.error(f"ffmpeg error: {e.stderr.decode('utf-8')}")
    logger.debug(f"Video: {output_file} splitting completed.")
    return output_file


### Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")  # Default is absolute
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--task_csv", type=str, default=TASK_CSV)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--spatial_keep_ratio", type=float, default=SPATIAL_KEEP_RATIO)
    parser.add_argument("--ct_drop_dict_path", type=str, default=CT_DROP_DICT_PATH)
    parser.add_argument("--to_save_drop_ct", type=bool, default=TO_SAVE_DROP_CT)
    parser.add_argument("--vision_pool_type", type=str, default=VISION_POOL_TYPE)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--nearest_window_size", type=int, default=NEAREST_WINDOW_SIZE)
    parser.add_argument("--attend_chunk_num", type=int, default=ATTEND_CHUNK_NUM)
    parser.add_argument("--input_step", type=int, default=INPUT_STEP)
    parser.add_argument('--user_query_retrieval', type=int, default=USER_QUERY_RETRIEVAL)
    parser.add_argument('--skip_ids_file', type=str, default=None)
    
    args = parser.parse_args()

    # Update global variables
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    DROP_METHOD = args.drop_method
    DROP_THRESHOLD = args.drop_threshold
    DROP_ABSOLUTE = not args.drop_relative
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_CSV = args.task_csv
    VIDEO_DIR = args.video_dir
    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    DR_SAVE_PATH = osp.join(RESULT_DIR, DR_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    CT_DROP_DICT_PATH = args.ct_drop_dict_path
    SPATIAL_KEEP_RATIO = args.spatial_keep_ratio
    VISION_POOL_TYPE = args.vision_pool_type
    NEAREST_WINDOW_SIZE = args.nearest_window_size
    ATTEND_CHUNK_NUM = args.attend_chunk_num
    INPUT_STEP = args.input_step
    CHUNK_SIZE = args.chunk_size
    USER_QUERY_RETRIEVAL = args.user_query_retrieval

    # chunk-level incremental token drop
    CHUNK_SIZE = args.chunk_size

    assert INPUT_STEP / CHUNK_SIZE == 2, f"Error input_step / chunk_size =  ({INPUT_STEP} / {CHUNK_SIZE}), which should be 2!"

    # Create result directory
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'output'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'drop'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)

    # Add file handler
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Print run info
    logger.info(f"Running {RUN_NAME} on StreamingBench")
    logger.info(f"Drop method: {DROP_METHOD}")
    logger.info(f"Drop threshold: {DROP_THRESHOLD}")
    logger.info("Drop absolute" if DROP_ABSOLUTE else "Drop relative")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task csv: {TASK_CSV}")
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

    # Load model and processor
    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    ## Use Qwen2.5-VL
    model = Qwen2_5_VLForConditionaIncrementalGeneration.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # attn_implementation="eager",
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info(f"Load model and processor from {CKPT_PATH}")

    # Load task info
    task_df = pd.read_csv(TASK_CSV)

    # load drop dict
    if CT_DROP_DICT_PATH is not None:
        path = os.path.join(RESULT_DIR, CT_DROP_DICT_PATH)
        if TO_SAVE_DROP_CT:
            ct_drop_dict = path
        else:
            with open(path, 'r') as f:
                ct_drop_dict = json.load(f)
                f.close()
    else:
        ct_drop_dict = None

    # Inference
    start_time = time.time()
    sample_filter = os.environ.get("TIMECHAT_SAMPLE_FILTER")
    video_repeat = int(os.environ.get("TIMECHAT_VIDEO_REPEAT", "1"))
    skip_ids = set()
    if args.skip_ids_file and osp.exists(args.skip_ids_file):
        with open(args.skip_ids_file) as f:
            skip_ids = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(skip_ids)} skip IDs from {args.skip_ids_file}")
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        if sample_filter and sample_filter not in row.question_id:
            continue
        if row.question_id in skip_ids:
            continue
        with torch.no_grad():
            try:
                question_id, task_type, question, time_stamp, answer, options, frames_required, temporal_clue_type = \
                    row.question_id, row.task_type, row.question, row.time_stamp, row.answer, row.options, row.frames_required, row.temporal_clue_type
                video_path = osp.join(VIDEO_DIR, f"sample_{question_id.split('_')[-2]}", "video.mp4")
                time_stamp_sec = time_to_seconds(time_stamp)
                video_path = split_video(video_path, 0, time_stamp_sec)
                fps = 1
                if 300 < time_stamp_sec <= 600:
                    fps = 0.5
                elif time_stamp_sec > 600:
                    fps = 0.2

                # fps=args.chunk_size
                # fps = 1
                # if 300 < time_stamp_sec <= 600:
                #     fps = 2
                # elif time_stamp_sec > 600:
                #     fps = 1

                system_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "min_pixels": MIN_PIXELS,
                                "max_pixels": MAX_PIXELS,
                                "max_frames": MAX_FRAMES,
                                "min_frames": MIN_FRAMES,
                                "fps": fps
                            },
                            {
                                "type": "text",
                                "text": ""
                            },
                        ],
                    }
                ]


                def get_past_length(past):
                    if past is None:
                        return 0
                    return past[0][0].shape[2]


                # system_text = processor.apply_chat_template(
                #     system_messages, tokenize=False, add_generation_prompt=False, add_system_message=True
                # )

                system_text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'

                vision_only_text = "<|vision_start|><|video_pad|><|vision_end|>"


                end_text = (f'<|im_start|>user\nYou are an advanced video question-answering AI assistant. '
                            f'You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, '
                            f'choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.'
                            f'\n\nQuestion: {question}\n\nOptions:\n{options}\n\nThe best option is:<|im_end|>\n<|im_start|>assistant\n')

                image_inputs, video_inputs = process_vision_info(system_messages)  # video inputs: [n_frames, channel, h, w]

                video_inputs = video_inputs[0]
                if video_repeat > 1:
                    video_inputs = video_inputs.repeat(video_repeat, 1, 1, 1)

                first_chunk = video_inputs[0][None]
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
                past_len = get_past_length(past_key_values)

                seq_len0 = inputs0.input_ids.shape[1]
                # cache_position = [cc for cc in range(seq_len0)]

                input_step = INPUT_STEP
                num_frames = video_inputs.shape[0]
                is_first_frame = True

                for start in range(0, num_frames, input_step):
                    end = min(start + input_step, num_frames)
                    chunk = video_inputs[start:end]  # [chunk, C, H, W]

                    vision_process_start = time.time()
                    chunk_inputs = processor(
                        text=[vision_only_text],
                        images=None,
                        videos=chunk,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")
                    vision_process_time = time.time() - vision_process_start
                    # print(f'Vision_process_time_{vision_process_time * 1000:.2f}_ms_{chunk.size(0)}_frames')

                    L = chunk_inputs.input_ids.shape[1]
                    cache_position = torch.arange(L, device="cuda") + past_len

                    out = model(
                        **chunk_inputs,
                        past_key_values=past_key_values,
                        # cache_position=cache_position,
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

                end_inputs = processor(
                        text=[end_text],
                        images=None,
                        videos=None,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")

                L = end_inputs.input_ids.shape[1]
                cache_position = torch.arange(L, device="cuda") + past_len

                generated_ids = model.generate(
                    **end_inputs,
                    past_key_values=past_key_values,
                    # cache_position=cache_position,
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

                # Clean up heavy objects immediately to free GPU between samples
                del generated_ids
                del end_inputs
                del past_key_values
                # cache_position = None
                torch.cuda.empty_cache()

                output_dict = {
                    'question_id': question_id,
                    'task_type': task_type,
                    'question': question,
                    'time_stamp': time_stamp,
                    'answer': answer,
                    'options': eval(options),
                    'frames_required': frames_required,
                    'temporal_clue_type': temporal_clue_type,
                    'response': response
                }
                with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
                    f.write(json.dumps(output_dict) + '\n')
            except Exception as e:
                logger.error(f"Error in processing {row}: {e}")
            # break
    end_time = time.time()
    cost_time = int(end_time - start_time)

    # Print results
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    with open(OUTPUT_JSONL, 'r') as f:
        lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        cnt_total['overall'] += 1
        cnt_total[item['task_type']] += 1
        if extract_characters_regex(item['response']) == item['answer']:
            cnt_correct['overall'] += 1
            cnt_correct[item['task_type']] += 1
    task_types = ['Object Perception', 'Causal Reasoning', 'Clips Summarize', 'Attribute Perception',
                  'Event Understanding', 'Text-Rich Understanding', 'Prospective Reasoning', 'Spatial Understanding',
                  'Action Perception', 'Counting']
    for task_type in task_types:
        if cnt_total[task_type] == 0:
            logger.info(f"- {task_type}: No question processed")
        else:
            logger.info(
                f"- {task_type}: {cnt_correct[task_type]}/{cnt_total[task_type]} = {100 * cnt_correct[task_type] / cnt_total[task_type]:.2f}%")
    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(
            f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {100 * cnt_correct['overall'] / cnt_total['overall']:.2f}%")


    # Collect drop ratio info
    if DROP_METHOD is not None and DR_SAVE_PATH is not None:
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
