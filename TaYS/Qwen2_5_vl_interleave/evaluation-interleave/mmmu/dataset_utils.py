import os
import pandas as pd
from typing import Dict
from common_utils import toliststr, decode_base64_to_image_file

MMMU_DATASET_URL = "https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv"
MMMU_DATASET_MD5 = "521afc0f3bf341e6654327792781644d"


import json
from typing import List


def load_dataset(
    json_path: str = "/code/chr/download/ActivityNet_Captions/activitynet_captions_val1.json",
    video_root: str = "/code/chr/download/ActivityNet_Captions/extracted/video",
) -> List[Dict[str, str]]:
    """Load ActivityNet Captions dataset for evaluation."""

    assert os.path.exists(json_path), f"JSON file {json_path} not found."
    assert os.path.exists(video_root), f"Video root {video_root} not found."

    with open(json_path, "r") as f:
        data = json.load(f)

    samples = []
    for item in data:
        video_id = item["video_id"]
        video_filename = item["video"]
        caption = item["caption"]  # Ground truth, optional for evaluation
        video_path = os.path.join(video_root, video_filename)

        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found. Skipping.")
            continue
        elif "mkv" in video_path:
            print(f"Warning: Video file {video_path} is in MKV format. Skipping.")
            continue
        else:
            print(video_path)

        samples.append(
            {
                "video_id": video_id,
                "video_path": video_path,
                "prompt": "<video>\nDescribe every scene and its significance in the video.",
                "gt_caption": caption,
            }
        )

    return samples


def load_dataset_VideoEspresso(
    json_path: str = "/code/data/VideoEspresso/VideoEspresso_bench_hard_le_30s.jsonl",
) -> List[Dict[str, str]]:
    """Load dataset for evaluation."""
    samples = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                video_path = item["video_path"]

                if not os.path.exists(video_path):
                    print(f"⚠️ Warning: Video file {video_path} not found. Skipping.")
                    continue
                elif "mkv" in video_path:
                    # print(
                    #     f"⚠️ Warning: Video file {video_path} is in MKV format. Skipping."
                    # )
                    continue
                # else:
                #     print(f"✅ Found video: {video_path}")

                samples.append(
                    {
                        "video_path": video_path,
                        "core_frames_paths": item.get("core_frames_paths", []),
                        "core_frames_captions": item.get("core_frames_captions", []),
                        "question": item.get("question", "").strip(),
                        "answer": item.get("answer", "").strip(),
                        "video_start": 0,
                        "video_end": item.get("duration", 0),
                        "evidence": item.get("evidence", "").strip(),
                        "task": item.get("task", ""),
                        "correct_answer": item.get("correct_answer", ""),
                        "options": item.get("options", []),
                        "duration": item.get("duration", 0),
                        "core_frames_timestamps": item.get(
                            "core_frames_timestamps", []
                        ),
                    }
                )
            except Exception as e:
                print(f"❌ Error parsing line: {e}")
                continue

    return samples


def load_dataset_livesports(
    json_path: str = "/code/data/PE-Video/test_PE_token_2plus.jsonl",
) -> List[Dict[str, str]]:
    """Load dataset for evaluation."""
    samples = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                video_id = item["video_id"]
                video_path = item["video_path"]

                if not os.path.exists(video_path):
                    print(f"⚠️ Warning: Video file {video_path} not found. Skipping.")
                    continue
                elif "mkv" in video_path:
                    print(
                        f"⚠️ Warning: Video file {video_path} is in MKV format. Skipping."
                    )
                    continue
                # else:
                #     print(f"✅ Found video: {video_path}")

                samples.append(
                    {
                        "video_id": video_id,
                        "video_path": video_path,
                        "caption": item.get("human_caption", ""),
                        "video_start": 0,
                        "video_end": item.get("video_duration_in_s", 0),
                    }
                )
            except Exception as e:
                print(f"❌ Error parsing line: {e}")
                continue

    return samples


def load_dataset_llava_video(
    json_path: str = "/code/Streaming_video/data/LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed_valid_eval.json",
    video_root: str = "/code/Streaming_video/data/LLaVA-Video-178K/0_30_s_academic_v0_1",
) -> List[Dict[str, str]]:
    """Load ActivityNet Captions dataset for evaluation."""

    assert os.path.exists(json_path), f"JSON file {json_path} not found."
    assert os.path.exists(video_root), f"Video root {video_root} not found."

    with open(json_path, "r") as f:
        data = json.load(f)

    samples = []
    for item in data:
        video_id = item["id"]
        video_filename = item["video"]
        caption = item["conversations"][1][
            "value"
        ]  # Ground truth, optional for evaluation
        video_path = os.path.join(video_root, video_filename)

        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found. Skipping.")
            continue
        elif "mkv" in video_path:
            print(f"Warning: Video file {video_path} is in MKV format. Skipping.")
            continue
        else:
            print(video_path)

        samples.append(
            {
                "video_id": video_id,
                "video_path": video_path,
                "prompt": item["conversations"][0]["value"],
                "gt_caption": caption,
            }
        )

    return samples


def dump_image(line, img_root):
    """Save image data to disk and return the path."""
    os.makedirs(img_root, exist_ok=True)

    if "image" in line:
        if isinstance(line["image"], list):
            tgt_path = []
            assert "image_path" in line
            for img, im_name in zip(line["image"], line["image_path"]):
                path = os.path.join(img_root, im_name)
                if not os.path.exists(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)
        else:
            tgt_path = os.path.join(img_root, f"{line['index']}.jpg")
            if not os.path.exists(tgt_path):
                decode_base64_to_image_file(line["image"], tgt_path)
            tgt_path = [tgt_path]
    else:
        assert "image_path" in line
        tgt_path = toliststr(line["image_path"])

    return tgt_path


def dump_video(line, video_root):
    """Save video file to disk and return the path.

    Args:
        line (dict): A sample entry containing 'video' and 'data_path' keys.
        video_root (str): Target folder to save dumped videos.

    Returns:
        tgt_path (str): Path to saved video file.
    """
    os.makedirs(video_root, exist_ok=True)

    tgt_path = [line["video_path"]]

    return tgt_path


def MMMU_preproc(data):
    """
    Preprocess MMMU dataset to reformulate open questions to multi-choice ones.
    This aligns with the implementation in multiple_choice.py
    """
    print("Preprocessing MMMU dataset...")
    cnt = 0
    As, Bs, Ans = list(data["A"]), list(data["B"]), list(data["answer"])
    lt = len(data)
    for i in range(lt):
        if pd.isna(As[i]):
            As[i] = Ans[i]
            Bs[i] = "Other Answers"
            cnt += 1
    print(
        f"During MMMU_preproc in Evaluation, {cnt} open questions are re-formulated to multi-choice ones."
    )
    data["A"] = As
    data["B"] = Bs
    return data
