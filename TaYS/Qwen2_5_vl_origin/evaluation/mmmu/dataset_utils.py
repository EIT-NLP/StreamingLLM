import os
from typing import Dict

import pandas as pd
from common_utils import decode_base64_to_image_file, toliststr

MMMU_DATASET_URL = "https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv"
MMMU_DATASET_MD5 = "521afc0f3bf341e6654327792781644d"

# def load_dataset(dataset_name='MMMU_DEV_VAL'):
#     """Load the MMMU dataset."""
#     data_root = os.path.join(os.environ['LMUData'])
#     os.makedirs(data_root, exist_ok=True)

#     file_name = f"{dataset_name}.tsv"
#     data_path = os.path.join(data_root, file_name)

#     # Download if not exists or MD5 doesn't match
#     if not os.path.exists(data_path) or md5(data_path) != MMMU_DATASET_MD5:
#         print(f"Downloading {dataset_name} dataset...")
#         download_file(MMMU_DATASET_URL, data_path)

#     # Load the dataset
#     data = pd.read_csv(data_path, sep='\t').iloc[:8]

#     # Process the dataset
#     data['index'] = [str(x) for x in data['index']]

#     # Handle image data
#     if 'image' in data:
#         data['image'] = [str(x) for x in data['image']]
#         image_map = {x: y for x, y in zip(data['index'], data['image'])}
#         for k in image_map:
#             if len(image_map[k]) <= 64:
#                 idx = image_map[k]
#                 assert idx in image_map and len(image_map[idx]) > 64
#                 image_map[k] = image_map[idx]

#         images = [toliststr(image_map[k]) for k in data['index']]
#         data['image'] = [x[0] if len(x) == 1 else x for x in images]

#     # Handle image paths
#     if 'image_path' in data:
#         paths = [toliststr(x) for x in data['image_path']]
#         data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

#     # Convert index to int if possible
#     if np.all([isinstance(x, int) or x.isdigit() for x in data['index']]):
#         data['index'] = [int(x) for x in data['index']]

#     return data


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


#   {
#     "video_id": "6gyD-Mte2ZM",
#     "url": "https://www.youtube.com/watch?v=6gyD-Mte2ZM",
#     "event_id": 2,
#     "begin": 26.127,
#     "end": 38.876,
#     "event_type": 1,
#     "video": "6gyD-Mte2ZM.mp4",
#     "class": "Bowling",
#     "event_title": "Tom's performance and struggles during the game.",
#     "event_asr": "[[26.127, 29.509, \"A season for him and his first television show.\"], [29.529, 31.131, \"A big cleanup effort here.\"], [31.151, 33.973, \"A lot of RPMs coming out of this hand right here.\"], [38.135, 38.876, \"Yeah!\"]]",
#     "event_asr_text": "A season for him and his first television show. A big cleanup effort here. A lot of RPMs coming out of this hand right here. Yeah!",
#     "preasr": "[[0.874, 8.337, \"The number two qualifier is a four-time PBA regional champion from Wesley Chapel, Florida, TD, Tom Doherty.\"], [10.358, 19.302, \"Doherty had to bowl 72 games over six days just to get here on television, by far more than any of our four finalists.\"]]",
#     "preasr_text": "The number two qualifier is a four-time PBA regional champion from Wesley Chapel, Florida, TD, Tom Doherty. Doherty had to bowl 72 games over six days just to get here on television, by far more than any of our four finalists."
#   },


def load_dataset_livesports(
    json_path: str = "/code/chr/download/PE-Video/test_2plus.jsonl",
    video_root: str = "/code/chr/download/PE-Video/test",
) -> List[Dict[str, str]]:
    """Load ActivityNet Captions dataset for evaluation."""

    assert os.path.exists(json_path), f"JSON file {json_path} not found."
    assert os.path.exists(video_root), f"Video root {video_root} not found."

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
                else:
                    print(f"✅ Found video: {video_path}")

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
