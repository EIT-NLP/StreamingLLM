import json
import os

from tqdm import tqdm

data_dir = "/code/chr/download/Live-WhisperX-526K/videos/videos"

output_jsonl = "/code/chr/download/Live-WhisperX-526K/filtered_live_whisperx_more.jsonl"


json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

matched_items = []

for json_file in tqdm(json_files):
    json_path = os.path.join(data_dir, json_file)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base_name = os.path.splitext(json_file)[0]
        mp4_path = os.path.join(data_dir, base_name + ".mp4")

        if not os.path.exists(mp4_path):
            print(f"[Warning] 视频不存在：{mp4_path}")
            continue

        user_block = data[0]["content"][0]
        if user_block["type"] == "video":
            user_block["video"] = mp4_path
            matched_items.append(data)

    except Exception as e:
        print(f"[Error] 处理 {json_file} 时出错: {e}")


with open(output_jsonl, "w", encoding="utf-8") as fw:
    for item in matched_items:
        fw.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"完成处理，共保存 {len(matched_items)} 条记录到：{output_jsonl}")


# import os
# import json
# import re


# jsonl_path = '/code/chr/download/Live-WhisperX-526K/live_whisperx_526k_with_seeks.jsonl'
# video_dir = '/code/chr/download/Live-WhisperX-526K/videos/videos'
# output_path = '/code/chr/download/Live-WhisperX-526K/filtered_live_whisperx_more.jsonl'


# video_files = os.listdir(video_dir)


# video_map = {}
# pattern = re.compile(r"(?P<id>[^_]+)_(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)_2fps\.mp4")

# for filename in video_files:
#     if filename.endswith('.mp4'):
#         match = pattern.match(filename)
#         if match:
#             video_id = match.group("id")
#             start = float(match.group("start"))
#             end = float(match.group("end"))
#             video_map.setdefault(video_id, []).append({
#                 'filename': filename,
#                 'start': start,
#                 'end': end,
#                 'full_path': os.path.join(video_dir, filename)
#             })


# matched_items = []
# from tqdm import tqdm
# with open(jsonl_path, 'r', encoding='utf-8') as f:
#     for idx, line in tqdm(enumerate(f)):
#         try:
#             item = json.loads(line)
#             user_block = item[0]['content'][0]
#             video_path = user_block['video']  # e.g., 'video/youtube/1B0DuykUGOM.mp4'
#             video_id = os.path.splitext(os.path.basename(video_path))[0]
#             video_start = user_block['video_start']
#             video_end = user_block['video_end']


#             candidates = video_map.get(video_id, [])
#             for cand in candidates:
#                 if abs(cand['start'] - video_start) < 0.01 and abs(cand['end'] - video_end) < 0.01:

#                     item[0]['content'][0]['video'] = cand['full_path']
#                     # item[0]['content'][0]['video_start'] = None
#                     # item[0]['content'][0]['video_end'] = None
#                     matched_items.append(item)


#         except Exception as e:
#             print(f"[Error] Line {idx}: {e}")


# with open(output_path, 'w', encoding='utf-8') as fw:
#     for entry in matched_items:
#         fw.write(json.dumps(entry, ensure_ascii=False) + '\n')
