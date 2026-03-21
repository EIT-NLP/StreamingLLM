def read_last_line(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        return lines[-1].strip() if lines else None


import json

last_line = read_last_line(
    "/code/chr/download/Live-WhisperX-526K/live_whisperx_526k_with_seeks.jsonl"
)
seeks = json.loads(last_line)
handles = []
annotation_path = (
    "/code/chr/download/Live-WhisperX-526K/live_whisperx_526k_with_seeks.jsonl"
)
handles.extend(zip([annotation_path] * len(seeks), seeks))

annotation_path, seek = handles[0]
with open(annotation_path) as f:
    f.seek(seek)
    line = f.readline()
line = json.loads(line)

print(line)
