import os

import cv2


def is_video_corrupted(video_path):
    """
    检查视频文件是否损坏。
    尝试打开视频并读取第一帧，如果失败则认为视频损坏。
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return True
        ret, _ = cap.read()
        cap.release()
        return not ret
    except Exception as e:
        print(f"Error checking video {video_path}: {e}")
        return True


def filter_corrupted_videos(directory):
    """
    遍历指定目录，检查所有视频文件是否损坏。
    """
    corrupted_videos = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mkv", ".mov")):
                video_path = os.path.join(root, file)
                if is_video_corrupted(video_path):
                    corrupted_videos.append(video_path)
                    print(f"Corrupted video found: {video_path}")
    return corrupted_videos


if __name__ == "__main__":
    video_directory = "/code/chr/download/ActivityNet_Captions/extracted/video/"
    corrupted_videos = filter_corrupted_videos(video_directory)
    print(f"Total corrupted videos: {len(corrupted_videos)}")
    if corrupted_videos:
        with open("corrupted_videos.txt", "w") as f:
            for video in corrupted_videos:
                f.write(video + "\n")
