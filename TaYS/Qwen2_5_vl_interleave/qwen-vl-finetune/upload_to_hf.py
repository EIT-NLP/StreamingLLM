from huggingface_hub import HfApi, create_repo

local_folder_path = "./output"


repo_id = "jinmingsjtu/video_streaming"


repo_type = "model"


try:
    create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True)
    print(f"仓库 '{repo_id}' 已确认存在或已创建。")
except Exception as e:
    print(f"创建或检查仓库时出错: {e}")


api = HfApi()

print(f"准备上传文件夹 '{local_folder_path}' 到 '{repo_id}'...")


api.upload_folder(
    folder_path=local_folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Upload output folder content",
)

print("文件夹上传成功！请在以下链接查看：")
print(f"https://huggingface.co/{repo_id}")
