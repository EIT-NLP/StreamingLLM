import re

# Define placeholders for dataset paths
# CAMBRIAN_737K = {
#     "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
#     "data_path": "",
# }

# MP_DOC = {
#     "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
#     "data_path": "PATH_TO_MP_DOC_DATA",
# }

# CLEVR_MC = {
#     "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
#     "data_path": "PATH_TO_CLEVR_MC_DATA",
# }

# VIDEOCHATGPT = {
#     "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
#     "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
# }

LLAVA_VIDEO = {
    "annotation_path": "/code/Streaming_video/data/LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed_valid_train.json",
    "data_path": "/code/chr/download/LLaVA-Video/LLaVA-Video-178K/0_30_s_academic_v0_1",
}

data_dict = {
    # "cambrian_737k": CAMBRIAN_737K,
    # "mp_doc": MP_DOC,
    # "clevr_mc": CLEVR_MC,
    # "videochatgpt": VIDEOCHATGPT,
    "llava_video": LLAVA_VIDEO,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["llava_video"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
