from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

from transformers import StoppingCriteria, StoppingCriteriaList


def find_project_root():
    """查找包含 code/, data/, model/ 三个目录的根目录"""
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (
            (parent / "code").exists()
            and (parent / "data").exists()
            and (parent / "model").exists()
        ):
            return parent

    raise FileNotFoundError("未找到包含 code/, data/, model/ 三个目录的根目录")


PROJECT_ROOT = find_project_root()
VIDEO_STREAMING_ROOT = PROJECT_ROOT / "code" / "video_streaming"
QWEN_GROUP_ROOT = VIDEO_STREAMING_ROOT / "Qwen2_5_vl_group"
MODEL_ROOT = PROJECT_ROOT / "model"


sys.path.insert(0, str(QWEN_GROUP_ROOT))
import logging
import math
import warnings
from typing import Optional, Tuple

import torch

from .base import BaseModel
from .prompt import Qwen2VLPromptMixin
from .util import (
    auto_split_flag,
    clean_think_answer_text,
    get_gpu_memory,
    get_rank_and_world_size,
    listinstr,
)


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * 2

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def ensure_image_url(image: str) -> str:
    prefixes = ["http://", "https://", "file://", "data:image;"]
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return "file://" + image
    raise ValueError(f"Invalid image: {image}")


def ensure_video_url(video: str) -> str:
    prefixes = ["http://", "https://", "file://", "data:video;"]
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return "file://" + video
    raise ValueError(f"Invalid video: {video}")


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"model.layers.{layer_cnt}"] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map["visual"] = rank
    device_map["model.embed_tokens"] = rank
    device_map["model.norm"] = last_gpu
    device_map["model.rotary_emb"] = last_gpu
    device_map["lm_head"] = last_gpu
    return device_map


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        do_sample: bool = False,
        num_beams: int = 1,
        no_repeat_ngram_size: int = None,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        if listinstr(["2.5", "2_5", "qwen25"], model_path.lower()):
            import sys

            from transformers import AutoProcessor

            sys.path.append(str(QWEN_GROUP_ROOT))
            from model_code.modeling_qwen2_5_vl_for_eval import (
                Qwen2_5_VLForConditionalGeneration,
            )

            MODEL_CLS = Qwen2_5_VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(
                model_path, padding_side="right"
            )

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        # If only one process and GPU memory is less than 40GB
        if "72b" in self.model_path.lower() or "32b" in self.model_path.lower():
            self.model = MODEL_CLS.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=split_model(),
                attn_implementation="flash_attention_2",
            )
            self.model.eval()
        elif auto_split_flag():
            assert world_size == 1, (
                "Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL"
            )
            # Will Use All GPUs to run one model
            self.model = MODEL_CLS.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="cpu",
                attn_implementation="flash_attention_2",
            )
            self.model.cuda().eval()

        torch.cuda.empty_cache()

    def _prepare_content(
        self, inputs: list[dict[str, str]], dataset: str | None = None
    ) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s["type"] == "image":
                item = {"type": "image", "image": ensure_image_url(s["value"])}
                if dataset == "OCRBench":
                    item["min_pixels"] = 10 * 10 * 28 * 28
                    warnings.warn(
                        f"OCRBench dataset uses custom min_pixels={item['min_pixels']}"
                    )
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item["min_pixels"] = self.min_pixels
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
            elif s["type"] == "video":
                item = {"type": "video", "video": ensure_video_url(s["value"])}
                if self.fps is not None:
                    item["fps"] = self.fps
                elif self.nframe is not None:
                    import cv2

                    video = cv2.VideoCapture(s["value"])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = (
                            frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        )
                        print(f"use {new_frame_count} for {s['value']}")
                        item["nframes"] = new_frame_count
                    else:
                        item["nframes"] = self.nframe
            elif s["type"] == "text":
                item = {"type": "text", "text": s["value"]}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(
        self,
        message,
        video_start,
        video_end,
        dataset=None,
        answer_len=None,
        evidence_token_per_frame: Optional[int] = None,
        num_core_frames: Optional[int] = None,
    ):
        """
        message      : str, 用户自然语言问题（如 "请给视频加字幕"）
        video_start  : float, 起始秒
        video_end    : float, 结束秒
        dataset      : 仅为了兼容旧接口，这里继续透传给 _prepare_content
        evidence_token_per_frame : 仅CoT需要，每个关键帧生成的token数
        num_core_frames : 仅CoT需要，核心帧数量

        return       : str, 将所有分段字幕拼成一行（每段前缀 [t0‐t1]）
        """
        self.dataset = dataset
        try:
            import sys

            sys.path.append(
                f"{VIDEO_STREAMING_ROOT}/Qwen2_5_vl_batch/qwen-vl-utils/src"
            )
            from qwen_vl_utils_forgen import process_vision_info_forgen
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err
        messages = []

        # self.system_prompt = "You are an expert video commentator providing real-time, insightful, and engaging commentary on visual content.\n"
        self.system_prompt = "You are a helpful assistant."
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        timing_info = {
            "video_time": None,
            "first_token_time": None,
            "think_time": None,
            "answer_time": None,
            "num_frame": None,
        }

        video_start_time = time.perf_counter()
        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        print(f"messages: {messages}")
        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info_forgen([messages], video_start, video_end)
        # print(text[0])
        # <|im_start|>system
        # You are a helpful assistant.<|im_end|>
        # <|im_start|>user
        # What is the scene and activity shown in this video?<|vision_start|><|video_pad|><|vision_end|><|im_end|>

        inputs_in = self.processor(
            text=text[0][:-22],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        inputs_in = inputs_in.to("cuda")
        running_text = text[0][-22:]
        inputs_out = self.processor(
            text=running_text, images=None, videos=None, return_tensors="pt"
        ).to("cuda")

        pos_in, _ = get_rope_index_25(
            2,
            inputs_in.input_ids,
            video_grid_thw=inputs_in.data["video_grid_thw"],
        )
        pos_out, _ = get_rope_index_25(
            2,
            inputs_out.input_ids,
        )
        t = inputs_in.data["video_grid_thw"][0][0]
        token_num_pre_frame = (
            inputs_in.data["video_grid_thw"][0][1]
            / 2
            * inputs_in.data["video_grid_thw"][0][2]
            / 2
        )

        SKIP = 151665
        THINK_END = 151666
        ANSWER_START = 151667
        ANSWER_END = 151668
        TOOL_CALL = 151657
        TOOL_END = 151658
        VISION_START = 151652
        VISION_END = 151653
        IM_START = 151644
        IM_END = 151645
        VIDEO_PAD = 151656

        BAD_WORDS_NON_LAST = [
            [ANSWER_START],
            [ANSWER_END],
            [TOOL_CALL],
            [TOOL_END],
            [VISION_START],
            [VISION_END],
            [IM_START],
            [IM_END],
            [VIDEO_PAD],
        ]
        BAD_WORDS_LAST = [
            [SKIP],
            [THINK_END],
            [TOOL_CALL],
            [TOOL_END],
            [VISION_START],
            [VISION_END],
            [IM_START],
            [IM_END],
            [VIDEO_PAD],
        ]

        VIDEO_TOKEN = 151656
        ASSISTANT_ID = 77091
        SYSTEM_ID = 8948
        vision_start_pos = (
            inputs_in.input_ids[0].eq(VISION_START).nonzero(as_tuple=True)[0].item()
        )
        vision_end_pos = (
            inputs_in.input_ids[0].eq(VISION_END).nonzero(as_tuple=True)[0].item()
        )

        captions = []
        all_tokens = []
        # turn input id  into  input embed
        pixel_values_videos: torch.Tensor = inputs_in["pixel_values_videos"]
        video_grid_thw: torch.Tensor = inputs_in["video_grid_thw"]
        pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
        inputs_embeds_in = self.model.model.embed_tokens(inputs_in.input_ids)
        inputs_embeds_out = self.model.model.embed_tokens(inputs_out.input_ids)
        input_ids_out = inputs_out.input_ids
        pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
        video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        n_video_tokens = (
            (inputs_in.input_ids == self.model.config.video_token_id).sum().item()
        )
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = inputs_in.input_ids == self.model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds_in)
        video_mask = mask_expanded.to(inputs_embeds_in.device)

        video_embeds = video_embeds.to(inputs_embeds_in.device, inputs_embeds_in.dtype)
        inputs_embeds_in = inputs_embeds_in.masked_scatter(video_mask, video_embeds)
        token_schedule = []

        video_end_time = time.perf_counter()
        timing_info["video_time"] = video_end_time - video_start_time

        timing_info["num_frame"] = int(t.item())

        think_time = 0
        for i in range(t):
            # print(f"frame {i}")
            if i != t - 1:
                temp_inputs_ids_in = inputs_in.input_ids[
                    :, : int(vision_start_pos + (i + 1) * token_num_pre_frame + 1)
                ]
                temp_pos_in = pos_in[
                    :, :, : int(vision_start_pos + (i + 1) * token_num_pre_frame + 1)
                ]
                temp_inputs_embeds_in = inputs_embeds_in[
                    :, : int(vision_start_pos + (i + 1) * token_num_pre_frame + 1), :
                ]
            else:
                temp_inputs_ids_in = inputs_in.input_ids
                temp_pos_in = pos_in
                temp_inputs_embeds_in = inputs_embeds_in

            # concat temp_inputs_embeds_in and inputs_embeds_out, temp_pos_in and pos_out
            inputs_embed = torch.cat([temp_inputs_embeds_in, inputs_embeds_out], dim=1)

            input_ids = torch.cat([temp_inputs_ids_in, input_ids_out], dim=1)

            pos = torch.cat([temp_pos_in, pos_out], dim=2)

            if self.CoT:
                frame_answer_tokens = random.randint(
                    int(evidence_token_per_frame * 1.4),
                    int(evidence_token_per_frame * 1.5),
                )
            elif "2plus" in self.dataset:
                frame_answer_tokens = random.randint(
                    (answer_len // t // 2), answer_len // t
                )
            else:
                frame_answer_tokens = 3

            if i != t - 1:
                token_schedule.append(frame_answer_tokens)
                if self.CoT:
                    kwargs = self.generate_kwargs
                    kwargs["max_new_tokens"] = frame_answer_tokens

                    class FirstTokenStop(StoppingCriteria):
                        def __init__(self, initial_len):
                            super().__init__()
                            self.first_token_start_time = None
                            self.first_token_end_time = None
                            self.initial_len = initial_len

                        def __call__(self, input_ids, scores, **kwargs):
                            if self.first_token_start_time is None:
                                self.first_token_start_time = time.perf_counter()
                            if (
                                self.first_token_end_time is None
                                and input_ids.shape[1] > self.initial_len
                            ):
                                self.first_token_end_time = time.perf_counter()
                            return False

                    # ===================================================

                    # ===================================================
                    if i == 0:
                        stop = FirstTokenStop(input_ids.shape[1])
                        stopping_criteria = StoppingCriteriaList([stop])

                        think_start_time = time.perf_counter()
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            inputs_embeds=inputs_embed,
                            position_ids=pos,
                            return_dict_in_generate=True,
                            video_grid_thw=video_grid_thw,
                            token_len_schedule=token_schedule,
                            eos_token_id=THINK_END,
                            bad_words_ids=BAD_WORDS_NON_LAST,
                            stopping_criteria=stopping_criteria,
                            **kwargs,
                        )

                        timing_info["first_token_time"] = (
                            stop.first_token_end_time - stop.first_token_start_time
                        )

                        think_end_time = time.perf_counter()
                        think_time += think_end_time - think_start_time
                    else:
                        think_start_time = time.perf_counter()
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            inputs_embeds=inputs_embed,
                            position_ids=pos,
                            return_dict_in_generate=True,
                            video_grid_thw=video_grid_thw,
                            token_len_schedule=token_schedule,
                            eos_token_id=THINK_END,
                            bad_words_ids=BAD_WORDS_NON_LAST,
                            **kwargs,
                        )

                        think_end_time = time.perf_counter()
                        think_time += think_end_time - think_start_time
                    # ===================================================
                else:
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embed,
                        position_ids=pos,
                        return_dict_in_generate=True,
                        max_new_tokens=frame_answer_tokens,
                        repetition_penalty=1.15,
                        video_grid_thw=video_grid_thw,
                        token_len_schedule=token_schedule,
                    )
            else:
                token_schedule.append(frame_answer_tokens)
                last_frame_answer_tokens = 512 if self.CoT else 300
                if self.CoT:
                    answer_start_time = time.perf_counter()
                    kwargs = self.generate_kwargs
                    kwargs["max_new_tokens"] = last_frame_answer_tokens
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embed,
                        position_ids=pos,
                        return_dict_in_generate=True,
                        video_grid_thw=video_grid_thw,
                        token_len_schedule=token_schedule,
                        eos_token_id=ANSWER_END,
                        bad_words_ids=BAD_WORDS_LAST,
                        **kwargs,
                    )

                    answer_end_time = time.perf_counter()
                    timing_info["answer_time"] = answer_end_time - answer_start_time
                else:
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embed,
                        position_ids=pos,
                        return_dict_in_generate=True,
                        max_new_tokens=last_frame_answer_tokens,
                        repetition_penalty=1.15,
                        video_grid_thw=video_grid_thw,
                        token_len_schedule=token_schedule,
                        # max_new_tokens=5*int(video_end-video_start)-int(video_end-video_start)*3+3, repetition_penalty=1.15,video_grid_thw=video_grid_thw
                    )

            new_tokens = outputs.sequences[0, input_ids.shape[1] :]
            # print("new_tokens: ", new_tokens)

            token_schedule[-1] = new_tokens.shape[0]

            caption = self.processor.tokenizer.decode(
                new_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()
            captions.append(f"{caption}")
            # if self.CoT and i == t - 1:
            #     cot_answer = new_tokens
            all_tokens.append(new_tokens)
            new_token_embeds = self.model.model.embed_tokens(new_tokens.unsqueeze(0))
            last_pos_id = pos_out[0, 0, -1]
            new_pos_out, _ = get_rope_index_25(
                2,
                new_tokens.unsqueeze(0),
            )
            new_pos_out = new_pos_out.add(last_pos_id + 1)
            inputs_embeds_out = torch.cat([inputs_embeds_out, new_token_embeds], dim=1)
            input_ids_out = torch.cat([input_ids_out, new_tokens.unsqueeze(0)], dim=1)
            pos_out = torch.cat([pos_out, new_pos_out], dim=2)
            if self.verbose:
                print(f"\033[32m{captions[-1]}\033[0m")

        timing_info["think_time"] = think_time
        # -------------------------------------------------

        # -------------------------------------------------
        all_tokens_tensor = torch.cat(all_tokens, dim=0)  # (total_length,)
        response = self.processor.tokenizer.decode(
            all_tokens_tensor,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()

        # response = " ".join(captions)

        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            counter = 1
            for i, ch in enumerate(resp):
                if ch == "{":
                    counter += 1
                elif ch == "}":
                    counter -= 1
                if counter == 0:
                    response = resp[:i]
                    break

        if self.CoT:
            response = clean_think_answer_text(response)
            return (
                captions,
                response,
                timing_info,
            )
        else:
            return response
