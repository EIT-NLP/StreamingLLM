from __future__ import annotations

import os
import sys
from pathlib import Path
import time
import threading
from transformers import TextIteratorStreamer


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
QWEN_INTERLEAVE_ROOT = VIDEO_STREAMING_ROOT / "Qwen2_5_vl_interleave"
MODEL_ROOT = PROJECT_ROOT / "model"


sys.path.insert(0, str(QWEN_INTERLEAVE_ROOT))
import warnings
import math
import logging
import random
import torch
from typing import Dict, Any, List
from .base import BaseModel
from .prompt import Qwen2VLPromptMixin
from .util import (
    get_rank_and_world_size,
    get_gpu_memory,
    auto_split_flag,
    listinstr,
    clean_think_answer_text,
)
from typing import Optional, Tuple


def save_topk_attention_indices(attentions, save_path, top_k=10, per_frame_token=0):
    """
    attentions: List of attention per step. Each element is a list of 28 layers,
                each layer is a tensor of shape (1, 28, query_len, key_len)
    Saves the top-k attention target token indices for each generated token.
    Each time creates a new folder (e.g., 0, 1, 2, ...) under base_path.
    """

    # Step 3: Extract and save attention indices
    topk_dict = {
        "per_frame_token": per_frame_token,
        "current_token": attentions[0][0].shape[-1],
    }

    for step in range(len(attentions)):
        layer_attns = attentions[step]  # List of 28 tensors
        layer_attns_tensor = torch.stack(layer_attns, dim=0).squeeze(
            1
        )  # (28, 28, query_len, key_len)

        last_query_attn = layer_attns_tensor[:, :, -1, :]  # (28, 28, key_len)
        avg_attn = (
            last_query_attn.mean(dim=0).mean(dim=0).to(torch.float).cpu().numpy()
        )  # (key_len,)

        topk_indices = np.argsort(avg_attn)[::-1][:top_k].tolist()
        topk_dict[f"token_{step + 1}"] = topk_indices

    with open(save_path, "w") as f:
        json.dump(topk_dict, f, indent=4)

    print(f"Top-{top_k} attention indices saved to: {save_path}")


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
            from transformers import AutoProcessor
            import sys

            sys.path.append(str(QWEN_INTERLEAVE_ROOT))
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

    @torch.no_grad()
    def interleave_inference(
        self,
        inputs: Any,
        video_start: float,
        video_end: float,
        answer_len: int = 0,
        evidence_token_per_frame: Optional[int] = None,
        num_core_frames: Optional[int] = None,
        # generate_kwargs: Dict[str, Any] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference with interleaved video‑text prompting.

        Args:
            self:  挂载到模型类的方法；需包含 self.model (LLM) 与 self.visual (vision encoder)。
            inputs: 一个字典或 HF BatchEncoding，字段：
                - input_ids              : (1, L)   已拼好 prompt + VIDEO_TOKEN 占位
                - position_ids           : (3, 1, L) 原始 3D RoPE 位置
                - pixel_values_videos    : (B, C, F, H, W) 视觉帧 PATCH 化前的张量
                - video_grid_thw         : (B, 3)    张量或列表，记录 [t, h, w] 的 patch grid 尺寸
            max_answer_tokens: 最后一段回答生成的最大 token 数。
            generate_kwargs: 其余传给 model.generate() 的参数。
            evidence_token_per_frame: 每帧生成的证据最大 token 数。仅在CoT时使用。
            num_core_frames: 核心帧数量。仅CoT需要。

        Returns:
            { 'input_ids': (1, L') 包含 prompt + 所有生成 token,
            'position_ids': (3, 1, L') 匹配的 3D 位置编码,
            'generated_answer': (1, <=max_answer_tokens + t*frame_answer_tokens) 所有回答拼接 }
        """

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

        IM_START = 151644
        IM_END = 151645
        VISION_START = 151652
        VISION_END = 151653
        VIDEO_TOKEN = 151656
        ASSISTANT_ID = 77091
        SYSTEM_ID = 8948
        END_TOKENS = [VISION_END, IM_END, 198]

        input_ids: torch.Tensor = inputs["input_ids"]
        position_ids: torch.Tensor = inputs["position_ids"]
        pixel_values_videos: torch.Tensor = inputs["pixel_values_videos"]
        video_grid_thw: torch.Tensor = inputs["video_grid_thw"]

        device = input_ids.device
        if input_ids.shape[0] != 1:
            raise ValueError("Only support batch size = 1")

        timing_info = {
            "video_time": None,
            "first_token_time": None,
            "think_time": None,
            "answer_time": None,
            "num_frame": None,
        }

        video_start_time = time.perf_counter()

        pixel_values_videos = pixel_values_videos.to(self.model.visual.dtype)
        video_embeds_full = self.model.visual(
            pixel_values_videos, grid_thw=video_grid_thw
        )

        video_token_mask = (input_ids == VIDEO_TOKEN).squeeze(0)
        if video_token_mask.sum().item() != video_embeds_full.shape[0]:
            raise ValueError(
                f"Video token count mismatch: {video_token_mask.sum()} tokens vs {video_embeds_full.shape[0]} embeds"
            )

        def replace_video_tokens(
            ids_slice: torch.Tensor, base_embeds: torch.Tensor, vid_ptr: int
        ):
            mask = ids_slice == VIDEO_TOKEN
            n = mask.sum().item()
            if n == 0:
                return base_embeds, vid_ptr
            emb_slice = base_embeds.clone()
            emb_slice[mask] = video_embeds_full[vid_ptr : vid_ptr + n].to(
                base_embeds.dtype
            )
            return emb_slice, vid_ptr + n

        t, h_orig, w_orig = video_grid_thw[0]
        h = int(h_orig // 2)
        w = int(w_orig // 2)
        frame_tokens = h * w

        vision_start_idx = (input_ids == VISION_START).nonzero()[0][1].item()
        video_end_idx = vision_start_idx + frame_tokens * t
        first_video_end_idx = vision_start_idx + frame_tokens + 1

        answer_start_idx = (input_ids == ASSISTANT_ID).nonzero()[0][1].item() - 1

        bos_ids = input_ids[:, answer_start_idx : answer_start_idx + 3]

        vid_ptr = 0

        # q + v1
        ids_cur = input_ids[:, :first_video_end_idx]
        embeds_cur_base = self.model.model.embed_tokens(ids_cur)
        embeds_cur, vid_ptr = replace_video_tokens(
            ids_cur.squeeze(0), embeds_cur_base.squeeze(0), vid_ptr
        )
        embeds_cur = embeds_cur.unsqueeze(0)

        pos_cur = position_ids[:, :, : ids_cur.shape[1]]
        cur_scalar_pos = pos_cur.max().item() + 1
        cur_t = 2
        cur_video_start_idx = first_video_end_idx

        all_gen_tokens: List[torch.Tensor] = []

        def append_generated(gen_ids: torch.Tensor):
            nonlocal ids_cur, embeds_cur, pos_cur, cur_scalar_pos
            ids_cur = torch.cat([ids_cur, gen_ids], dim=1)
            gen_embeds = self.model.model.embed_tokens(gen_ids)
            embeds_cur = torch.cat([embeds_cur, gen_embeds], dim=1)
            new_pos = (
                torch.arange(gen_ids.shape[1], device=device).reshape(1, 1, -1)
                + cur_scalar_pos
            )
            pos_gen = new_pos.repeat(3, 1, 1)
            pos_cur = torch.cat([pos_cur, pos_gen], dim=2)
            cur_scalar_pos += gen_ids.shape[1]

        # ---- a1 (BOS 3 tokens + generate) ----
        ids_cur = torch.cat([ids_cur, bos_ids], dim=1)
        bos_embeds = self.model.model.embed_tokens(bos_ids)
        embeds_cur = torch.cat([embeds_cur, bos_embeds], dim=1)
        pos_bos = (
            torch.arange(bos_ids.shape[-1], device=device).reshape(1, 1, -1)
            + cur_scalar_pos
        )
        pos_bos = pos_bos.repeat(3, 1, 1)
        pos_cur = torch.cat([pos_cur, pos_bos], dim=2)
        cur_scalar_pos += bos_ids.shape[-1]

        video_end_time = time.perf_counter()
        timing_info["video_time"] = video_end_time - video_start_time

        timing_info["num_frame"] = int(t.item())

        think_time = 0
        if self.CoT:
            frame_answer_tokens = random.randint(
                int(evidence_token_per_frame * 1.1),
                int(evidence_token_per_frame * 1.4),
            )
            # frame_answer_tokens = evidence_token_per_frame
            core_frame_num = 1

        elif "2plus" in self.dataset:
            frame_answer_tokens = random.randint(
                (answer_len // t // 2), answer_len // t
            )
        else:
            frame_answer_tokens = 3

        if frame_answer_tokens > 0:
            # print("frame1:")
            if self.CoT:
                kwargs = self.generate_kwargs
                kwargs["max_new_tokens"] = frame_answer_tokens

                streamer = TextIteratorStreamer(
                    self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
                )

                # ===================================================

                # ===================================================

                think_start_time = time.perf_counter()
                outputs_container = []
                first_token_time = None
                thread = threading.Thread(
                    target=lambda: outputs_container.append(
                        self.model.generate(
                            inputs_embeds=embeds_cur,
                            position_ids=pos_cur,
                            eos_token_id=THINK_END,
                            bad_words_ids=BAD_WORDS_NON_LAST,
                            streamer=streamer,
                            return_dict_in_generate=True,
                            **kwargs,
                        )
                    )
                )
                thread.start()

                generated_text = ""
                for token_text in streamer:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - think_start_time
                    generated_text += token_text

                thread.join()
                outputs = outputs_container[0]

                # timing_info["first_token_time"] = (
                #     stop.first_token_end_time - stop.first_token_start_time
                # )
                timing_info["first_token_time"] = first_token_time

                think_end_time = time.perf_counter()
                think_time += think_end_time - think_start_time
            else:
                outputs = self.model.generate(
                    inputs_embeds=embeds_cur,
                    position_ids=pos_cur,
                    max_new_tokens=frame_answer_tokens,
                    repetition_penalty=1.15,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )
            # save_topk_attention_indices(outputs['attentions'],save_path=os.path.join(attn_save_dir, 'step0.json'),per_frame_token=frame_tokens)
            new_answer = outputs["sequences"]
            all_gen_tokens.append(new_answer)
            append_generated(new_answer)

        while cur_video_start_idx < video_end_idx:
            # print(f"frame{cur_t//2+1}:")
            next_video_ids = input_ids[
                :, cur_video_start_idx : cur_video_start_idx + frame_tokens
            ]
            next_base_embeds = self.model.model.embed_tokens(next_video_ids)
            next_embeds, vid_ptr = replace_video_tokens(
                next_video_ids.squeeze(0), next_base_embeds.squeeze(0), vid_ptr
            )
            next_embeds = next_embeds.unsqueeze(0)

            t_pos = torch.full(
                (1, 1, frame_tokens),
                cur_t + cur_scalar_pos,
                device=device,
                dtype=torch.long,
            )
            h_idx = torch.arange(h, device=device).repeat_interleave(w) + cur_scalar_pos
            w_idx = torch.arange(w, device=device).repeat(h) + cur_scalar_pos
            hw_pos = torch.stack([h_idx, w_idx], dim=0).reshape(2, -1).unsqueeze(1)
            pos_video = torch.cat([t_pos, hw_pos], dim=0)

            ids_cur = torch.cat([ids_cur, next_video_ids], dim=1)
            embeds_cur = torch.cat([embeds_cur, next_embeds], dim=1)
            pos_cur = torch.cat([pos_cur, pos_video], dim=2)
            cur_scalar_pos = pos_cur.max().item() + 1
            cur_t += 2
            cur_video_start_idx += frame_tokens

            is_last_frame = cur_video_start_idx >= video_end_idx
            if is_last_frame:
                end_ids = torch.tensor(
                    END_TOKENS, device=device, dtype=input_ids.dtype
                ).reshape(1, -1)
                end_embeds = self.model.model.embed_tokens(end_ids)
                ids_cur = torch.cat([ids_cur, end_ids], dim=1)
                embeds_cur = torch.cat([embeds_cur, end_embeds], dim=1)
                new_pos = (
                    torch.arange(end_ids.shape[1], device=device).reshape(1, 1, -1)
                    + cur_scalar_pos
                )
                pos_end = new_pos.repeat(3, 1, 1)
                pos_cur = torch.cat([pos_cur, pos_end], dim=2)
                cur_scalar_pos += end_ids.shape[1]

            if is_last_frame:
                frame_answer_tokens = 512 if self.CoT else 300
            elif self.CoT:
                frame_answer_tokens = random.randint(
                    int(evidence_token_per_frame * 1.1),
                    int(evidence_token_per_frame * 1.4),
                )
                # frame_answer_tokens = evidence_token_per_frame

                core_frame_num += frame_answer_tokens > 0
            elif "2plus" in self.dataset:
                frame_answer_tokens = random.randint(
                    answer_len // t // 2, answer_len // t
                )
            else:
                frame_answer_tokens = 3

            if frame_answer_tokens <= 0:
                continue

            # print(f"frame_answer_tokens: {frame_answer_tokens}")
            if self.CoT:
                if is_last_frame:
                    kwargs = self.generate_kwargs
                    kwargs["max_new_tokens"] = frame_answer_tokens

                    answer_start_time = time.perf_counter()
                    outputs = self.model.generate(
                        inputs_embeds=embeds_cur,
                        position_ids=pos_cur,
                        return_dict_in_generate=True,
                        output_attentions=True,
                        eos_token_id=ANSWER_END,
                        bad_words_ids=BAD_WORDS_LAST,
                        **kwargs,
                    )

                    answer_end_time = time.perf_counter()
                    timing_info["answer_time"] = answer_end_time - answer_start_time
                else:
                    kwargs = self.generate_kwargs
                    kwargs["max_new_tokens"] = frame_answer_tokens

                    think_start_time = time.perf_counter()
                    outputs = self.model.generate(
                        inputs_embeds=embeds_cur,
                        position_ids=pos_cur,
                        return_dict_in_generate=True,
                        output_attentions=True,
                        eos_token_id=THINK_END,
                        bad_words_ids=BAD_WORDS_NON_LAST,
                        **kwargs,
                    )

                    think_end_time = time.perf_counter()
                    think_time += think_end_time - think_start_time
            else:
                outputs = self.model.generate(
                    inputs_embeds=embeds_cur,
                    position_ids=pos_cur,
                    max_new_tokens=frame_answer_tokens,
                    repetition_penalty=1.15,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )

            # save_topk_attention_indices(outputs['attentions'],save_path=os.path.join(attn_save_dir, f'step{count}.json'),per_frame_token=frame_tokens)
            new_answer = outputs["sequences"]
            # if self.CoT and is_last_frame:
            #     cot_answer = new_answer
            all_gen_tokens.append(new_answer)
            append_generated(new_answer)

        timing_info["think_time"] = think_time

        final_generated = torch.cat(all_gen_tokens, dim=1)
        if self.CoT:
            return {
                "input_ids": ids_cur,
                "position_ids": pos_cur,
                "generated_answer": final_generated,
                "all_gen_tokens": all_gen_tokens,
                "timing_info": timing_info,
            }
        else:
            return {
                "input_ids": ids_cur,
                "position_ids": pos_cur,
                "generated_answer": final_generated,
            }

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
        answer_len   : 根据human caption算的长度
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
        inputs = self.processor(
            text=text, images=images, videos=videos, padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        position_ids, _ = get_rope_index_25(
            2,
            inputs.input_ids,
            video_grid_thw=inputs.data["video_grid_thw"],
        )
        inputs["position_ids"] = position_ids

        video_end_time = time.perf_counter()
        video_time = video_end_time - video_start_time

        ans = self.interleave_inference(
            inputs=inputs,
            video_start=video_start,
            video_end=video_end,
            answer_len=answer_len,
            evidence_token_per_frame=evidence_token_per_frame,
            num_core_frames=num_core_frames,
        )

        timing_info = ans["timing_info"]
        timing_info["video_time"] += video_time
        new_tokens = ans["generated_answer"][0]
        caption = self.processor.tokenizer.decode(
            new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).strip()
        response = caption

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
            step_texts = []
            for token in ans["all_gen_tokens"]:
                text = self.processor.tokenizer.decode(
                    token[0],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ).strip()
                step_texts.append(text)
            return (
                step_texts,
                response,
                timing_info,
            )
        else:
            return response
