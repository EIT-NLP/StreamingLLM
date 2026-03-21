from __future__ import annotations

import os
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
QWEN_GROUP_ROOT = VIDEO_STREAMING_ROOT / "Qwen2_5_vl_batch"
MODEL_ROOT = PROJECT_ROOT / "model"

sys.path.insert(0, str(QWEN_GROUP_ROOT))
import logging
import math
import warnings

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
        repetition_penalty=1.15,
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
            from model_code.modeling_qwen2_5_vl import (
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

    def generate_inner(self, message, video_start, video_end, dataset=None):
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
        inputs = self.processor(
            text=text, images=images, videos=videos, padding=True, return_tensors="pt"
        ).to("cuda")
        video_end_time = time.perf_counter()
        timing_info["video_time"] = video_end_time - video_start_time
        timing_info["num_frame"] = int(inputs.data["video_grid_thw"][0][0].item())
        timing_info["think_time"] = 0
        if self.CoT:

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

            stop = FirstTokenStop(inputs.input_ids.shape[1])
            stopping_criteria = StoppingCriteriaList([stop])

            answer_start_time = time.perf_counter()
            generated_ids = self.model.generate(
                **inputs,
                **self.generate_kwargs,
                stopping_criteria=stopping_criteria,
            )
            answer_end_time = time.perf_counter()
            timing_info["first_token_time"] = (
                stop.first_token_end_time - stop.first_token_start_time
            )
            timing_info["answer_time"] = answer_end_time - answer_start_time
        else:
            generated_ids = self.model.generate(
                **inputs,
                repetition_penalty=1.15,
                max_new_tokens=300,
                **self.generate_kwargs,
            )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        response = out[0]

        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        response = clean_think_answer_text(response)
        return response, timing_info
