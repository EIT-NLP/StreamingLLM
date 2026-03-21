import copy
import functools
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import tqdm
import transformers

# from .rope2d import get_rope_index_25, get_rope_index_2
from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoProcessor, logging

logger = logging.get_logger(__name__)

FPS = 2


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
        raise NotImplementedError(
            "[get_rope_index_25] The fallback path (no input_ids or no vision grid info) is not implemented. "
            "Please provide both input_ids and image/video grid_thw, or extend this function if needed."
        )
        # if attention_mask is not None:
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        #     position_ids = (
        #         position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        #     )
        #     max_position_ids = position_ids.max(0, keepdim=False)[0].max(
        #         -1, keepdim=True
        #     )[0]
        #     mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        # else:
        #     position_ids = (
        #         torch.arange(input_ids.shape[1], device=input_ids.device)
        #         .view(1, 1, -1)
        #         .expand(3, input_ids.shape[0], -1)
        #     )
        #     mrope_position_deltas = torch.zeros(
        #         [input_ids.shape[0], 1],
        #         device=input_ids.device,
        #         dtype=input_ids.dtype,
        #     )

        # return position_ids, mrope_position_deltas


@dataclass
class DataArguments:
    annotation_paths: list[str] = field(default_factory=list)
    initial_fps_frames: int = int(FPS) * 3
    streaming_fps_frames: int = int(FPS)
    with_context: bool = False


# --- some utils ---
def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2)  # avoid last \n
        while f.read(1) != b"\n":
            f.seek(-2, 1)
        return f.readline()


def bytes_to_pil(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == "P":
        image = image.convert("RGBA")
    return image.convert("RGB")


def get_phrase_before_timestamp(
    text_stream, timestamp, video_start, start_from: int = 0
):
    # phrase = ''
    # for i, (ws, we, word) in enumerate(text_stream[start_from:]):
    #     if timestamp >= we-video_start:
    #         phrase += ' ' + word.strip()
    #     else:
    #         break
    # return phrase.strip(), i + start_from
    phrase = ""
    for i, (ws, we, word) in enumerate(text_stream[start_from:]):
        phrase += " " + word.strip()

    return phrase.strip(), i + start_from


# --- some utils ---
import linecache


def read_jsonl_line_linecache(path: str, index: int):
    line = linecache.getline(path, index + 1)
    if line:
        return json.loads(line)
    raise IndexError("Index out of range")


class LMMDataset(Dataset):
    def __init__(
        self, *, annotation_paths: list[str], processor: AutoProcessor, data_args=None
    ):
        super().__init__()
        initial_fps_frames = getattr(data_args, "initial_fps_frames", 6)
        streaming_fps_frames = getattr(data_args, "streaming_fps_frames", 2)
        with_context = getattr(data_args, "with_context", False)
        self.data_args = data_args
        # self.data_args.image_processor.max_pixels = data_args.max_pixels
        # self.data_args.image_processor.min_pixels = data_args.min_pixels
        # self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        # self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels
        self.get_rope_index = get_rope_index_25

        self.handles = []
        for annotation_path in annotation_paths:
            assert annotation_path.endswith(".jsonl"), (
                "Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            )
            logger.warning(
                f"Load {annotation_path}. Please ensure its last line stores the seek indices..."
            )
            # seeks = json.loads(readlastline('/code/chr/download/Live-WhisperX-526K/live_whisperx_526k_with_seeks.jsonl'))
            self.handles = annotation_path
            with open(annotation_path, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            self.len = line_count
            logger.warning(f"Successfully loaded {annotation_path}")
        if ("Qwen2VL" in processor.__class__.__name__) or (
            "Qwen2_5_VL" in processor.__class__.__name__
        ):
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = (
                processor.tokenizer("<|im_start|>assistant\n<|im_end|>").input_ids
            )
        else:
            raise NotImplementedError(
                f"Video preprocess not implemented for {processor.__class__.__name__}"
            )
        self.processor = processor
        # self.with_context = with_context
        self.with_context = True
        self.initial_fps_frames = initial_fps_frames
        self.streaming_fps_frames = streaming_fps_frames
        try:
            from data.tos import tos_loader

            self.remote_loader = functools.partial(tos_loader, length_check=True)
        except:
            logger.warning(
                "Failed to initialize remote_loader. Load video locally instead."
            )

    def load_conversation(self, index):
        annotation_path = self.handles
        return read_jsonl_line_linecache(annotation_path, index)

    def preprocess_image(self, element: dict):
        if hasattr(self, "remote_loader"):
            return Image.open(self.remote_loader(element["image"]))
        return element["image"]

    def preprocess_video(self, element: dict):
        if (
            "pos" in element
        ):  # for sharegpt. implement smart_nframes and smart_resize for pil images video
            positions = [0] + element["pos"]
            nframes = smart_nframes(
                element, total_frames=len(positions) - 1, video_fps=FPS
            )
            sampler = torch.linspace(0, len(positions) - 2, nframes).round().long()
            data_bytes = self.remote_loader(
                element["video"], length_check=True, return_io=False
            )
            video = torch.stack(
                [
                    pil_to_tensor(
                        bytes_to_pil(data_bytes[positions[i] : positions[i + 1]])
                    )
                    for i in sampler
                ]
            )
            video = _spatial_resize_video(video)
            return video
        return element["video"]

    def preprocess_text(self, element: str):
        if self.with_context and ("title" in element or "previous" in element):
            previous = element.get("previous", "")
            if previous:
                title = ""
            else:
                title = element.get("title", "")
            return (element["text"] + f"\n{title}\n{previous}").strip()
        return element["text"]

    def preprocess_conversation_batch_stream(self, conversation: list):
        user_message, assistant_message = conversation
        user_content, assistant_content = (
            user_message["content"],
            assistant_message["content"],
        )
        user_video_dict, user_query_dict = user_content
        assert "video" in user_video_dict, (
            "please check your data, ensure the video info in the first user content"
        )
        assistant_text_stream = assistant_message["content"][0]["text_stream"]

        # load video in strict fps
        video_start = user_video_dict["video_start"]
        video_end = user_video_dict["video_end"]
        user_video_dict["video_end"] = (
            user_video_dict["video_end"] - user_video_dict["video_start"]
        )
        user_video_dict["video_start"] = 0
        # print(user_video_dict['video_end'])
        # print(user_video_dict['video_start'])
        try:
            clip, _, clip_pts = _read_video_decord_plus(
                user_video_dict, return_pts=True, strict_fps=True
            )
        except:
            video_start = user_video_dict["video_start"]
            video_end = user_video_dict["video_end"]
            duration = video_end - video_start
            n_frames = int(duration * 2)
            H, W = 224, 224
            epsilon = 1e-4
            fake_clip = np.full((n_frames, 3, H, W), 1.0 - epsilon, dtype=np.float32)
            fake_clip_tensor = torch.from_numpy(fake_clip)
            fake_pts = np.linspace(video_start, video_end, n_frames).tolist()
            clip = fake_clip_tensor
            clip_pts = fake_pts
        clip = _spatial_resize_video(clip)
        user_video_dict["video_start"] = video_start
        user_video_dict["video_end"] = video_end

        # make conversation
        start_timestamp, end_timestamp = 0, self.initial_fps_frames / FPS
        # phrase, next_start_from = get_phrase_before_timestamp(assistant_text_stream, clip_pts[self.initial_fps_frames - 1],video_start)
        phrase, next_start_from = get_phrase_before_timestamp(
            assistant_text_stream, clip_pts[-1], video_start
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    user_query_dict,
                    {
                        "type": "video",
                        "video": clip,
                    },  # {'type': 'video', 'video': clip[:self.initial_fps_frames]},
                ],
            },
        ]
        conversation_answer = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": phrase}],
            }  # ' ...' denotes the streaming is not ended
        ]

        frames_list = [clip]

        while conversation_answer[-1]["content"][0]["text"] == " ...":
            conversation = conversation[:-1]
            conversation_answer = conversation_answer[:-1]
            frames_list = frames_list[:-1]
        conversation = conversation + conversation_answer
        return conversation, frames_list

    def preprocess_conversation_group_stream(self, conversation: list):
        user_message, assistant_message = conversation
        user_content, assistant_content = (
            user_message["content"],
            assistant_message["content"],
        )
        user_video_dict, user_query_dict = user_content
        assert "video" in user_video_dict, (
            "please check your data, ensure the video info in the first user content"
        )
        assistant_text_stream = assistant_message["content"][0]["text_stream"]

        # load video in strict fps
        video_start = user_video_dict["video_start"]
        video_end = user_video_dict["video_end"]
        user_video_dict["video_end"] = (
            user_video_dict["video_end"] - user_video_dict["video_start"]
        )
        user_video_dict["video_start"] = 0
        # print(user_video_dict['video_end'])
        # print(user_video_dict['video_start'])
        try:
            clip, _, clip_pts = _read_video_decord_plus(
                user_video_dict, return_pts=True, strict_fps=True
            )
        except:
            video_start = user_video_dict["video_start"]
            video_end = user_video_dict["video_end"]
            duration = video_end - video_start
            n_frames = int(duration * 2)
            H, W = 224, 224
            epsilon = 1e-4
            fake_clip = np.full((n_frames, 3, H, W), 1.0 - epsilon, dtype=np.float32)
            fake_clip_tensor = torch.from_numpy(fake_clip)
            fake_pts = np.linspace(video_start, video_end, n_frames).tolist()
            clip = fake_clip_tensor
            clip_pts = fake_pts
        clip = _spatial_resize_video(clip)
        user_video_dict["video_start"] = video_start
        user_video_dict["video_end"] = video_end

        # make conversation
        start_timestamp, end_timestamp = 0, self.initial_fps_frames / FPS
        user_query_dict2 = copy.deepcopy(user_query_dict)

        previous = user_query_dict2.get("previous", "")
        if previous:
            title = ""
        else:
            title = user_query_dict2.get("title", "")
        user_query_dict2["text"] = f"\n{title}\n{previous}".strip()
        user_query_dict2["text"] = f"{user_query_dict2['text']}\n"
        phrase, next_start_from = get_phrase_before_timestamp(
            assistant_text_stream, clip_pts[self.initial_fps_frames - 1], video_start
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    user_query_dict,
                    # {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'},
                    {"type": "video", "video": clip[: self.initial_fps_frames]},
                ],
            },
        ]
        conversation_answer = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": phrase + " ..."}],
            }  # ' ...' denotes the streaming is not ended
        ]

        frames_list = [clip[: self.initial_fps_frames]]
        for i in range(self.initial_fps_frames, len(clip), self.streaming_fps_frames):
            start_timestamp, end_timestamp = (
                i / FPS,
                (i + self.streaming_fps_frames) / FPS,
            )
            phrase, next_start_from = get_phrase_before_timestamp(
                assistant_text_stream,
                clip_pts[i + self.streaming_fps_frames - 1],
                video_start,
                start_from=next_start_from,
            )
            frames = clip[i : i + self.streaming_fps_frames]
            conversation.extend(
                [
                    {
                        "role": "user",
                        "content": [
                            # {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'},
                            {"type": "video", "video": frames},
                        ],
                    },
                ]
            )
            conversation_answer.extend(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": phrase + " ..."}],
                    }  # ' ...' denotes the streaming is not ended
                ]
            )
            frames_list.append(frames)
        # remove the last with no phrase

        while conversation_answer[-1]["content"][0]["text"] == " ...":
            conversation = conversation[:-1]
            conversation_answer = conversation_answer[:-1]
            frames_list = frames_list[:-1]
        # conversation.extend([
        #     {
        #         'role': 'user', 'content': [
        #             user_query_dict2,
        #         ]
        #     },
        # ])
        conversation = conversation + conversation_answer
        return conversation, frames_list

    def preprocess_conversation_stream(self, conversation: list):
        user_message, assistant_message = conversation
        user_content, assistant_content = (
            user_message["content"],
            assistant_message["content"],
        )
        user_video_dict, user_query_dict = user_content
        assert "video" in user_video_dict, (
            "please check your data, ensure the video info in the first user content"
        )
        assistant_text_stream = assistant_message["content"][0]["text_stream"]

        # load video in strict fps
        video_start = user_video_dict["video_start"]
        video_end = user_video_dict["video_end"]
        user_video_dict["video_end"] = (
            user_video_dict["video_end"] - user_video_dict["video_start"]
        )
        user_video_dict["video_start"] = 0
        # print(user_video_dict['video_end'])
        # print(user_video_dict['video_start'])
        try:
            clip, _, clip_pts = _read_video_decord_plus(
                user_video_dict, return_pts=True, strict_fps=True
            )
        except:
            video_start = user_video_dict["video_start"]
            video_end = user_video_dict["video_end"]
            duration = video_end - video_start
            n_frames = int(duration * 2)
            H, W = 224, 224
            epsilon = 1e-4
            fake_clip = np.full((n_frames, 3, H, W), 1.0 - epsilon, dtype=np.float32)
            fake_clip_tensor = torch.from_numpy(fake_clip)
            fake_pts = np.linspace(video_start, video_end, n_frames).tolist()
            clip = fake_clip_tensor
            clip_pts = fake_pts
        clip = _spatial_resize_video(clip)
        user_video_dict["video_start"] = video_start
        user_video_dict["video_end"] = video_end

        # make conversation
        start_timestamp, end_timestamp = 0, self.initial_fps_frames / FPS

        phrase, next_start_from = get_phrase_before_timestamp(
            assistant_text_stream, clip_pts[self.initial_fps_frames - 1], video_start
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Time={start_timestamp:.1f}-{end_timestamp:.1f}s",
                    },
                    {"type": "video", "video": clip[: self.initial_fps_frames]},
                    user_query_dict,
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": phrase + " ..."}],
            },  # ' ...' denotes the streaming is not ended
        ]
        frames_list = [clip[: self.initial_fps_frames]]
        for i in range(self.initial_fps_frames, len(clip), self.streaming_fps_frames):
            start_timestamp, end_timestamp = (
                i / FPS,
                (i + self.streaming_fps_frames) / FPS,
            )
            phrase, next_start_from = get_phrase_before_timestamp(
                assistant_text_stream,
                clip_pts[i + self.streaming_fps_frames - 1],
                video_start,
                start_from=next_start_from,
            )
            frames = clip[i : i + self.streaming_fps_frames]
            conversation.extend(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Time={start_timestamp:.1f}-{end_timestamp:.1f}s",
                            },
                            {"type": "video", "video": frames},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": phrase + " ..."}],
                    },  # ' ...' denotes the streaming is not ended
                ]
            )
            frames_list.append(frames)
        # remove the last with no phrase
        while conversation[-1]["content"][0]["text"] == " ...":
            conversation = conversation[:-2]
            frames_list = frames_list[:-1]
        return conversation, frames_list

    def getitem(self, index):
        conversation = self.load_conversation(index)

        special_process_for_stream, image_inputs, video_inputs = False, None, None
        for message in conversation:
            if message["role"] == "user":
                for element in message["content"]:
                    if hasattr(self, "remote_loader"):
                        element["remote_loader"] = self.remote_loader
                    modal = element["type"]
                    element[modal] = getattr(self, f"preprocess_{modal}")(element)
                    if isinstance(element[modal], torch.Tensor):
                        if video_inputs is None:
                            video_inputs = [element[modal]]
                        else:
                            video_inputs.append(element[modal])
            else:
                for element in message["content"]:
                    special_process_for_stream = "text_stream" in element
                    break
        if special_process_for_stream:
            # conversation, video_inputs = self.preprocess_conversation_stream(conversation)
            conversation, video_inputs = self.preprocess_conversation_batch_stream(
                conversation
            )
            image_inputs = None
        else:
            if not video_inputs and not image_inputs:
                image_inputs, video_inputs = process_vision_info(conversation)
        texts = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            returns_tensors="pt",
        )
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            do_rescale=False,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids

        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            input_ids,
            video_grid_thw=inputs.data["video_grid_thw"],
        )
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (sample_idx, im_end_idx) in zip(
            im_start_idxs, im_end_idxs
        ):
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                labels[sample_idx, im_start_idx + 3 : im_end_idx + 1] = input_ids[
                    sample_idx, im_start_idx + 3 : im_end_idx + 1
                ]
        inputs["labels"] = labels
        inputs["position_ids"] = position_ids
        return inputs

    def __getitem__(self, index):
        return self.getitem(index)

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1
        return batched_inputs[0]

    def __len__(self):
        return self.len


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    processor = AutoProcessor.from_pretrained(
        "/code/chr/download/Qwen2.5-VL-7B-Instruct", padding_side="right"
    )
    train_dataset = LMMDataset(
        annotation_paths=[
            "/code/chr/download/Live-WhisperX-526K/short_filtered_3token.jsonl",
            # 'llava_video_178k_with_seeks.jsonl',
            # 'llava_hound_video_with_seeks.jsonl',
            # 'llava_ov_multi_image_with_seeks.jsonl',
            # 'llava_ov_single_image_text_mix_with_seeks.jsonl'
        ],
        processor=processor,
        data_args=data_args,
    )
    data_collator = train_dataset.data_collator
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "/code/chr/download/Qwen2.5-VL-7B-Instruct", padding_side="right"
    )
    # model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B', torch_dtype='auto', attn_implementation='flash_attention_2', device_map='cuda')
    # model.to('cuda')

    dataset = LMMDataset(
        annotation_paths=[
            "/code/chr/download/Live-WhisperX-526K/short_filtered_3token.jsonl",
            # 'llava_video_178k_with_seeks.jsonl',
            # 'llava_hound_video_with_seeks.jsonl',
            # 'llava_ov_multi_image_with_seeks.jsonl',
            # 'llava_ov_single_image_text_mix_with_seeks.jsonl'
        ],
        processor=processor,
    )
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=dataset.data_collator,
    )

    for batch in tqdm.tqdm(dataloader):
        pass
    # for i in tqdm.tqdm(range(len(dataset))):
    #     conversation = dataset.__getitem__(i)
    # inputs.to('cuda')
    # with torch.inference_mode():
    #     model(**inputs)
