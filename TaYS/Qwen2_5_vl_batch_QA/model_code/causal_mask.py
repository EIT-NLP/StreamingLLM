import torch

IM_START = 151644
IM_END = 151645
VISION_START = 151652
VISION_END = 151653
VIDEO_TOKEN = 151656
ASSISTANT_ID = 77091  # assistant token
SYSTEM_ID = 8948  # system token

import os

import matplotlib.pyplot as plt


def save_causal_mask_as_image(
    causal_mask: torch.Tensor, index: int = 0, save_path: str = "example.png"
):
    """
    Visualize and save causal mask as a grayscale image.

    Args:
        causal_mask: Tensor of shape (B, 1, L, L)
        index: which batch index to visualize
        save_path: where to save the image
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mask = causal_mask[index, 0]  # (L, L)

    vis_mask = (mask != 0).float().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(vis_mask, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"✅ Saved causal mask image to: {save_path}")


def parse_segments(input_ids_1d):
    """Return list of (start, end) for each <|im_start|>...<|im_end|> block."""
    im_start_idxs = (input_ids_1d == IM_START).nonzero(as_tuple=True)[0]
    im_end_idxs = (input_ids_1d == IM_END).nonzero(as_tuple=True)[0]
    assert len(im_start_idxs) == len(im_end_idxs), (
        "Mismatched <|im_start|> and <|im_end|> counts"
    )
    return list(zip(im_start_idxs.tolist(), im_end_idxs.tolist()))


def categorize_segments(input_ids_1d, segments):
    """Return (P, Q, [V], [A]) with assistant_id used for identifying answer blocks"""
    P, Q, V_list, A_list = None, None, [], []
    for s, e in segments:
        tokens = input_ids_1d[s : e + 1]
        if VISION_START in tokens and VISION_END in tokens:
            V_list.append((s, e))
        elif SYSTEM_ID in tokens:
            P = (s, e)
        elif ASSISTANT_ID in tokens:
            A_list.append((s, e))
        elif Q is None:
            Q = (s, e)
    return P, Q, V_list, A_list


def build_batch_custom_causal_mask(input_ids: torch.Tensor, dtype=torch.float32):
    """
    Args:
        input_ids: (B, L) tensor
    Returns:
        causal_mask: (B, 1, L, L) tensor, -inf for masked, 0 for visible
    """
    B, L = input_ids.shape
    device = input_ids.device
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((B, 1, L, L), min_dtype, device=device)

    for b in range(B):
        ids = input_ids[b]
        segments = parse_segments(ids)
        P, Q, V_list, A_list = categorize_segments(ids, segments)

        causal_mask[b, 0] = torch.triu(
            torch.full((L, L), min_dtype, device=device), diagonal=1
        )

        for i, (a_s, a_e) in enumerate(A_list):
            allowed = list(range(P[0], P[1] + 1)) + list(range(Q[0], Q[1] + 1))
            for v in V_list[: i + 1]:
                allowed += list(range(v[0], v[1] + 1))
            for prev_a in A_list[:i]:
                allowed += list(range(prev_a[0], prev_a[1] + 1))
            for t in range(a_s, a_e + 1):
                allowed.append(t)
                causal_mask[b, 0, t, :] = min_dtype
                causal_mask[b, 0, t, allowed] = 0.0

    return causal_mask


import torch

# Constants for special tokens
IM_START = 151644
IM_END = 151645
VISION_START = 151652
VISION_END = 151653
VIDEO_TOKEN = 151656
ASSISTANT_ID = 77091
SYSTEM_ID = 8948


def build_inference_causal_mask(
    input_ids: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor,
    config,
    dtype=torch.float32,
):
    """
    Build a causal mask that reflects interleave layout during inference.
    Only applies to the current decoding step (sequence_length == 1).
    """
    batch_size, target_length = attention_mask.shape
    sequence_length = 1  # inference step is token-by-token
    device = input_ids.device
    min_dtype = torch.finfo(dtype).min

    # Determine causal mask dimensions
    if past_key_values is not None and hasattr(past_key_values, "get_max_cache_shape"):
        total_kv_len = past_key_values.get_max_cache_shape()
    else:
        total_kv_len = target_length

    causal_mask = torch.full(
        (batch_size, 1, sequence_length, total_kv_len),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )

    # Update attention based on streaming rule: assistant_t attends to P + Q + V_1..V_t
    for b in range(batch_size):
        ids = input_ids[b]  # 1D sequence
        cache_pos = cache_position[b]  # scalar

        # use entire context window to check alignment
        full_ids = ids[: cache_pos.item() + 1]  # accumulated input

        # segment parsing
        def parse_segments(input_ids_1d):
            im_start_idxs = (input_ids_1d == IM_START).nonzero(as_tuple=True)[0]
            im_end_idxs = (input_ids_1d == IM_END).nonzero(as_tuple=True)[0]
            return list(zip(im_start_idxs.tolist(), im_end_idxs.tolist()))

        def categorize_segments(input_ids_1d, segments):
            P, Q, V_list, A_list = None, None, [], []
            for s, e in segments:
                tokens = input_ids_1d[s : e + 1]
                if VISION_START in tokens and VISION_END in tokens:
                    V_list.append((s, e))
                elif SYSTEM_ID in tokens:
                    P = (s, e)
                elif ASSISTANT_ID in tokens:
                    A_list.append((s, e))
                elif Q is None:
                    Q = (s, e)
            return P, Q, V_list, A_list

        segments = parse_segments(full_ids)
        P, Q, V_list, A_list = categorize_segments(full_ids, segments)

        current_t = cache_pos.item()
        allow_set = set()

        # If current pos inside assistant answer
        for i, (a_s, a_e) in enumerate(A_list):
            if a_s <= current_t <= a_e:
                allow_set.update(range(P[0], P[1] + 1))
                allow_set.update(range(Q[0], Q[1] + 1))
                for v in V_list[: i + 1]:
                    allow_set.update(range(v[0], v[1] + 1))
                for prev_a in A_list[:i]:
                    allow_set.update(range(prev_a[0], prev_a[1] + 1))
                break  # no need to continue
        else:
            # not an answer token, allow standard causal
            allow_set.update(range(0, current_t + 1))

        allow_list = sorted(list(allow_set))
        causal_mask[b, 0, 0, allow_list] = 0.0

    # Optionally include padding mask
    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        padding_mask = causal_mask + attention_mask[:, None, None, :].to(device)
        causal_mask = causal_mask.masked_fill(padding_mask == 0, min_dtype)

    return causal_mask


input_ids = torch.tensor(
    [
        [  # (B=1, L=...)
            151644,
            8948,
            1001,
            1002,
            151645,  # P
            151644,
            1003,
            151652,
            151656,
            151656,
            151653,
            151645,  # V1
            151644,
            1004,
            151652,
            151656,
            151656,
            151653,
            151645,  # V2
            151644,
            1005,
            151652,
            151656,
            151656,
            151653,
            151645,  # V3
            151644,
            1006,
            151652,
            151656,
            151656,
            151653,
            151645,  # V4
            151644,
            2001,
            2002,
            151645,  # Q
            151644,
            77091,
            3001,
            3002,
            151645,  # A1
            151644,
            77091,
            3003,
            3004,
            151645,  # A2
            151644,
            77091,
            3005,
            3006,
            151645,  # A3
            151644,
            77091,
            3007,
            3008,
            151645,  # A4
        ]
    ]
)


causal_mask = build_batch_custom_causal_mask(input_ids)
save_path = "/code/Streaming_video/pe_version/Qwen2_5_vl_batch/model_code/example.png"
save_causal_mask_as_image(causal_mask, index=0, save_path=save_path)
