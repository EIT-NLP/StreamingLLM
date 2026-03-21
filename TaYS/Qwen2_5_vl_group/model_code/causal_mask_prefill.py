import torch


def save_causal_mask_as_image(
    causal_mask: torch.Tensor, index: int = 0, save_path: str = "example.png"
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        causal_mask[0, 0],
        cmap="viridis",
        linewidth=0.005,
    )
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.show()


IM_START = 151644
IM_END = 151645
VISION_START = 151652
VISION_END = 151653
VIDEO_TOKEN = 151656
ASSISTANT_ID = 77091
SYSTEM_ID = 8948


def parse_segments(input_ids_1d):
    """Return list of (start, end) for each <|im_start|>...<|im_end|> block."""
    im_start_idxs = (input_ids_1d == IM_START).nonzero(as_tuple=True)[0]
    im_end_idxs = (input_ids_1d == IM_END).nonzero(as_tuple=True)[0]
    assert len(im_start_idxs) == len(im_end_idxs), (
        "Mismatched <|im_start|> and <|im_end|> counts"
    )
    return list(zip(im_start_idxs.tolist(), im_end_idxs.tolist()))


def build_group_custom_causal_mask(
    input_ids: torch.Tensor,
    video_frame_token_counts: list,  # [B]
    ANSWER_GROUP_SIZE=3,
    token_len_schedule=None,
    dtype=torch.float32,
    debug=False,
):
    """
    Build custom causal mask for group-based video-answer mapping, per-sample frame token counts.

    Args:
        input_ids: (B, L)
        video_frame_token_counts: List[int], 每个样本 "每帧 token 数"
    Returns:
        causal_mask: (B, 1, L, L)
    """
    B, L = input_ids.shape
    device = input_ids.device
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((B, 1, L, L), min_dtype, device=device)

    for b in range(B):
        ids = input_ids[b]
        # segments = parse_segments(ids)
        im_start_idxs = (ids == IM_START).nonzero(as_tuple=True)[0]
        visual_start_idxs = (ids == VISION_START).nonzero(as_tuple=True)[0]
        P = (im_start_idxs[0], im_start_idxs[1] - 1)
        QV = (im_start_idxs[1], im_start_idxs[2] - 1)
        visual_token_len = im_start_idxs[2] - 1 - visual_start_idxs[0]
        A = ids[im_start_idxs[2] :]
        A_list = []
        if token_len_schedule is not None and len(token_len_schedule) > 0:
            # ------------------------------------------------------------
            # token_len_schedule = [len_ans1, len_ans2, ...]

            # ------------------------------------------------------------
            offset = 3
            A_list = []

            for ans_len in token_len_schedule:
                if offset >= len(A):
                    break
                start = im_start_idxs[2] + offset
                end = im_start_idxs[2] + min(offset + ans_len - 1, len(A) - 1)
                A_list.append((start, end))
                offset += ans_len

        else:
            for i in range(3, len(A), ANSWER_GROUP_SIZE):
                A_list.append(
                    (
                        im_start_idxs[2] + i,
                        im_start_idxs[2] + min(i + ANSWER_GROUP_SIZE - 1, len(A) - 1),
                    )
                )

        causal_mask[b, 0] = torch.triu(
            torch.full((L, L), min_dtype, device=device), diagonal=1
        )
        if len(A_list) < 1:
            continue
        frame_num = int(visual_token_len / video_frame_token_counts[b])
        vision_frame_spans = []
        start_idx = visual_start_idxs[0] + 1
        frame_token_count = video_frame_token_counts[b]
        for i in range(frame_num):
            s = start_idx + i * frame_token_count
            e = s + frame_token_count - 1
            vision_frame_spans.append((s, e))
        vision_frame_spans[0] = (vision_frame_spans[0][0] - 1, vision_frame_spans[0][1])
        if VISION_END in ids:
            vision_frame_spans[-1] = (
                vision_frame_spans[-1][0],
                vision_frame_spans[-1][1] + 3,
            )
        A_list[0] = (A_list[0][0] - 3, A_list[0][1])
        for group_idx, (a_s, a_e) in enumerate(A_list):
            allowed = list(range(P[0], P[1] + 1))

            q_tokens_end = visual_start_idxs[0] - 1
            allowed += list(range(QV[0], q_tokens_end + 1))

            for i in range(group_idx + 1):
                if i < len(vision_frame_spans):
                    vs, ve = vision_frame_spans[i]
                    allowed += list(range(vs, ve + 1))

            if group_idx > 0:
                prev_a_start = A_list[0][0]
                allowed += list(range(prev_a_start, a_s))

            for t in range(a_s, a_e + 1):
                allowed.append(t)
                causal_mask[b, 0, t, :] = min_dtype
                causal_mask[b, 0, t, allowed] = 0.0

        if len(A_list) >= 1:
            assitant_prefix = vision_frame_spans[1][0]
            prefix_span = A_list[0][0] + 3
            # for assistant tokens
            # causal_mask[
            #     b, 0, prefix_span - 3 : prefix_span, assitant_prefix : prefix_span - 3
            # ] = min_dtype
            # for answer tokens
            causal_mask[b, 0, prefix_span:-1, : prefix_span - 3] = causal_mask[
                b, 0, prefix_span + 1 :, : prefix_span - 3
            ]
            causal_mask[0, 0, -1] = 0

    return causal_mask


# Token          ID
# -----------------------
# <skip>        151665
# <think>       151666
# </think>       151667
# <answer>       151668
# </answer>      151669


# input_ids = torch.tensor([[  # (B=1, L=...)
#     151644, 8948, 1001, 1002, 151645,  # P
#     151644, 2001, 2002, 1003,          # Q
#     151652,  # VISION_START
#     151656, 151656, 151656, 151656, 151656,   # Frame 1 (5 token)
#     # 151656, 151656, 151656, 151656, 151656,   # Frame 2 (5 token)
#     # 151656, 151656, 151656, 151656, 151656,   # Frame 3 (5 token)
#     # 151656, 151656, 151656, 151656, 151656,   # Frame 4 (5 token)
#     # 151653,  # VISION_END
#     # 151645,  # end of QV
#     151644, 77091, 198#,3001, 3002, 3003,3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 151645
#     # 151644, 77091, 198, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 151645  # A (11 tokens)
# ]])


# input_ids = torch.tensor([[  # (B=1, L=...)
#     151644, 8948, 1001, 1002, 151645,  # P
#     151644, 2001, 2002, 1003,          # Q
#     151652,  # VISION_START
#     151656, 151656, 151656, 151656, 151656,   # Frame 1 (5 token)
#     151656, 151656, 151656, 151656, 151656,   # Frame 2 (5 token)
#     # 151656, 151656, 151656, 151656, 151656,   # Frame 3 (5 token)
#     # 151656, 151656, 151656, 151656, 151656,   # Frame 4 (5 token)
#     # 151653,  # VISION_END
#     # 151645,  # end of QV
#     151644, 77091, 198,3001#, 3002, 3003#,3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 151645
#     # 151644, 77091, 198, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 151645  # A (11 tokens)
# ]])


# input_ids = torch.tensor([[  # (B=1, L=...)
#     151644, 8948, 1001, 1002, 151645,  # P
#     151644, 2001, 2002, 1003,          # Q
#     151652,  # VISION_START
#     151656, 151656, 151656, 151656, 151656,   # Frame 1 (5 token)
#     151656, 151656, 151656, 151656, 151656,   # Frame 2 (5 token)
#     151656, 151656, 151656, 151656, 151656,   # Frame 3 (5 token)
#     # 151656, 151656, 151656, 151656, 151656,   # Frame 4 (5 token)
#     # 151653,  # VISION_END
#     # 151645,  # end of QV
#     151644, 77091, 198,3001, 3002, 3003#,3004, 3005, 3006#, 3007, 3008, 3009, 3010, 3011, 151645
#     # 151644, 77091, 198, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 151645  # A (11 tokens)
# ]])


# input_ids = torch.tensor([[  # (B=1, L=...)
#     151644, 8948, 1001, 1002, 151645,  # P
#     151644, 2001, 2002, 1003,          # Q
#     151652,  # VISION_START
#     151656, 151656, 151656, 151656, 151656,   # Frame 1 (5 token)
#     151656, 151656, 151656, 151656, 151656,   # Frame 2 (5 token)
#     151656, 151656, 151656, 151656, 151656,   # Frame 3 (5 token)
#     151656, 151656, 151656, 151656, 151656,   # Frame 4 (5 token)
#     # 151653,  # VISION_END
#     # 151645,  # end of QV
#     # 198,
#     151644, 77091, 198,3001, 3002, 3003,3004, 3005, 3006#, 3007, 3008, 3009, 3010, 3011, 151645
#     # 151644, 77091, 198, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 151645  # A (11 tokens)
# ]])


# input_ids = torch.tensor([[  # (B=1, L=...)
#     151644, 8948, 1001, 1002, 151645,  # P
#     151644, 2001, 2002, 1003,          # Q
#     151652,  # VISION_START
#     151656, 151656, 151656, 151656, 151656,   # Frame 1 (5 token)
#     151656, 151656, 151656, 151656, 151656,   # Frame 2 (5 token)
#     151656, 151656, 151656, 151656, 151656,   # Frame 3 (5 token)
#     151656, 151656, 151656, 151656, 151656,   # Frame 4 (5 token)
#     151653,  # VISION_END
#     151645,  # end of QV

#     151644, 77091, 198,
#     3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012,3013,3014,3015,3016,3017,1,2,3,4,5,151645  # A (11 tokens)
# ]])


# video_frame_token_counts = [
#     5  # batch 0
# ]


# causal_mask = build_group_custom_causal_mask(
#     input_ids,
#     video_frame_token_counts,
#     ANSWER_GROUP_SIZE=3,
#     token_len_schedule=[1, 2, 3],
#     debug=True,
# )


# save_path = "eval_step4.png"
# save_causal_mask_as_image(causal_mask, index=0, save_path=save_path)
