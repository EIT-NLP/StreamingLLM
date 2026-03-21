import torch

IM_START = 151644
IM_END = 151645
VISION_START = 151652
VISION_END = 151653
VIDEO_TOKEN = 151656
ASSISTANT_ID = 77091
SYSTEM_ID = 8948
SKIP_ID = 151665
THINK_END = 151667
# Token          ID
# -----------------------
# <skip>        151665
# <think>       151666
# </think>       151667
# <answer>       151668
# </answer>      151669


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
    dtype=torch.float32,
    debug=False,
    use_random_wait_k=False,
    use_random_frame_chunk=False,
):
    """
    Build causal mask for frame-to-sentence mapping:
    [frame_group_0] -> [sentence_0]\n
    [frame_group_0,1] -> [sentence_1]\n
    ...
    [all frames] -> [final_sentence] (no \n required)

    All "sentences" are delimited by THINK_END, last one may not have it.
    """
    B, L = input_ids.shape
    device = input_ids.device
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((B, 1, L, L), min_dtype, device=device)
    all_debug_info = []

    for b in range(B):
        debug_info_for_sample = {}
        ids = input_ids[b]
        segments = parse_segments(ids)

        P, QV, A_list = None, None, []
        for s, e in segments:
            tokens = ids[s : e + 1]
            if SYSTEM_ID in tokens:
                P = (s, e)
                debug_info_for_sample["Prompt"] = P
            elif ASSISTANT_ID in tokens:
                A_list.append((s + 3, e))
                debug_info_for_sample["Assistant Preamble"] = (s, s + 2)
            elif QV is None:
                QV = (s, e)

        assert P is not None and QV is not None and len(A_list) > 0, (
            "Missing required segments"
        )
        a_s, a_e = A_list[0]

        QV_tokens = ids[QV[0] : QV[1] + 1]
        vision_start_pos = (QV_tokens == VISION_START).nonzero(as_tuple=True)[0].item()
        vision_end_pos = (QV_tokens == VISION_END).nonzero(as_tuple=True)[0].item()

        debug_info_for_sample["Question"] = (QV[0] + 1, QV[0] + vision_start_pos - 1)

        vision_token_start = QV[0] + vision_start_pos + 1
        vision_token_end = QV[0] + vision_end_pos - 1
        total_video_token_len = vision_token_end - vision_token_start + 1

        N_token_per_frame = video_frame_token_counts[b]
        num_frames = total_video_token_len // N_token_per_frame

        vision_frame_spans = []
        curr = vision_token_start
        if use_random_frame_chunk:
            remaining = num_frames
            while remaining > 0:
                max_chunk = 4
                upper = min(max_chunk, remaining)
                frames_in_chunk = random.randint(1, upper)
                num_tokens = frames_in_chunk * N_token_per_frame
                v_s = int(curr)
                v_e = int(curr + num_tokens - 1)
                vision_frame_spans.append((v_s, v_e))
                curr += num_tokens
                remaining -= frames_in_chunk
        else:
            for i in range(num_frames):
                v_s = int(curr)
                v_e = int(curr + N_token_per_frame - 1)
                vision_frame_spans.append((v_s, v_e))
                curr += N_token_per_frame

        for i, (vs, ve) in enumerate(vision_frame_spans):
            debug_info_for_sample[f"Video Group {i}"] = (vs, ve)

        if debug:
            print(f"\n=== [Batch {b}] ===")
            print(f"P: {P}")
            print(f"QV: {QV}")
            print(f"A_list: {A_list}")

        sentence_end_positions = (ids[a_s : a_e + 1] == THINK_END).nonzero(
            as_tuple=True
        )[0]
        sentence_end_positions = (sentence_end_positions + a_s).tolist()

        sentence_spans = []
        start = a_s
        for end_pos in sentence_end_positions:
            sentence_spans.append((start, end_pos))
            start = end_pos + 1

        if start <= a_e:
            sentence_spans.append((start, a_e))

        num_sentence_groups = len(sentence_spans)
        num_vision_groups = len(vision_frame_spans)

        if debug:
            print(f"\n=== [Batch {b}] ===")
            print(f"Vision Groups: {num_vision_groups}")
            for i, (vs, ve) in enumerate(vision_frame_spans):
                n_frames = (ve - vs + 1) // N_token_per_frame
                print(f"  Video Group {i}: [{vs}, {ve}] ({n_frames} frames)")
            print(f"Sentence Groups: {num_sentence_groups}")
            for i, (ss, se) in enumerate(sentence_spans):
                print(f"  Sentence {i}: [{ss}, {se}] (len={se - ss + 1})")

        for i, (ss, se) in enumerate(sentence_spans):
            debug_info_for_sample[f"Sentence Group {i}"] = (ss, se)

        global_prefix = list(range(P[0], P[1] + 1))
        global_prefix.append(vision_token_start - 1)
        global_prefix.append(QV[0] - 1)

        q_tokens_start = QV[0]
        q_tokens_end = vision_token_start - 2
        if q_tokens_end >= q_tokens_start:
            global_prefix.extend(range(q_tokens_start, q_tokens_end + 1))

        causal_mask[b, 0] = torch.triu(
            torch.full((L, L), min_dtype, device=device), diagonal=1
        )

        for group_idx in range(num_vision_groups):
            s_s, s_e = sentence_spans[group_idx]

            allowed = global_prefix.copy()

            for j in range(group_idx + 1):
                fv_s, fv_e = vision_frame_spans[j]
                allowed.extend(range(fv_s, fv_e + 1))
                if j == num_vision_groups - 1:
                    allowed.extend([fv_e + 1, fv_e + 2, fv_e + 3])

            for j in range(group_idx):
                prev_s, prev_e = sentence_spans[j]
                allowed.extend(range(prev_s, prev_e + 1))

            allowed.extend([A_list[0][0] - 3, A_list[0][0] - 2, A_list[0][0] - 1])

            for t in range(s_s, s_e + 1):
                allowed.append(t)
                causal_mask[b, 0, t, :] = min_dtype
                causal_mask[b, 0, t, allowed] = 0.0

        all_debug_info.append(debug_info_for_sample)

        if len(vision_frame_spans) > 1:
            assistant_prefix = vision_frame_spans[1][0]
        else:
            assistant_prefix = vision_frame_spans[0][1] + 1

        prefix_span = QV[1] + 5

        causal_mask[
            b, 0, prefix_span - 3 : prefix_span, assistant_prefix : prefix_span - 3
        ] = min_dtype

        causal_mask[b, 0, prefix_span:-1, : prefix_span - 3] = causal_mask[
            b, 0, prefix_span + 1 :, : prefix_span - 3
        ]

    if debug:
        return causal_mask, all_debug_info
    else:
        return causal_mask


# def save_causal_mask_as_image(
#     causal_mask: torch.Tensor, debug_info: dict, save_path: str = "example.png"
# ):
#     """

#     """
#     fig, ax = plt.subplots(figsize=(16, 16))
#     L = causal_mask.shape[-1]


#     sns.heatmap(

#         cmap="viridis",
#         ax=ax,
#         linewidth=0.5,
#         cbar=False,
#     )


#     step = 2
#     tick_positions = np.arange(0, L, step)
#     ax.set_xticks(tick_positions + 0.5)
#     ax.set_yticks(tick_positions + 0.5)
#     ax.set_xticklabels(tick_positions, fontsize=16)
#     ax.set_yticklabels(tick_positions, fontsize=16)
#     ax.tick_params(axis="x", length=15, width=2, labelsize=16, pad=10)
#     ax.tick_params(axis="y", length=15, width=2, labelsize=16, pad=10)


#     colors = {
#         "Prompt": "cyan",
#         "Question": "magenta",
#         "Video": "black",
#         "Sentence": "green",
#         "Assistant Preamble": "blue",
#     }


#     for key, (start, end) in debug_info.items():
#         width = end - start + 1
#         color = "white"

#         if "Prompt" in key:
#             color = colors["Prompt"]
#         elif "Question" in key:
#             color = colors["Question"]
#         elif "Video" in key:
#             color = colors["Video"]
#         elif "Sentence" in key:
#             color = colors["Sentence"]
#         elif "Preamble" in key:
#             color = colors["Assistant Preamble"]


#         rect_y = patches.Rectangle(
#             (-0.5, start),
#             0.5,
#             width,
#             linewidth=2,
#             edgecolor=color,
#             facecolor=color,
#             clip_on=False,
#         )
#         ax.add_patch(rect_y)


#         rect_x = patches.Rectangle(
#             (start, L),
#             width,
#             0.5,
#             linewidth=2,
#             edgecolor=color,
#             facecolor=color,
#             clip_on=False,
#         )
#         ax.add_patch(rect_x)


#         if "Sentence Group" in key:
#             try:
#                 group_idx = int(key.split(" ")[-1])

#                 visible_vision_groups = [
#                     f"Video Group {j}"
#                     for j in range(group_idx + 1)
#                     if f"Video Group {j}" in debug_info
#                 ]
#                 if not visible_vision_groups:
#                     continue

#                 first_v_start = debug_info[visible_vision_groups[0]][0]
#                 last_v_end = debug_info[visible_vision_groups[-1]][1]
#                 v_width = last_v_end - first_v_start + 1


#                 rect = patches.Rectangle(
#                     (first_v_start, start),
#                     v_width,
#                     width,
#                     linewidth=2,
#                     edgecolor=colors["Sentence"],
#                     facecolor="none",
#                     linestyle="--",
#                     alpha=0.8,
#                 )
#                 ax.add_patch(rect)
#             except Exception as e:
#                 print(f"Warning: failed to draw alignment for {key}: {e}")


#     legend_patches = [patches.Patch(color=c, label=n) for n, c in colors.items()]
#     fig.legend(
#         handles=legend_patches,
#         loc="lower center",
#         ncol=len(legend_patches),
#         bbox_to_anchor=(0.5, -0.02),
#         fontsize=16,
#         title="Token Types",
#         title_fontsize=18,
#     )


#     plt.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95)
#     plt.savefig(save_path, bbox_inches="tight", pad_inches=0.3, dpi=200)
#     plt.show()
#     print(f"Mask saved to: {save_path}")


# # ==============================

# # ==============================


# vision_tokens = (
#     [VISION_START] + [VIDEO_TOKEN] * 15 + [VISION_END]


# input_ids = [
#     # System
#     IM_START,
#     *system_tokens,
#     IM_END,
#     # User (QV)
#     IM_START,
#     *question_tokens,
#     *vision_tokens,
#     IM_END,

#     IM_START,  #  <|im_start|>
#     77091,  # assistant
#     198,  # \n
#     *sentence_3,
#     *sentence_0,
#     *sentence_1,
#     *sentence_2,
#     # *sentence_4,
#     *answer_tokens,
#     IM_END,
# ]


# input_ids = torch.tensor([input_ids])

# print(f"Input IDs shape: {input_ids.shape}")
# print(f"Total length: {input_ids.shape[-1]}")


# causal_mask, debug_info_list = build_group_custom_causal_mask(
#     input_ids=input_ids,
#     video_frame_token_counts=video_frame_token_counts,
#     dtype=torch.float32,
#     debug=True,


# )


# debug_info = debug_info_list[0]

# save_path = "frame_to_sentence_mask.png"
# save_causal_mask_as_image(causal_mask, debug_info, save_path=save_path)
