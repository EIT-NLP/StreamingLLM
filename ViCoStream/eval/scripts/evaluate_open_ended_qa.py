#!/usr/bin/env python3
# evaluate_open_ended_qa_refactored.py
# 封装版本：支持作为函数 evaluate_open_stream() 调用

import os
import ast
import json
import time
from tqdm import tqdm
from collections import defaultdict
from multiprocessing.pool import Pool
from openai import OpenAI


def chunk_list(lst, n_chunks):
    """Split list lst into n_chunks roughly equal parts (n_chunks >= 1)."""
    if n_chunks <= 1:
        return [lst]
    k, m = divmod(len(lst), n_chunks)
    parts = []
    i = 0
    for j in range(n_chunks):
        add = k + (1 if j < m else 0)
        if add > 0:
            parts.append(lst[i:i+add])
        else:
            parts.append([])
        i += add
    return parts


def init_client(api_key, base_url):
    global client
    client = OpenAI(api_key=api_key, base_url=base_url)



def annotate(prediction_set, caption_files, output_dir, model, sleep_per_call):
    """Evaluates question and answer pairs using GPT"""
    global client
    if client is None:
        raise RuntimeError("Client not initialized in process")

    for file in tqdm(caption_files, desc="annotate"):
        key = file[:-5]
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']

        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.002,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs "
                            "for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide only a Python dictionary string with keys 'pred' ('yes' or 'no') and 'score' (integer 0–5). "
                            "Example: {'pred': 'yes', 'score': 5}"
                        )
                    }
                ],
                timeout=100,
            )

            response_message = (
                completion["choices"][0]["message"]["content"]
                if isinstance(completion, dict)
                else completion.choices[0].message.content
            )

            
            try:
                response_dict = ast.literal_eval(response_message)
            except Exception:
                print(f"⚠️ Eval parse failed for {key}, got: {response_message}")
                continue

            result_qa_pair = [response_dict, qa_set]
            os.makedirs(output_dir, exist_ok=True)  # TODO make codes more simple
            with open(f"{output_dir}/{key}.json", "w", encoding="utf-8") as f:
                json.dump(result_qa_pair, f)

            time.sleep(sleep_per_call)

        except Exception as e:
            print(f"⚠️ Error processing file '{key}': {e}")
            time.sleep(1)



def evaluate_open_stream(pred_path, output_dir, output_json,
                         num_tasks=1, num_chunks=1,
                         api_key=None, api_base="you_base_url",
                         model="gpt-3.5", sleep_per_call=0.5):
    """
    Run open-ended QA evaluation and return summary results.
    """

    # ========== Step 1. Read prediction file(s) ==========
    pred_contents = []
    if num_chunks > 1:
        for _idx in range(num_chunks):
            file = os.path.join(pred_path, f"{num_chunks}_{_idx}.jsonl")
            if not os.path.exists(file):
                print(f"[main] Warning: file not found: {file}")
                continue
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            pred_contents.append(json.loads(line))
                        except Exception as e:
                            print(f"[main] failed parse line in {file}: {e}")
    else:
        file = os.path.join(pred_path, "pred.jsonl")
        if not os.path.exists(file):
            if os.path.isfile(pred_path):
                file = pred_path
            else:
                raise FileNotFoundError(f"No pred.jsonl found at {file}")
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pred_contents.append(json.loads(line))
                    except Exception as e:
                        print(f"[main] failed parse line in {file}: {e}")
                        

    # ========== Step 2. Normalize IDs ==========
    print(f"[main] Loaded {len(pred_contents)} predictions from {pred_path}")
    video_id_counts = {}
    new_pred_contents = []
    for sample in pred_contents:
        video_id = sample.get('id', '')
        video_id_counts[video_id] = video_id_counts.get(video_id, -1) + 1
        new_id = f"{video_id}_{video_id_counts[video_id]}"
        sample['id'] = new_id
        if 'question' not in sample and 'qustion' in sample:
            sample['question'] = sample.get('qustion', '')
        if 'pred' not in sample and 'response' in sample:
            sample['pred'] = sample.get('response', '')
        new_pred_contents.append(sample)

    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    os.makedirs(output_dir, exist_ok=True)

    # ========== Step 3. Build prediction_set ==========
    prediction_set = {
        s['id']: {"q": s.get('question', ''),
                  "a": s.get('answer', ''),
                  "pred": s.get('pred', ''),
                  "a_type": s.get('answer_type')}
        for s in new_pred_contents
    }

    # ========== Step 4. Annotate ==========
    init_client(api_key, api_base)

    incomplete_files = [f for f in caption_files if not os.path.exists(os.path.join(output_dir, f))]
    print(f"[main] Need evaluate {len(incomplete_files)}/{len(caption_files)} predictions")
    if incomplete_files:
        workers = min(num_tasks, max(1, len(incomplete_files)))
        if workers == 1:
            # Avoid forking after model inference. Forking a process that already
            # initialized CUDA/large thread pools can deadlock before the first
            # eval json is written.
            annotate(prediction_set, incomplete_files, output_dir, model, sleep_per_call)
        else:
            parts = chunk_list(incomplete_files, workers)
            task_args = [(prediction_set, part, output_dir, model, sleep_per_call) for part in parts]
            with Pool(initializer=init_client, initargs=(api_key, api_base), processes=workers) as pool:
                pool.starmap(annotate, task_args)

    # ========== Step 5. Combine results ==========
    combined_contents = {}
    missing_files = []
    for file_name in caption_files:
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as jf:
                content = json.load(jf)
                if isinstance(content, list) and len(content) >= 1:
                    combined_contents[file_name[:-5]] = content
        except Exception as e:
            print(f"[combine] failed to read {file_name}: {e}")

    if missing_files:
        print(f"[combine] Warning: {len(missing_files)} eval files missing. First missing: {missing_files[:5]}")

    output_parent = os.path.dirname(output_json)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as outf:
        json.dump(combined_contents, outf, ensure_ascii=False, indent=2)
    print("All evaluation completed! Combined saved to:", output_json)

    return combined_contents
