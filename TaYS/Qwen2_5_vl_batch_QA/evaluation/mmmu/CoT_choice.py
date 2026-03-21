import argparse
import json
import re

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a JSONL file using Qwen2.5-VL to select answers from options."
    )

    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="/code/code/video_streaming/Qwen2_5_vl_interleave/evaluation-interleave/mmmu/output/qwen2_5vl-CoT-interleave-0914.jsonl",
        help="Path to the input JSONL file containing 'result_with_think', 'options', etc.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="/code/code/video_streaming/Qwen2_5_vl_interleave/evaluation-interleave/mmmu/output/qwen2_5vl-CoT-interleave-0914-choice.jsonl",
        help="Path to save the output JSONL file. If not provided, appends '_select_answer' to input filename.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/code/code/video_streaming/Qwen2_5_vl_interleave/qwen-vl-finetune/output/qwen2_5vl-pe-interleave-CoT",
        help="Path to the fine-tuned Qwen2.5-VL model. Should support vision/language inputs if needed.",
    )

    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    input_jsonl = args.input_jsonl
    output_jsonl = args.output_jsonl
    if output_jsonl is None:
        output_jsonl = input_jsonl.replace(".jsonl", "_select_answer.jsonl")

    predictions = []

    print(f"Reading input from: {input_jsonl}")
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON on line {line_num}: {e}")
                continue

            task = data.get("task", "Unknown")
            question = data.get("messages", [{}])[0].get("value", "")
            evidence = data.get("result_with_think", "").strip()
            options = data.get("options", [])
            correct_answer = data.get("correct_answer", "")

            options_prompt = "\n".join([opt.strip() for opt in options])

            prompt = f"""Please finish the {task} task.
                        Question: {question}
                        Your inference evidence is: {evidence}
                        You have the following options:
                        {options_prompt}
                        Select the answer and only give one option letter (e.g., (A), (D))."""

            messages = [{"role": "user", "content": prompt}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(
                text=[text], images=None, videos=None, padding=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    num_beams=1,
                    temperature=0.0,
                    repetition_penalty=1.0,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            pred_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            match = re.search(r"\([A-Z]\)", pred_text)
            predicted_answer = match.group(0) if match else pred_text[:4]  # fallback

            result = {
                "task": task,
                "question": question,
                "options": options,
                "result_with_think": evidence,
                "predicted_answer": predicted_answer,
                "correct_answer": correct_answer,
                "raw_output": pred_text,
            }
            predictions.append(result)

            print(
                f"Correct: {correct_answer} | Predicted: {predicted_answer} | Raw: '{pred_text}'"
            )

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for item in predictions:
            f_out.write(
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    print(f"\n✅ 所有预测结果已保存至：{output_jsonl}")


if __name__ == "__main__":
    main()
