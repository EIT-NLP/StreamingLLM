import argparse
import json
import re
from pathlib import Path


def extract_option(text):
    """
    Extracts the first option letter (A, B, C, or D) from a given string.

    Args:
        text (str): The string containing the answer or option.

    Returns:
        str: The extracted letter (e.g., 'A') or None if no letter is found.
    """
    if not isinstance(text, str):
        return None
    match = re.search(r"[A-D]", text)
    return match.group(0) if match else None


def calculate_accuracy_by_task(jsonl_file_path, output_result_path=None):
    """
    Calculates accuracy by task category from a JSONL file and saves the result.

    Args:
        jsonl_file_path (str): Path to the input JSONL file.
        output_result_path (str, optional): Path to save evaluation results. If None, uses auto name.

    Returns:
        dict: Per-task accuracy and overall accuracy.
    """
    try:
        with open(jsonl_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file at {jsonl_file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    if not lines:
        print("Warning: File is empty.")
        return {}

    task_stats = {}
    total_correct = 0
    total_count = 0
    model_answer_none_count = 0

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue  # skip empty lines

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON on line {line_num}: {e}")
            continue

        task = item.get("task", "Unknown")
        model_output = item.get("predicted_answer")
        correct_answer = item.get("correct_answer")

        pred_letter = extract_option(model_output)
        true_letter = extract_option(correct_answer)

        if task not in task_stats:
            task_stats[task] = {"correct": 0, "total": 0}

        if true_letter is None:
            continue

        task_stats[task]["total"] += 1
        total_count += 1

        if pred_letter and pred_letter == true_letter:
            task_stats[task]["correct"] += 1
            total_correct += 1
        elif pred_letter is None:
            model_answer_none_count += 1

    overall_acc = (total_correct / total_count) * 100 if total_count > 0 else 0.0

    print(f"\n{'Task':<40} {'Correct':<10} {'Total':<10} {'Accuracy (%)':<12}")
    print("-" * 80)
    for task, stats in sorted(task_stats.items()):
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        print(f"{task:<40} {stats['correct']:<10} {stats['total']:<10} {acc:<12.2f}")

    print("-" * 80)
    print(f"{'Overall':<40} {total_correct:<10} {total_count:<10} {overall_acc:<12.2f}")
    print(f"\nModel failed to extract answer in {model_answer_none_count} cases.")

    result = {
        task: {
            "accuracy": round((stats["correct"] / stats["total"]) * 100, 2),
            "correct": stats["correct"],
            "total": stats["total"],
        }
        for task, stats in task_stats.items()
    }
    result["__overall__"] = {
        "accuracy": round(overall_acc, 2),
        "correct": total_correct,
        "total": total_count,
        "model_answer_failed_count": model_answer_none_count,
    }

    if output_result_path is None:
        input_path = Path(jsonl_file_path)
        output_result_path = input_path.with_name(
            input_path.stem + "_accuracy.json"
        ).as_posix()

    try:
        with open(output_result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"\n📊 详细评估结果已保存至：{output_result_path}")
    except Exception as e:
        print(f"❌ 无法保存结果文件：{e}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions from a JSONL file by task."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="/code/code/video_streaming/Qwen2_5_vl_interleave/evaluation-interleave/mmmu/output/qwen2_5vl-CoT-interleave-0914-choice.jsonl",
        help="Path to the input JSONL file (with predicted answers).",
    )
    parser.add_argument(
        "--output_result",
        type=str,
        default="/code/code/video_streaming/Qwen2_5_vl_interleave/evaluation-interleave/mmmu/output/qwen2_5vl-CoT-interleave-0914-score.json",
        help="Path to save accuracy results (JSON). If not provided, auto-generated.",
    )
    args = parser.parse_args()

    result = calculate_accuracy_by_task(args.input_jsonl, args.output_result)

    if result is not None:
        print("\n✅ Evaluation completed.")
