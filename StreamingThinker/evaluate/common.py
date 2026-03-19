import json
import re
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.distributed import all_gather_object, is_initialized
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import utils  # noqa: E402
from dataloader_hf import StreamingDataCollator  # noqa: E402


SPECIAL_TOKENS = ["<SEP>", "<EOQ>", "<EOS>", "<EOT>", "<EOR>", "<EOA>", "<ignore>"]
SUPPORTED_BACKBONES = ["Qwen3"]


def build_parser(*, default_mode, default_config, default_attn_implementation, description):
    parser = ArgumentParser(description=description)

    parser.add_argument(
        "--model_name_or_path",
        "--LLM_path",
        dest="model_name_or_path",
        required=True,
        help="Model checkpoint or Hugging Face model ID used for generation.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        "--Tokenizer_path",
        dest="tokenizer_name_or_path",
        default=None,
        help="Tokenizer path or model ID. Defaults to --model_name_or_path.",
    )
    parser.add_argument(
        "--model_backbone",
        "--LLM_backbone",
        dest="model_backbone",
        default="Qwen3",
        choices=SUPPORTED_BACKBONES,
        help="Streaming model implementation to load.",
    )
    parser.add_argument(
        "--data_config",
        "--params",
        dest="data_config",
        default=str(default_config),
        help="JSON config containing dataset and prompt settings.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "outputs" / "eval"),
        help="Directory used to save evaluation artifacts.",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Filename prefix for generated JSON outputs.",
    )
    parser.add_argument("--inference_mode", default=default_mode, choices=["batch", "streaming"])
    parser.add_argument("--split_mode", default="sentence", choices=["word", "sentence", "token"])
    parser.add_argument(
        "--per_device_eval_batch_size",
        "--per_bs",
        dest="per_device_eval_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn_implementation", default=default_attn_implementation, choices=["eager", "sdpa"])
    parser.add_argument(
        "--batch_from_position_zero",
        "--is_batch_from_zero",
        dest="batch_from_position_zero",
        action=BooleanOptionalAction,
        default=False,
        help="Reset target position indices for batch decoding.",
    )

    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_tokenizer(tokenizer_name_or_path, config, *, padding_side):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side=padding_side, config=config)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def extract_boxed_answer(text):
    match = re.findall(r"\\box(?:ed)?\{(.*?)\}", text, re.DOTALL)
    return match[0] if match else ""


def extract_reference_answer(text):
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    match = re.findall(r"####\s*([^\n]+)", text, re.DOTALL)
    return match[0].strip() if match else text.strip()


def gather_objects(accelerator, values):
    if accelerator.num_processes == 1 or not is_initialized():
        return values

    gathered = [None for _ in range(accelerator.num_processes)]
    all_gather_object(gathered, values)
    merged = []
    for chunk in gathered:
        merged.extend(chunk)
    return merged


def get_model_class(model_backbone):
    if model_backbone == "Qwen3":
        from Qwen3.qwen_streaming import Qwen3ForCausalLM_stream

        return Qwen3ForCausalLM_stream

    raise ValueError(f"Unsupported model_backbone: {model_backbone}")


def run_evaluation(args):
    setup_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = utils.Params(args.data_config)
    params.file_path = utils.resolve_path(params.file_path, REPO_ROOT)

    tokenizer_name_or_path = args.tokenizer_name_or_path or args.model_name_or_path

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config._attn_implementation = args.attn_implementation

    padding_side = "left" if args.inference_mode == "batch" else "right"
    tokenizer = build_tokenizer(tokenizer_name_or_path, config, padding_side=padding_side)

    model_cls = get_model_class(args.model_backbone)
    model = model_cls.from_pretrained(
        args.model_name_or_path,
        ignore_mismatched_sizes=True,
        config=config,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    data_collator = StreamingDataCollator(
        file_path=params.file_path,
        tokenizer=tokenizer,
        Instruct=params.Instruct,
        end_Instruct=params.end_Instruct,
        assistant_Instruct=params.assistant_Instruct,
        reasoning_connect_1=params.reasoning_connect_1,
        reasoning_connect_2=params.reasoning_connect_2,
        reasoning_connect_3=params.reasoning_connect_3,
        inference_mode=args.inference_mode,
        split_mode=args.split_mode,
        is_training=False,
        pe_cache_start=0,
        is_batch_from_pe_zero=args.batch_from_position_zero,
    )

    dataloader = DataLoader(
        data_collator.dataset_loader(),
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator.collate_fn_inference,
    )

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()
    model_for_generation = accelerator.unwrap_model(model)

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    results = []
    generations = []
    progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)

    for batch in progress_bar:
        input_ids = batch.get("input_ids")
        if input_ids is None:
            input_ids = batch.get("source_tokens")

        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        prompt_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        if args.inference_mode == "batch":
            output_sequences, _ = model_for_generation.generate(
                input_ids=input_ids.to(accelerator.device),
                attention_mask=attention_mask.to(accelerator.device),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                generate_mode=args.inference_mode,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            generated_tokens = output_sequences[:, input_ids.shape[1]:]
            output_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            token_lengths = (generated_tokens != tokenizer.pad_token_id).sum(dim=1).tolist()

            for prompt, generation, answer, token_length in zip(prompt_texts, output_texts, labels, token_lengths):
                prediction = extract_boxed_answer(generation)
                results.append(
                    {
                        "prediction": prediction,
                        "answer": answer,
                        "token_length": token_length,
                    }
                )
                generations.append(
                    {
                        "prompt": prompt,
                        "generation": generation,
                        "prediction": prediction,
                        "answer": answer,
                        "correct": prediction == answer,
                    }
                )
        else:
            generated_texts, generation_lengths = model_for_generation.streaming_generate(
                input_ids=input_ids.to(accelerator.device),
                attention_mask=attention_mask.to(accelerator.device),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                assistant_token=batch["assistant_token"].to(accelerator.device),
                end_Instruct=params.end_Instruct,
                generate_mode=args.inference_mode,
                tokenizer=tokenizer,
                pe_cache_length=0,
                split_mode=args.split_mode,
                _lengths=batch["_lengths"],
                wait_k=1,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=False)
            decoded_generations = [
                tokenizer.decode(sample, skip_special_tokens=False) for sample in generated_texts
            ]
            if isinstance(generation_lengths, torch.Tensor):
                generation_lengths = generation_lengths.tolist()

            for prompt, label_text, generation, token_length in zip(
                prompt_texts,
                label_texts,
                decoded_generations,
                generation_lengths,
            ):
                prediction = extract_boxed_answer(generation)
                answer = extract_reference_answer(label_text)
                results.append(
                    {
                        "prediction": prediction,
                        "answer": answer,
                        "token_length": token_length,
                    }
                )
                generations.append(
                    {
                        "prompt": prompt,
                        "generation": generation,
                        "prediction": prediction,
                        "answer": answer,
                        "correct": prediction == answer,
                    }
                )

    accelerator.wait_for_everyone()
    results = gather_objects(accelerator, results)
    generations = gather_objects(accelerator, generations)

    if accelerator.is_main_process:
        output_prefix = args.output_prefix or f"{Path(args.model_name_or_path).name}_{args.inference_mode}"
        results_path = output_dir / f"{output_prefix}_results.json"
        generations_path = output_dir / f"{output_prefix}_generations.json"

        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)

        with open(generations_path, "w", encoding="utf-8") as file:
            json.dump(generations, file, ensure_ascii=False, indent=2)

        print(f"Saved evaluation summary to {results_path}")
        print(f"Saved generations to {generations_path}")
