# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import os
import pathlib
import shutil
import sys
from pathlib import Path

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# from qwenvl.data.data_qwen import make_supervised_data_module
from model_code.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwenvl.data.lmm_dataset_for_batch import make_supervised_data_module
from qwenvl.train.argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from trainer import replace_qwen2_vl_attention_class
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Trainer,
)

local_rank = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def add_special_tokens_safely(tokenizer, new_tokens):
    """
    安全地向 tokenizer 添加新的 special tokens，保留原有的 additional_special_tokens。

    Args:
        tokenizer: Hugging Face tokenizer
        model: 对应的语言模型
        new_tokens: list of str, 要添加的新 token

    Returns:
        bool: 是否有新 token 被添加
    """

    current_vocab = set(tokenizer.get_vocab().keys())

    tokens_to_add = [t for t in new_tokens if t not in current_vocab]
    if not tokens_to_add:
        rank0_print("🟢 所有指定的 token 已存在于词表中，无需添加。")
        return False

    orig_special_tokens = tokenizer.special_tokens_map.get(
        "additional_special_tokens", []
    )

    updated_special_tokens = orig_special_tokens + [
        t for t in tokens_to_add if t not in orig_special_tokens
    ]

    rank0_print(f"📌 正在添加新 token: {tokens_to_add}")
    rank0_print(
        f"🔧 更新后的 additional_special_tokens 总数: {len(updated_special_tokens)}"
    )

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": updated_special_tokens}
    )

    if num_added > 0:
        rank0_print(f"✅ 成功添加 {num_added} 个新 token 到词表")

    return num_added > 0


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    data_args.image_processor = processor.image_processor
    data_args.model_type = "qwen2.5vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Token          ID
    # -----------------------
    # <think>        151665
    # </think>       151666
    # <answer>       151667
    # </answer>      151668
    NEW_SPECIAL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]
    was_updated = add_special_tokens_safely(tokenizer, NEW_SPECIAL_TOKENS)
    if not was_updated:
        rank0_print("ℹ️ 未检测到新 token 添加，继续执行...")
    else:
        if model_args.resize_token_embeddings:
            model.resize_token_embeddings(len(tokenizer))
            model.config.vocab_size = len(tokenizer)
            rank0_print(f"✅ Resized embeddings, new vocab size = {len(tokenizer)}")

    model.tokenizer = tokenizer

    model.randK = data_args.randK
    model.randF = data_args.randF

    model.dataset_use = data_args.dataset_use

    set_model(model_args, model)

    # if torch.distributed.get_rank() == 0:
    #     model.visual.print_trainable_parameters()
    #     model.model.print_trainable_parameters()

    processor.tokenizer = tokenizer
    data_module = make_supervised_data_module(processor=processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = "chat_template.json"
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    if model_args.resize_token_embeddings:
        model.config.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    # train(attn_implementation="flash_attention_2")
    train()
