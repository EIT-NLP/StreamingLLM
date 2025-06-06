from transformers import AutoTokenizer, TrainingArguments, AutoConfig
from dataloader_hf import StreamingDataCollator
from streaming_trainer import StreamingSFTTrainer
from models.Llama3.llama_streaming import LlamaForCausalLM_stream
from models.Qwen2_5.qwen_streaming import Qwen2ForCausalLM_stream
from models.Gemma2.gemma2_streaming import Gemma2ForCausalLM_stream
import wandb
import os
os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
from peft import LoraConfig, get_peft_model
import torch
from transformers.trainer_utils import is_main_process
from accelerate import Accelerator
from argparse import ArgumentParser
import utils

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--pe_cache_length", type=float, default=0)
    parser.add_argument("--training_mode", type=str, default="streaming",
                        choices=["batch", "streaming"])
    parser.add_argument("--LLM_backbone", type=str, default="Qwen",
                        choices=["Qwen", "Llama"])
    parser.add_argument("--LLM_path", type=str, required=True, help="Path to the LLM model.")
    parser.add_argument("--wait_k", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--split_mode", type=str, default="word", 
                        choices=["word", "token"])
    parser.add_argument("--params", type=str, default="./configs/params_qwen.json")
    parser.add_argument("--lr", type=float, required=True, help="learning rate.")
    parser.add_argument("--warmup_steps", type=int, required=True, help="warmup steps.")
    parser.add_argument("--warmup_type", type=str, default="linear")
    parser.add_argument("--per_bs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--acc_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--is_lora", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default='./checkpoints/Qwen')
    parser.add_argument("--log_dir", type=str, default='./logs')
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = get_args()
    params = utils.Params(args.params)
    torch.cuda.set_device(args.device)
    setup_seed(args.seed)


    config = AutoConfig.from_pretrained(args.LLM_path)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_path,padding_side='right', config=config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    CausalLM_stream = {
        "Qwen": Qwen2ForCausalLM_stream,
        "Llama": LlamaForCausalLM_stream,
        "Gemma": Gemma2ForCausalLM_stream,
    }
    CausalLM_stream = CausalLM_stream[args.LLM_backbone]




    # StreamingDataCollator
    data_collator = StreamingDataCollator(
        file_path=params.file_path,
        tokenizer=tokenizer,
        Instruct=params.Instruct,
        user_Instruct=params.user_Instruct,
        assistant_Instruct=params.assistant_Instruct,
        end_Instruct=params.end_Instruct,
        source_key=params.source_key,
        target_key=params.target_key,
        training_mode=args.training_mode,  # "batch", "streaming"
        split_mode=args.split_mode,        # "word", "sentence"
        if_add_space=params.if_add_space,
        pe_cache_length=args.pe_cache_length,
        wait_k = args.wait_k,
    )


    accelerator = Accelerator()
    if accelerator.is_main_process:
        wandb.init(
            project=f"Streaming",
            name=f"{args.LLM_backbone}-{args.training_mode}",
            config={
            "learning_rate": args.lr,
            "batch_size": args.per_bs,
            "seed":1,
            "lora":(32,64)
            },
            mode="offline"
        )





    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_bs,
        per_device_eval_batch_size=args.per_bs,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.warmup_type,
        logging_dir=args.log_dir,
        remove_unused_columns=False,
        logging_steps=25,
        report_to="wandb",
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=True,
        gradient_accumulation_steps=args.acc_steps,
    )


    stream_model = CausalLM_stream.from_pretrained(args.LLM_path, config=config)
    if args.is_lora:
        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj","k_proj","v_proj", "o_proj","gate_up_proj","up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="all",
            task_type="CAUSAL_LM",
        )
        stream_model = get_peft_model(stream_model, peft_config)




    # SFTTrainer
    trainer = StreamingSFTTrainer(
        model=stream_model,
        args=training_args,
        train_dataset=data_collator.dataset_loader(),
        data_collator=data_collator.collate_fn,  
        source_instruct_length=data_collator.source_instruct_length,  # source instruct length
    )

    # start training
    trainer.train()

if __name__ == "__main__":
    main()
