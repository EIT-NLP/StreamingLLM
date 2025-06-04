from transformers import AutoTokenizer, AutoConfig
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader_hf import StreamingDataCollator
from Models.Llama3.llama_streaming import LlamaForCausalLM_stream
from Models.Qwen2_5.qwen_streaming import Qwen2ForCausalLM_stream
from Models.Gemma2.gemma2_streaming import Gemma2ForCausalLM_stream
os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import peft
from argparse import ArgumentParser
import logging
import utils
from lagging import calculate_al_and_laal
import sacrebleu

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--pe_cache_length", type=float, default=0)
    parser.add_argument("--inference_mode", type=str, default="streaming",
                        choices=["batch", "streaming"])
    parser.add_argument("--LLM_backbone", type=str, default="Qwen",
                        choices=["Qwen", "Llama", "Gemma"])
    parser.add_argument("--LLM_path", type=str, required=True, help="Path to the LLM model.")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--wait_k", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument("--split_mode", type=str, default="word", 
                        choices=["word", "token"])
    parser.add_argument("--params", type=str, default="./configs/params_qwen_inference.json")
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = get_args()
    params = utils.Params(args.params)
    setup_seed(0)

    logging.basicConfig(
        filename=f"{args.output_dir}/output.log",  
        level=logging.INFO,       
        format="%(asctime)s - %(levelname)s - %(message)s",  
    )


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

    model = CausalLM_stream.from_pretrained(args.LLM_path, ignore_mismatched_sizes=True, config=config)

    if args.lora_path is not None:
        model = peft.PeftModel.from_pretrained(model, args.lora_path)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

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
        inference_mode=args.inference_mode,  # "batch", "streaming"
        split_mode=args.split_mode,        # "word", "token"
        if_add_space=params.if_add_space,
        pe_cache_length=args.pe_cache_length,
        wait_k = args.wait_k,
    )

    data_collator_dataset = data_collator.dataset_loader() 
    dataloader = DataLoader(
        data_collator_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator.collate_fn_inference  # ✅ use the collate function
    )

    accelerator = Accelerator(mixed_precision="bf16")
    stream_model, dataloader = accelerator.prepare(model, dataloader)


    target_txt_lt=[]
    output_text_lt=[]
    AL = []
    LAAL = []

    stream_model.eval()
    for step, batch in enumerate(tqdm(dataloader)):
        input_ids = batch.get("source_tokens", None)
        attention_mask = batch.get("attention_mask", None)
        _lengths = batch.get("_lengths", None)
        # position_ids = batch.get("position_ids", None)
        # ...
        inference_mode = batch.get("inference_mode", "streaming") 
        split_mode = batch.get("split_mode", None)
        _lengths_index = batch.get("_lengths_index", None)
        wait_k = batch.get("wait_k", None)
        assistant_token = batch.get("assistant_token", None)
        source_txt = batch.get("source_txt", None)
        target_txt = batch.get("target_txt", None)

        if inference_mode == "streaming":
            output_sequences, wait_lagging = stream_model.generate(
                input_ids=input_ids.to(accelerator.device),
                attention_mask=attention_mask.to(accelerator.device),
                max_new_tokens=1024,
                generate_mode=inference_mode,
                split_mode=split_mode,
                pe_cache_length = args.pe_cache_length,
                tokenizer=tokenizer,
                end_Instruct=params.end_Instruct,
                _lengths = _lengths,
                _lengths_index = _lengths_index,
                wait_k = wait_k,
                source_words = source_txt,
                assistant_token = assistant_token.to(accelerator.device),
                # # output_attentions = True,
            )
        elif inference_mode == "batch":
            output_sequences, wait_lagging = stream_model.generate(
                input_ids=input_ids.to(accelerator.device),
                attention_mask=attention_mask.to(accelerator.device),
                max_new_tokens=1024,
            )


        

        output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        target_txt_lt.extend(target_txt)
        output_text_lt.extend([output_text,])

        output_text_len = len(output_text.split())
        al, laal = calculate_al_and_laal(len(_lengths[0]['source_seg_len'])-1, output_text_len, wait_lagging[:output_text_len])
        AL.append(al)
        LAAL.append(laal)


        if accelerator.is_main_process:
            print("streaming-input:\n", source_txt)
            print("streaming-output:\n", output_text)


    avg_LAAL = sum(LAAL) / len(LAAL)
    avg_AL = sum(AL) / len(AL)
    bleu = sacrebleu.corpus_bleu(output_text_lt, [target_txt_lt])

    logging.info(f"BLEU score: {bleu.score:.2f}")
    logging.info(f"all_AL score: {avg_AL:.2f}")
    logging.info(f"all_LAAL score: {avg_LAAL:.2f}")


if __name__ == "__main__":
    main()