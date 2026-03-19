from datasets import Dataset
import json
# from torch.utils.data import Dataset
import torch
import re
from typing import Literal, Optional
from transformers import PreTrainedTokenizer
import random

try:
    import ijson
except ImportError:  # pragma: no cover - optional fast path
    ijson = None



class StreamingDataCollator():
    def __init__(
                self, 
                file_path: str, 
                tokenizer: PreTrainedTokenizer, 
                Instruct: str,             # eg: '<|system|>Translate the following paragraph\n'
                assistant_Instruct: str,   # eg: '<|assitant|>'
                end_Instruct: str,         # eg: '<|end|>'

                reasoning_connect_1: str,
                reasoning_connect_2: str,
                reasoning_connect_3: str,

                training_mode: Literal['streaming', 'batch'] = 'streaming',
                split_mode: Literal['word', 'sentence', 'token'] = 'word', 
                inference_mode: Literal['batch', 'streaming'] = 'streaming',
                is_training = True,

                # if_add_space: bool = False,# split_mode =='word'; for Llama, gemma, if_add_space=True
                pe_cache_start=0,          # start position id of target tokens in streaming mode
                is_batch_from_pe_zero: bool = False,  # whether to use batch from pe_cache, only for batch mode
                ):
        assert training_mode in ['streaming', 'batch']
        assert split_mode in ['word', 'sentence', 'token']
        assert inference_mode in ['batch', 'streaming']

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.Instruct = Instruct
        self.Instruct_reset = Instruct
        self.assistant_Instruct = assistant_Instruct
        self.assistant_Instruct_token = self.tokenizer(self.assistant_Instruct, add_special_tokens=False)['input_ids']

        self.end_Instruct = end_Instruct
        self.pe_cache_start = pe_cache_start

        self.reasoning_connect_1 = reasoning_connect_1
        self.reasoning_connect_2 = reasoning_connect_2
        self.reasoning_connect_3 = reasoning_connect_3

        
        self.training_mode = training_mode
        self.inference_mode = inference_mode
        self.split_mode = split_mode
        self.is_training = is_training
        self.is_batch_from_pe_zero = is_batch_from_pe_zero
        self.count = 0


    def _load_samples(self, file_path):
        """
        Load JSON data in a memory-efficient way using `ijson`.
        This method parses a JSON array **incrementally**, yielding one sample at a time.
        Parameters:
        - file_path: str, Path to the JSON file.
        Yields:
        - A single JSON object (dict) at a time.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            if ijson is not None:
                for item in ijson.items(f, "item"):
                    yield item
            else:
                for item in json.load(f):
                    yield item


    def extract_before_eot_split(self, text, end_token='<EOT>'):
        """
        Split text into chunks ending with the provided token pattern.

        Args:
            text: Input string.
            end_token: End marker expressed as a regex pattern, such as
                ``<EOT>`` or ``<EOQ>|<EOT>``.
        """
        pattern = f'(.*?)({end_token})'
        
        matches = re.findall(pattern, text, re.DOTALL)
        return [content + tag for content, tag in matches]

    def extract_instruct(self, text):
        if self.is_training:
            Instruct = random.choice(self.Instruct)
            
            reasoning_stage1 = text.get("streaming_reasoning",None)
            reasoning_stage2 = text.get("deep_reasoning",None)
            final_result = text.get("answer",None)

            reasoning = text.get("reasoning",None)
            result = text.get("result",None)

            reasoning_connect_1 = random.choice(self.reasoning_connect_1) 
            reasoning_connect_2 = random.choice(self.reasoning_connect_2) 
            reasoning_connect_3 = random.choice(self.reasoning_connect_3)
            if reasoning_stage1 is not None:
                if self.split_mode =="sentence":

                    source_txt_lt = [Instruct+'<SEP>']  + [text['Instruct_Context']["instruct"]+'<EOQ>'] + self.extract_before_eot_split(text['Instruct_Context']["context"],end_token='<EOS>')
                    source_txt_lt[-1] = source_txt_lt[-1] + self.end_Instruct
                    target_txt_lt = [reasoning_connect_1+'<SEP>'] + self.extract_before_eot_split(reasoning_stage1,end_token='<EOQ>|<EOT>') + [reasoning_connect_2 + reasoning_stage2 + reasoning_connect_3 + final_result + self.end_Instruct]
                    

            elif reasoning is not None:
                if self.split_mode =="sentence":
                    source_txt_lt = [Instruct+'<SEP>']  + [text['Instruct_Context']["instruct"]+'<EOQ>'] + self.extract_before_eot_split(text['Instruct_Context']["context"],end_token='<EOS>') +  [self.end_Instruct]
                    target_txt_lt = [reasoning_connect_1+'<SEP>'] + [reasoning] + [reasoning_connect_2 + reasoning_connect_3 + result + self.end_Instruct]


            inputs = ''.join(source_txt_lt)
            labels = ''.join(target_txt_lt)
        else:
            Instruct = random.choice(self.Instruct)
            
            reasoning_stage1 = text.get("streaming_reasoning",None)
            reasoning_stage2 = text.get("deep_reasoning",None)
            final_result = text.get("answer",None)

            reasoning = text.get("reasoning",None)
            result = text.get("result",None)

            source_txt_lt = [Instruct+'<SEP>' + text['Instruct_Context']["instruct"]+'<EOQ>'] + self.extract_before_eot_split(text['Instruct_Context']["context"],end_token='<EOS>')
            source_txt_lt[-1] = source_txt_lt[-1]+self.end_Instruct
            target_txt_lt =None


            inputs = ''.join(source_txt_lt)
            labels = text.get("answer",None)

        return (inputs, labels, source_txt_lt, target_txt_lt)

    def generator(self, samples):
        """
        Process the dataset by splitting text and constructing formatted prompts.
        Parameters:
        - samples: Iterable, Generator that yields JSON samples.
        Yields:
        - Dictionary with processed text fields.
        """
        for text in samples:
            (inputs, labels, source_txt_lt, target_txt_lt) = self.extract_instruct(text)
            # yield {"input_txt":inputs, "labels":labels, }
            yield {"input_txt":inputs,       "labels":labels,    "Instruct":self.Instruct,
                   "source_txt_lt":source_txt_lt, "target_txt_lt":target_txt_lt}

    def dataset_loader(self, file_paths=None):
        """
        file_paths: list of file paths to be mixed together
        """
        if file_paths is None:
            file_paths = self.file_path 
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        assert isinstance(file_paths, list), "file_paths should be a list of file paths"

        def combined_generator():
            from random import shuffle
            generators = [self._load_samples(fp) for fp in file_paths]
            # interleave samples from all files
            while generators:
                shuffle(generators)  # optional random shuffle between generators
                for gen in generators[:]:  # iterate copy of list
                    try:
                        item = next(gen)
                        yield from self.generator([item])
                    except StopIteration:
                        generators.remove(gen)
        
        return Dataset.from_generator(combined_generator)


    '''for training'''
    def collate_fn(self, batch_data):
        """
        Process batch data, encode with tokenizer and pad
        """
        # input_texts = batch_data["input_txt"]
        input_texts = [item["input_txt"] for item in batch_data]
        labels = [item["labels"] for item in batch_data]
        concat_texts = [inp + tgt for inp, tgt in zip(input_texts, labels)]
        source_txt_lt = [item["source_txt_lt"] for item in batch_data]
        target_txt_lt = [item["target_txt_lt"] for item in batch_data]
        instruct = [item["Instruct"] for item in batch_data]


        concat_tokens = self.tokenizer(concat_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        batch_len = concat_tokens['input_ids'].shape[1]

        source_seg_len_lt = []
        for item in source_txt_lt:
            source_seg_len = []
            for sentence in item:
                token_num = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)['input_ids'].shape[1]
                source_seg_len.append(token_num)
            source_seg_len_lt.append(source_seg_len)

        target_seg_len_lt = []
        for item in target_txt_lt:
            target_seg_len = []
            for sentence in item:
                token_num = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)['input_ids'].shape[1]
                target_seg_len.append(token_num)
            target_seg_len_lt.append(target_seg_len)
        


        labels_tokens = concat_tokens["input_ids"].clone()
        for i, (inp, tgt) in enumerate(zip(input_texts, labels)):
            inp_ids = self.tokenizer(inp, add_special_tokens=False)["input_ids"]
            inp_len = len(inp_ids)
            labels_tokens[i, :inp_len] = -100  # input
            labels_tokens[i][concat_tokens["attention_mask"][i] == 0] = -100 #padding

        if self.training_mode == 'batch':
            if not self.is_batch_from_pe_zero:
                # batch_len = concat_tokens['input_ids'].shape[1]
                position_ids = [list(range(batch_len))]
            else:
                position_ids = []
                for source_seg_len, target_seg_len in zip(source_seg_len_lt, target_seg_len_lt):
                    source_token_len = sum(source_seg_len)
                    target_token_len = sum(target_seg_len)
                    position_id = list(range(source_token_len))
                    position_id.extend(list(range(batch_len - source_token_len)))
                    position_ids.append(position_id)
                    # print("position_id:", position_ids)
        else:
            position_ids = []
            for source_seg_len, target_seg_len in zip(source_seg_len_lt, target_seg_len_lt):
                source_token_len = sum(source_seg_len)
                target_token_len = sum(target_seg_len)
                position_id = list(range(source_token_len))
                position_id.extend(list(range(batch_len - source_token_len)))
                position_ids.append(position_id)
                # print("position_id:", position_ids)


        position_ids = torch.tensor(position_ids)

        _lengths = []
        for source_seg_len, target_seg_len in zip(source_seg_len_lt, target_seg_len_lt):
            _lengths.append({'source_token_len':sum(source_seg_len),'source_seg_len':source_seg_len,
                             'target_token_len':sum(target_seg_len),'target_seg_len':target_seg_len,
                             'input_batch_len': batch_len,
                             })
        return {
            "input_ids": concat_tokens["input_ids"],
            "attention_mask": concat_tokens["attention_mask"],
            "labels": labels_tokens,
            "source_seg_len": source_seg_len_lt,
            "target_seg_len": target_seg_len_lt,
            "position_ids": position_ids,
            "training_mode": self.training_mode,
            "split_mode": self.split_mode,
            "_lengths": _lengths
        }

    '''for inference'''
    def collate_fn_inference(self, batch_data):
        """
        Process batch data for inference, encode with tokenizer and pad.
        """
        # Batch mode is now much simpler.
        if self.inference_mode == 'batch':
            # 1. Extract source and target texts from the batch.
            # Assuming the generator yields dictionaries with 'input_txt' and 'labels'.
            # You might need to adjust the keys based on what `extract_instruct` and `generator` actually produce for inference.
            # Let's assume 'input_txt' is the source and 'labels' is the target.
            source_texts = [item["input_txt"] for item in batch_data]
            # target_texts = [item["labels"] for item in batch_data]
            labels = [item["labels"] for item in batch_data]

            # 2. Tokenize the source texts with padding.
            # This prepares the `input_ids` and `attention_mask` for the model.
            source_tokens = self.tokenizer(
                source_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                add_special_tokens=False
            )

            # 3. Return a clean dictionary for batch inference.
            # The target texts are kept as strings for evaluation later (e.g., Sacrebleu).
            return {
                "input_ids": source_tokens["input_ids"],
                "attention_mask": source_tokens["attention_mask"],
                "labels": labels, # Pass raw target strings for evaluation
                "inference_mode": self.inference_mode,
            }
        elif self.inference_mode == 'streaming':
            source_txt = [item["input_txt"] for item in batch_data]
            source_tokens = self.tokenizer(source_txt, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
            target = [item["labels"] for item in batch_data]
            target_tokens = self.tokenizer(target, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)

            source_seg_len_lt = []
            source_txt_lt = [item["source_txt_lt"] for item in batch_data]
            for item in source_txt_lt:
                source_seg_len = []
                for sentence in item:
                    token_num = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)['input_ids'].shape[1]
                    source_seg_len.append(token_num)
                source_seg_len_lt.append(source_seg_len)

            position_ids = []
            max_seq_len = 0
            for source_seg_len in source_seg_len_lt:
                source_token_len = sum(source_seg_len)
                position_id = list(range(source_token_len))
                position_ids.append(position_id)
                max_seq_len = max(max_seq_len, source_token_len)

            # position_ids = position_ids[:source_tokens["input_ids"].shape[-1]]
            padded_position_ids = []
            for pid in position_ids:
                pad_num = max_seq_len - len(pid)
                padded_pid = [0] * pad_num + pid 
                padded_position_ids.append(padded_pid)
            position_ids = padded_position_ids
            position_ids = torch.tensor(position_ids)


            _lengths = []
            for source_seg_len in source_seg_len_lt:
                _lengths.append({'source_token_len':sum(source_seg_len),'source_seg_len':source_seg_len,
                                # 'input_batch_len': batch_len,
                                })
            
            return {
                "source_tokens": source_tokens["input_ids"],
                "attention_mask": source_tokens["attention_mask"],
                "labels": target_tokens["input_ids"],
                "_lengths": _lengths,
                "position_ids": position_ids,
                # "target_txts": target,
                # "attn_mask_index": attn_mask_index,    # Mask defining different token types (source, target, padding) for loss calculation.
                # ...
                "inference_mode": self.inference_mode,
                "split_mode": self.split_mode,
                # "_lengths_index": _lengths_index,
                # ...
                "assistant_token":torch.tensor(self.assistant_Instruct_token),
                "source_txt":source_txt,
                # "target_txt":target_txt,
            }
            
