from datasets import Dataset
import torch
import re
from typing import Literal, Optional
from transformers import PreTrainedTokenizer
import ijson



class StreamingDataCollator():
    def __init__(
                self, 
                file_path: str, 
                tokenizer: PreTrainedTokenizer, 
                Instruct: str,             # eg: '<|system|>Translate the following paragraph\n'
                user_Instruct: str,        # eg: '<|user|>'
                assistant_Instruct: str,   # eg: '<|assitant|>'
                end_Instruct: str,         # eg: '<|end|>'
                training_mode: Literal['streaming', 'batch'] = 'streaming',
                split_mode: Literal['word', 'token'] = 'word', 
                inference_mode: Literal['batch', 'streaming'] = 'streaming',
                source_key: Optional[str] = None, 
                target_key: Optional[str] = None, 
                if_add_space: bool = False,# split_mode =='word'; for Llama, gemma, if_add_space=True
                pe_cache_length=0,          # start position id of target tokens in streaming mode
                wait_k = None,
                ):
        """"
        training_mode: 'streaming' or 'batch'
        split_mode: 'word' or 'token'
        """
        # super().__init__()
        assert training_mode in ['streaming', 'batch']
        assert split_mode in ['word', 'token']
        assert inference_mode in ['batch', 'streaming']

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.source_key = source_key
        self.target_key = target_key
        self.Instruct = Instruct
        self.user_Instruct = user_Instruct
        self.assistant_Instruct = assistant_Instruct
        self.end_Instruct = end_Instruct
        self.pe_cache_length = pe_cache_length
        self.assistant_Instruct_token = self.tokenizer(self.assistant_Instruct, add_special_tokens=False)['input_ids']
        self.source_instruct_length = self.tokenizer(self.Instruct+self.user_Instruct, add_special_tokens=False)['input_ids'].__len__()
        self.target_instruct_length = self.tokenizer(self.assistant_Instruct, add_special_tokens=False)['input_ids'].__len__()
        self.end_Instruct_length = self.tokenizer(self.end_Instruct, add_special_tokens=False)['input_ids'].__len__()

        self.training_mode = training_mode
        self.split_mode = split_mode
        self.inference_mode = inference_mode
        self.if_add_space = if_add_space

        self.wait_k = wait_k
        

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
            for item in ijson.items(f, "item"):  # Parse JSON objects one by one
                yield item


    def generator(self, samples):
        """
        Process the dataset by splitting text and constructing formatted prompts.
        Parameters:
        - samples: Iterable, Generator that yields JSON samples.
        Yields:
        - Dictionary with processed text fields.
        """
        for text in samples:
            # split text into words or sentences (remove redundant spaces or newlines)
            if self.split_mode in ['word', 'token']: 
                source_txt_lt = text[self.source_key].split()
                target_txt_lt = text[self.target_key].split()
            elif self.split_mode == 'sentence':
                source_txt_lt =re.split(r'([.?!])\s+', text[self.source_key])
                target_txt_lt =re.split(r'([.?!])\s+', text[self.target_key])
            # construct source and target text
            source_txt = ' '.join(source_txt_lt)
            target_txt = ' '.join(target_txt_lt)

            source = self.Instruct + self.user_Instruct + source_txt + self.end_Instruct
            target = self.assistant_Instruct + target_txt + self.end_Instruct
            if self.training_mode  in ['batch','streaming']:
                input_txt = source + target
            
            yield {"source":source,       "target":target,    "Instruct":self.Instruct,
                   "input_txt":input_txt, "source_txt":source_txt, "target_txt":target_txt,
                   "source_txt_lt":source_txt_lt, "target_txt_lt":target_txt_lt}

    def dataset_loader(self):
        return Dataset.from_generator(lambda: self.generator(
                                        self._load_samples(self.file_path),  # Use streaming JSON parser
                                        )
                                    )
    



    def collate_fn(self, batch_data):
        """
        Process batch data, encode with tokenizer and pad
        """
        # input_texts = batch_data["input_txt"]
        input_texts = [item["input_txt"] for item in batch_data]
        input_tokens = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)

        source_txt_lt = [item["source_txt_lt"] for item in batch_data]
        target_txt_lt = [item["target_txt_lt"] for item in batch_data]
        lengths = input_tokens['input_ids'].shape[1]
        _lengths, position_ids, attn_mask_index = self.calculate_lengths(source_txt_lt, target_txt_lt, lengths, _mode = self.training_mode,)
        _lengths_index = torch.tensor(range(len(_lengths))).unsqueeze(1) # index of each element in _lengths (in the batch)


        return {
            "input_ids": input_tokens["input_ids"],
            "attention_mask": input_tokens["attention_mask"],
            "labels": input_tokens["input_ids"],
            "_lengths": _lengths,
            "position_ids": position_ids,
            "attn_mask_index": attn_mask_index,    # Mask defining different token types (source, target, padding) for loss calculation.
            # ...
            "training_mode": self.training_mode,
            "_lengths_index": _lengths_index,
            "wait_k": self.wait_k,
        }





    '''for inference'''
    def collate_fn_inference(self, batch_data):
        """
        Process batch data, encode with tokenizer and pad
        """
        # input_texts = batch_data["input_txt"]
        input_texts = [item["input_txt"] for item in batch_data]
        input_tokens = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)

        source_txt = [item["source_txt"] for item in batch_data]
        target_txt = [item["target_txt"] for item in batch_data]

        source = [item["source"] for item in batch_data]
        source_tokens = self.tokenizer(source, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        target = [item["target"] for item in batch_data]
        target_tokens = self.tokenizer(target, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)

        source_txt_lt = [item["source_txt_lt"] for item in batch_data]
        target_txt_lt = [item["target_txt_lt"] for item in batch_data]
        lengths = input_tokens['input_ids'].shape[1]
        _lengths, position_ids, attn_mask_index = self.calculate_lengths(source_txt_lt, target_txt_lt, lengths, _mode = self.inference_mode,)
        _lengths_index = torch.tensor(range(len(_lengths))).unsqueeze(1) # index of each element in _lengths (in the batch)

        position_ids = position_ids[:source_tokens["input_ids"].shape[-1]]


        return {
            "source_tokens": source_tokens["input_ids"],
            "attention_mask": source_tokens["attention_mask"],
            "labels": target_tokens["input_ids"],
            "_lengths": _lengths,
            "position_ids": position_ids,
            "target_txts": target,   # with start and end tokens
            # "attn_mask_index": attn_mask_index,    # Mask defining different token types (source, target, padding) for loss calculation.
            # ...
            "inference_mode": self.inference_mode,
            "split_mode": self.split_mode,
            "_lengths_index": _lengths_index,
            "wait_k": self.wait_k,
            # ...
            "assistant_token":torch.tensor(self.assistant_Instruct_token),
            "source_txt":source_txt,
            "target_txt":target_txt, # without start and end tokens
        }








    def add_space(self, text, end_token):
        assert self.split_mode == 'word'
        text_add_space = []
        for i,word in enumerate(text):
            if i > 0 and word != end_token and word !=self.tokenizer.pad_token:
                word = " " + word
            text_add_space.append(word) 
        return text_add_space

    def calculate_lengths(self, source_txt_lt, target_txt_lt, input_batch_len, _mode):
        """
        Calculate the length of each token in the source and target text. 
        'split_mode': 'word' or if 'word' split the text into words, if 'sentence' split the text into sentences.
        'source_txt_lt': list of source text (batch).
        'target_txt_lt': list of target text (batch).
        'lengths': tuple of instruction_length, source_length, target_length of batch data.
        'padding_tokens': padding tokens of LLM.
        """
        _lengths = []
        position_ids = []
        attn_mask_index = []
        for index, (source_text, target_text)  in enumerate(zip(source_txt_lt,target_txt_lt)):
            # 0. add space between words
            if self.split_mode == 'word' and self.if_add_space:
                source_text = self.add_space(source_text, self.end_Instruct)
                target_text = self.add_space(target_text, self.end_Instruct)

            # 1. calculate the token length of each segment in the source and target
            source_token_lt = [self.source_instruct_length]
            target_token_lt = [self.target_instruct_length]
            for idx, seg in enumerate(source_text):
                _token = self.tokenizer(seg, return_tensors='pt', padding=True, add_special_tokens=False)['input_ids']
                source_token_lt.append(_token.shape[1]) 
            for idx, seg in enumerate(target_text):
                _token = self.tokenizer(seg, return_tensors='pt', padding=True, add_special_tokens=False)['input_ids']
                target_token_lt.append(_token.shape[1])
            source_token_lt.append(self.end_Instruct_length) 
            target_token_lt.append(self.end_Instruct_length)

            input_token_len = sum(source_token_lt) + sum(target_token_lt)
            source_token_len = sum(source_token_lt)
            target_token_len = sum(target_token_lt)
            _lengths.append({'source_token_len':source_token_len,'source_seg_len':source_token_lt,
                             'target_token_len':target_token_len,'target_seg_len':target_token_lt,
                             'input_token_len':input_token_len,
                             'input_batch_len': input_batch_len})

            # 2. calculate the position_ids of each sample in the batch          
            if _mode == 'streaming':
                assert input_batch_len-source_token_len >= 0
                position_id = list(range(source_token_len))
                position_id.extend(list(range(self.pe_cache_length, self.pe_cache_length + input_batch_len-source_token_len)))
            elif _mode == 'batch':
                position_id = list(range(input_batch_len))
            assert len(position_id) == input_batch_len
            position_ids.append(position_id)
            

            # 3. calculate the attention mask index of each sample in the batch
            # mask_index == 1: location of target tokens in batch-processing mode
            # mask_index == 2: location of target tokens in streaming mode
            # mask_index == 0: location of source tokens & padding tokens, do not help to calculate loss
            # mask_index == -1: location of prompt tokens, predict the target prompt token in streaming mode
            if _mode == 'streaming':
                mask_index = torch.zeros((1, input_batch_len))
                mask_index[0, source_token_len : source_token_len+target_token_len] = 2
                mask_index[0, self.source_instruct_length:self.source_instruct_length+1] = -1
            elif _mode == 'batch':
                mask_index = torch.zeros((1, input_batch_len))
                mask_index[0, source_token_len : source_token_len+target_token_len] = 1
            attn_mask_index.append(mask_index)

        position_ids = torch.tensor(position_ids)
        return _lengths, position_ids, attn_mask_index

