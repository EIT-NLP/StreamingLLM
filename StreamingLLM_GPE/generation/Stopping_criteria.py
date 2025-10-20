# Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology).
# Copy and modified from 'Simul-LLM' repository.


import torch
from transformers.generation.stopping_criteria import StoppingCriteria


class StopTokenCriteria(StoppingCriteria):
    def __init__(
            self,
            tokenizer,
            max_new_tokens: int,
            end_Instruct = '<|end|>'
        ):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.end_Instruct = end_Instruct


    def __call__(self, target_ids: torch.LongTensor, scores: torch.FloatTensor, token_count, **kwargs) -> bool:
        token_pred = self.tokenizer.decode(target_ids[0][-1:])
        token_preds = self.tokenizer.decode(target_ids[0])
        is_done = False
        remove_last_token = False
        terminating_tok = [".", ",", ":", ";", "?", "!"]
        # if ' ' in token_preds[1:] or token_pred in terminating_tok or '<|end|>' in token_pred or token_count >= self.max_new_tokens:
        if ' ' in token_preds[1:] or token_pred in terminating_tok or self.end_Instruct in token_pred or token_count >= self.max_new_tokens:
            is_done = True
        # if is_done and target_ids[0].shape[0] >= 2:
        if ' ' in token_preds[1:] and target_ids[0].shape[0] >= 2: 
            remove_last_token = True
        return torch.tensor(is_done), torch.tensor(remove_last_token)