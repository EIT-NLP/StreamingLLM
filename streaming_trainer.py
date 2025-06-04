import torch.nn.functional as F
from dataloader_hf import StreamingDataCollator
from transformers import Trainer


class StreamingSFTTrainer(Trainer):
    def __init__(self, *args, source_instruct_length, training_stage=None, loss_ignore_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_instruct_length = source_instruct_length
        self.loss_ignore_index = loss_ignore_index
        self.training_stage = training_stage
        print(f"🔍 Trainer received data_collator: {self.data_collator}")


    def compute_loss(self, model, inputs, num_items_in_batch = None):
        labels = inputs.get("labels")
        position_ids = inputs.get("position_ids", None)
        attn_mask_index = inputs.get("attn_mask_index", None)
        training_mode = inputs.get("training_mode", "unified")
        batch_size = labels.shape[0]
        _lengths = inputs.get("_lengths", None)
        _lengths_index = inputs.get("_lengths_index", None)
        wait_k = inputs.get("wait_k", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=position_ids,
            # ...
            training_mode = training_mode,
            is_training = True,
            _lengths = _lengths,
            _lengths_index = _lengths_index,
            labels = labels,
            attn_mask_index = attn_mask_index,
            wait_k = wait_k,
        )
        loss = outputs.loss      
        return loss
