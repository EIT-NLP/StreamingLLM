import torch
from transformers import Trainer


class StreamingSFTTrainer(Trainer):
    def __init__(self, *args, training_mode="batch", **kwargs):
        super().__init__(*args, **kwargs)
        self.training_mode = training_mode

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        model_inputs = dict(inputs)

        try:
            labels = model_inputs.pop("labels")
            position_ids = model_inputs.pop("position_ids", None)
            training_mode = model_inputs.pop("training_mode", self.training_mode)
            split_mode = model_inputs.pop("split_mode", None)
            model_inputs.pop("source_seg_len", None)
            model_inputs.pop("target_seg_len", None)
            lengths = model_inputs.pop("_lengths", None)

            outputs = model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                position_ids=position_ids,
                training_mode=training_mode,
                split_mode=split_mode,
                is_training=True,
                labels=labels,
                _lengths=lengths,
                output_attentions=True,
                num_items_in_batch=num_items_in_batch,
            )
            loss = outputs.loss
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise

            print(f"[Rank {self.args.local_rank}] OOM detected at step {self.state.global_step}")
            input_ids = model_inputs.get("input_ids")
            if input_ids is not None:
                print(f"OOM batch shape: {tuple(input_ids.shape)}")

            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=self.args.device, requires_grad=True)
            outputs = None

        return (loss, outputs) if return_outputs else loss
