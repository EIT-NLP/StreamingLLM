# coding=utf-8
# Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology).
#
# This file is a modified version of the original Gemma2 model implementation from:
# The Google Inc. HuggingFace Inc. team.
#
# Original license and copyright as follows:
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gemma2.modeling_gemma2 import *
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
import torch
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation.utils import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal
from transformers.cache_utils import Cache
from generation.generate import unified_PreTrainedModel 
if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward, prepare_fa2_from_position_ids, _upad_input
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

class QueryCache(Cache):
    def __init__(self) -> None:
        super().__init__()
        self.query_cache: List[torch.Tensor] = []
        self._seen_tokens = 0
    def update(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ):
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += query_states.shape[-2]

        # Update the cache
        if len(self.query_cache) <= layer_idx:
            self.query_cache.append(query_states)
        else:
            self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx], query_states], dim=-2)

        return self.query_cache[layer_idx]
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.query_cache) <= layer_idx:
            return 0
        return self.query_cache[layer_idx].shape[-2]

class DynamicCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()

    def pop(self):
        self._seen_tokens -= 1

        # Update the cache
        target_key_cache = []
        target_value_cache = []
        for key_cache,value_cache in zip(self.key_cache, self.value_cache):
            target_key_cache.append(key_cache[...,:-1,:])
            target_value_cache.append(value_cache[...,:-1,:])
        self.key_cache = target_key_cache
        self.value_cache = target_value_cache
 

@dataclass
class CausalLMOutputWithPast_stream(CausalLMOutputWithPast):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    source_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    similarity_scores_batch: Optional[torch.Tensor] = None
    similarity_labels_batch: Optional[torch.Tensor] = None
    similarity_loss: Optional[torch.Tensor] = None
    KL_loss: Optional[torch.Tensor] = None
    NLL_loss: Optional[torch.Tensor] = None

@dataclass
class BaseModelOutputWithPast_stream(BaseModelOutputWithPast):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    source_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    causal_mask: Optional[torch.Tensor] = None
    _wait: Optional[list] = None




class Gemma2Attention_stream(Gemma2Attention):
    def __init__(self, config: Gemma2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        self.rotary_emb = Gemma2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        is_training = kwargs.get("is_training", None)
        source_key_value = kwargs.get("source_key_value", None)
        ReadAction = kwargs.get("ReadAction", None)
        generate_mode = kwargs.get("generate_mode", "batch")


        if is_training:
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

            if self.config.attn_logit_softcapping is not None:
                attn_weights = attn_weights / self.config.attn_logit_softcapping
                attn_weights = torch.tanh(attn_weights)
                attn_weights = attn_weights * self.config.attn_logit_softcapping

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

        else:
            ...
            # To do: add inference code here
            if generate_mode == 'batch':
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


                kv_seq_len = key_states.shape[-2]
                if past_key_value is not None:
                    if self.layer_idx is None:
                        raise ValueError(
                            f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                            "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                            "with a layer index."
                        )
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
                # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                if position_embeddings is None:
                    logger.warning_once(
                        "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                        "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                        "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                        "removed and `position_embeddings` will be mandatory."
                    )
                    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len) # need to update
                    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
                else:
                    cos, sin = position_embeddings

                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                # repeat k/v heads if n_kv_heads < n_heads
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

                if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                        f" {attn_weights.size()}"
                    )

                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, value_states)

                if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(bsz, q_len, -1)

                attn_output = self.o_proj(attn_output)

                if not output_attentions:
                    attn_weights = None
            elif generate_mode == 'streaming':
                assert ReadAction is not None
                # assert remove_last_token is not None

                bsz, q_len, _ = hidden_states.size()

                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
                
                ## milti-head
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                kv_seq_len = key_states.shape[-2]  
                if ReadAction:          
                    if source_key_value is not None:
                        if self.layer_idx is None:
                            raise ValueError(
                                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                                "with a layer index."
                            )
                        kv_seq_len += source_key_value.get_usable_length(kv_seq_len, self.layer_idx) 
                elif not ReadAction:
                    if past_key_value is not None:
                        if self.layer_idx is None:
                            raise ValueError(
                                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                                "with a layer index."
                            )
                        kv_seq_len = kv_seq_len + past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        
                cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)  
                if ReadAction:              
                    cache_kwargs_source = {"sin": sin, "cos": cos, "cache_position": None}  # Specific to RoPE models
                    key_states, value_states = source_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs_source) 
                elif not ReadAction: 
                    cache_kwargs_target = {"sin": sin, "cos": cos, "cache_position": None}  # Specific to RoPE models
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs_target)  


                # repeat k/v heads if n_kv_heads < n_heads
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                        f" {attn_weights.size()}"
                    )

                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights += causal_mask

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

                attn_output = torch.matmul(attn_weights, value_states)

                if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )

                attn_output = attn_output.transpose(1, 2).contiguous()
                # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                attn_output = attn_output.reshape(bsz, q_len, -1)

                attn_output = self.o_proj(attn_output)

                if not output_attentions:
                    attn_weights = None

        return attn_output, attn_weights, source_key_value, past_key_value



class Gemma2FlashAttention2_stream(Gemma2Attention_stream):
    """
    Gemma2 flash attention module. This module inherits from `Gemma2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ...
        # To do: add Gemma2FlashAttention2_stream code here
        

class Gemma2SdpaAttention_stream(Gemma2Attention_stream):
    """
    Gemma2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Gemma2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Gemma2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ...
        # To do: add Gemma2SdpaAttention_stream code here





Gemma2_ATTENTION_CLASSES_STREAM = {
    "eager": Gemma2Attention_stream,
    "flash_attention_2": Gemma2FlashAttention2_stream,
    "sdpa": Gemma2SdpaAttention_stream,
}




class Gemma2DecoderLayer_stream(Gemma2DecoderLayer):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__(config,layer_idx)
        self.self_attn = Gemma2_ATTENTION_CLASSES_STREAM[config._attn_implementation](config, layer_idx=layer_idx)
   
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # Flash-attn is a 2D tensor
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask[:, -self.sliding_window :]
            else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                if attention_mask.shape[-1] <= 1:  # when decoding
                    attention_mask = attention_mask[:, :, :, -self.sliding_window :]

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, source_key_value, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        hidden_states = self.post_attention_layernorm(attn_outputs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            # outputs += (present_key_value,)
            outputs += (source_key_value,past_key_value)

        return outputs

GEMMA2_STREAMING_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


class Gemma2Model_stream(Gemma2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Gemma2DecoderLayer`]

    Args:
        config: Gemma2Config
    """

    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer_stream(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rotary_emb = Gemma2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


    def generate_attention_mask(self, input_batch_len, source_seg_len, target_seg_len, wait_tokens_list, training_mode = 'streaming', device = 'cpu'):
        """ 
        for streaming training mode
        input_batch_len: int, the tokens length of input batch
        source_seg_len: list, the tokens length of source sentence
        target_seg_len: list, the tokens length of target sentence
        wait_tokens_list: list, the tokens length of wait tokens in target sentence
        """
        assert training_mode in ['streaming']
        if not isinstance(wait_tokens_list, list):
            wait_tokens_list = wait_tokens_list.tolist()
        assert len(wait_tokens_list) == len(target_seg_len)

        if training_mode == 'streaming':
            # Step 1: Calculate attn_mask
            prompt_len = source_seg_len[0]
            source_token_len = sum(source_seg_len)
            target_token_len = sum(target_seg_len)
            total_len = source_token_len + target_token_len
            streaming_start = source_token_len
            inf = -3e+38
            attn_mask = torch.triu(torch.ones(input_batch_len, input_batch_len), diagonal=1).to(device)*inf
            if total_len < input_batch_len:
                attn_mask[total_len:, :] = inf
            for index,num in enumerate(target_seg_len):
                wait_words = wait_tokens_list[index]
                wait_tokens = sum(source_seg_len[:wait_words]) # prompt as 1 word
                attn_mask[streaming_start:streaming_start+num, wait_tokens:source_token_len] = inf
                streaming_start += num

            # Need to move the source attention mask in streaming part up by one
            attn_mask[source_token_len : source_token_len + target_token_len-1, :source_token_len] = attn_mask[source_token_len+1 :source_token_len + target_token_len, :source_token_len]

            # # Step 2: Calculate attn_mask_index
            # # attn_mask_index == 2: location of target tokens in streaming mode
            # # attn_mask_index == 0: location of source tokens & padding tokens, do not help to calculate loss
            # # attn_mask_index == -1: location of prompt tokens, predict the target prompt token in streaming mode
            # attn_mask_index = torch.zeros((1, input_batch_len)).to(device)
            # attn_mask_index[0, source_token_len:total_len] = 2
            # attn_mask_index[0, prompt_len:prompt_len+1] = -1

        return attn_mask#, attn_mask_index
        
    def generate_wait_words(self,source_seg_len, target_seg_len, wait_k):
        wait_tokens_list = []
        source_word_total = len(source_seg_len)

        for tgt_idx in range(len(target_seg_len)):
            source_words_waited = min(wait_k + tgt_idx, source_word_total)
            wait_tokens_list.append(source_words_waited)

        return wait_tokens_list
    
    @add_start_docstrings_to_model_forward(GEMMA2_STREAMING_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # ...
        training_mode: Optional[str] = 'streaming', # 'batch' or 'streaming'
        generate_mode: Optional[str] = 'batch', # 'batch' or 'streaming'
        split_mode: Optional[str] = 'word', # 'word', 'token'
        is_training: Optional[bool] = False,
        source_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        ReadAction: Optional[bool] = False,
        remove_last_token: Optional[bool] = False,
        _lengths: Optional[List[dict]] = None, # length info of source and target tokens
        wait_k: Optional[float] = None, # the number of wait tokens
    ) -> Union[Tuple, BaseModelOutputWithPast_stream]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)


        # embed positions
        hidden_states = inputs_embeds

        # # # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if is_training:
            if training_mode == 'batch':
                causal_mask = self._update_causal_mask(
                    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
                )
                _wait = None
            elif training_mode in ['streaming']:
                if split_mode == 'word':
                    assert wait_k is not None
                    inf = -3e+38
                    batch_size = inputs_embeds.shape[0]
                    input_batch_len = _lengths[0]['input_batch_len']
                    causal_mask = torch.triu(torch.ones(input_batch_len, input_batch_len), diagonal=1).to(inputs_embeds.device)*inf
                    causal_mask = causal_mask.repeat(batch_size, 1, 1, 1)
                    _wait = []
                    for index, len_dict in enumerate(_lengths):
                        source_len, source_seg_len = len_dict['source_token_len'], len_dict['source_seg_len']
                        target_len, target_seg_len = len_dict['target_token_len'], len_dict['target_seg_len']
                        wait_tokens_list = self.generate_wait_words(source_seg_len, target_seg_len, wait_k)
                        attn_mask = self.generate_attention_mask(input_batch_len, source_seg_len, target_seg_len, 
                                                                        wait_tokens_list, training_mode = training_mode, device = inputs_embeds.device)
                        causal_mask[index] = attn_mask
                        _wait.append(wait_tokens_list)

        else:
            if generate_mode == 'batch':
                cache_position = torch.arange(history_source_length, current_length, dtype=torch.long, device=cache_position.device)
                causal_mask = self._update_causal_mask(
                    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
                )
                _wait = None
            elif generate_mode == 'streaming':
                _wait = None
                assert ReadAction is not None
                if ReadAction:
                    history_source_length = source_key_values.get_seq_length()
                    current_length = history_source_length + inputs_embeds.shape[1]
                    cache_position = torch.arange(history_source_length, current_length, dtype=torch.long, device=cache_position.device)
                    causal_mask = self._update_causal_mask(attention_mask[:,:current_length], inputs_embeds, cache_position, past_key_values, output_attentions)
                else:
                    causal_mask = None
            else:
                raise ValueError(f"generate_mode should be 'batch' or 'streaming', but got {generate_mode}")


        # normalized
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    # ...
                    is_training = is_training,
                    source_key_value = source_key_values,
                    ReadAction = ReadAction,
                    remove_last_token = remove_last_token,
                    generate_mode = generate_mode,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast_stream(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            source_key_values=source_key_values,
            causal_mask = causal_mask,
            _wait = _wait,
        )




class Gemma2ForCausalLM_stream(unified_PreTrainedModel, Gemma2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Gemma2Model_stream(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.source_key_values = None
        self.target_key_values = None
        self.past_key_values = None

    def set_cache(self):
        self.past_key_values = DynamicCache()
        self.source_key_values = DynamicCache()
        self.target_key_values = DynamicCache()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # ...
        training_mode: Optional[str] = 'streaming', # 'batch' or 'streaming'
        generate_mode: Optional[str] = 'batch', # 'batch' or 'streaming'
        split_mode: Optional[str] = 'word', # 'word', 'token'
        is_training: Optional[bool] = False,
        source_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        ReadAction: Optional[bool] = False,
        remove_last_token: Optional[bool] = False,
        _lengths: Optional[List[dict]] = None, # length info of source and target tokens
        _lengths_index: Optional[torch.tensor] = None, # length info of source and target tokens
        attn_mask_index: Optional[torch.tensor] = None, # attn_mask_index
        wait_k: Optional[float] = None, # the number of wait tokens
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast_stream]:
        if _lengths_index is not None: # split GPU manually
            _lengths = [_lengths[i] for i in _lengths_index]
            if is_training:
                attn_mask_index = [attn_mask_index[i] for i in _lengths_index]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            # ...
            training_mode = training_mode, # 'batch' or 'streaming'
            generate_mode = generate_mode, # 'batch' or 'streaming'
            split_mode = split_mode, # 'word', 'token'
            is_training = is_training,
            source_key_values = source_key_values,
            ReadAction = ReadAction,
            remove_last_token = remove_last_token,
            _lengths = _lengths,
            wait_k = wait_k,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        

        '''NLL loss'''
        loss = None
        if labels is not None and attn_mask_index is not None:
            loss = self.loss_function(training_mode=training_mode, logits=logits, labels=labels, attn_mask_index=attn_mask_index)


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        


        return CausalLMOutputWithPast_stream(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def loss_function(self, training_mode, logits, labels, attn_mask_index):
        '''NLL loss'''
        # **batch mode**: calculate loss for `attn_mask_index > 0`
        if training_mode == "batch":
            shift_logits = logits.contiguous()  # [batch, seq_len-1, vocab]
            shift_labels = labels.contiguous()  # [batch, seq_len-1]
            attn_mask_index = torch.cat(attn_mask_index)
            loss_mask = attn_mask_index.contiguous().view(-1)
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch*seq_len, vocab_size]
            shift_labels = shift_labels.view(-1)  # [batch*seq_len]
            '''for batch mode'''
            loss_mask_batch_labels = torch.where(loss_mask==1)[0]
            loss_mask_batch_logits = loss_mask_batch_labels.clone() - 1
            shift_logits_batch = shift_logits[loss_mask_batch_logits]
            shift_labels_batch = shift_labels[loss_mask_batch_labels]

            loss_NLL = F.cross_entropy(shift_logits_batch, shift_labels_batch)

        # **streaming mode**: calculate loss for `attn_mask_index == 2` and ensure that the first token of the streaming target is predicted at the position of `source_instruct_length`
        elif training_mode == "streaming":
            shift_logits = logits.contiguous()  # [batch, seq_len-1, vocab]
            shift_labels = labels.contiguous()  # [batch, seq_len-1]
            attn_mask_index = torch.cat(attn_mask_index)
            loss_mask = attn_mask_index.contiguous().view(-1)
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch*seq_len, vocab_size]
            shift_labels = shift_labels.view(-1)  # [batch*seq_len]
            '''for streaming mode'''
            loss_mask_streaming_labels = torch.where(loss_mask==2)[0]
            # find the first index of attn_mask_index == 2
            loss_mask_streaming_logits = (attn_mask_index == 2)
            # calculate the number of index==2 in each sample, and find the first index of each sample in flattened tensor
            count_per_sample = loss_mask_streaming_logits.sum(dim=1)
            first_idx = torch.cat([torch.tensor([0], device=count_per_sample.device), count_per_sample[:-1]])
            first_idx = first_idx.cumsum(dim=0)
            # replace the first index of index==2 with the index of -1
            loss_mask_streaming_logits = loss_mask_streaming_labels.clone() - 1
            loss_mask_streaming_logits[first_idx] = torch.where(loss_mask==-1)[0]
            shift_logits_streaming = shift_logits[loss_mask_streaming_logits]
            shift_labels_streaming = shift_labels[loss_mask_streaming_labels]

            loss_NLL = F.cross_entropy(shift_logits_streaming, shift_labels_streaming)
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")
        
        return loss_NLL

