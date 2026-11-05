# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JetSpec++ token corruption and Qwen3VL tap-aligned hidden-state adapter."""

from copy import deepcopy

import torch
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from .modeling_dflash import DFlashDecoderLayer

__all__ = ["JetSpecAdapter", "jetspec_training_tokens"]


def jetspec_training_tokens(input_ids, anchors, block_size, excluded_ids):
    """Keep a clean prefix and replace the suffix with tokens sampled from each anchor's past."""
    bsz, seq_len = input_ids.shape
    offsets = torch.arange(block_size, device=input_ids.device)
    positions = (anchors[..., None] + offsets).clamp(max=seq_len - 1)
    clean = input_ids.gather(1, positions.flatten(1)).view(*anchors.shape, block_size)
    allowed = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in excluded_ids:
        allowed &= input_ids != token_id
    cumulative = allowed.long().cumsum(-1).contiguous()
    counts = cumulative.gather(1, anchors)
    draws = torch.rand_like(clean, dtype=torch.float32) * counts[..., None]
    sampled_positions = torch.searchsorted(cumulative, (draws.long() + 1).flatten(1))
    noise = input_ids.gather(1, sampled_positions.clamp(max=seq_len - 1)).view_as(clean)
    # P(m=0)=1/4; the remaining mass is uniform over m=1,...,block_size-1.
    prefix = torch.randint(1, block_size, anchors.shape, device=input_ids.device)
    prefix = torch.where(torch.rand_like(anchors, dtype=torch.float32) < 0.25, 0, prefix)
    return torch.where(offsets <= prefix[..., None], clean, noise).view(bsz, -1)


class _AdapterLayer(DFlashDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.prefix_norm = deepcopy(self.input_layernorm)

    def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask=None):
        return super().forward(
            hidden_states, self.prefix_norm(target_hidden), position_embeddings, attention_mask
        )


class JetSpecAdapter(nn.Module):
    """Predict ordered target taps from tokens, feeding each prediction into the next stream."""

    def __init__(self, text_model, target_layer_ids, depths):
        """Initialize stream MLPs from target taps and their attention from matching context layers."""
        super().__init__()
        self.config = deepcopy(text_model.config)
        self.config._attn_implementation = "sdpa"
        self.config.attention_sink_bias = False
        self.target_layer_ids = list(target_layer_ids)
        self.depths = list(depths)
        self.streams = nn.ModuleList()
        for tap, depth in zip(target_layer_ids, depths):
            stream = nn.ModuleList()
            for index in range(depth):
                source_id = tap if index == 0 else tap - depth + 1 + index
                source = text_model.layers[source_id]
                attention_source = text_model.layers[tap + 1] if index == 0 else source
                layer = _AdapterLayer(self.config, source_id)
                if next(source.parameters()).device.type != "meta":
                    layer.self_attn.load_state_dict(attention_source.self_attn.state_dict())
                    layer.input_layernorm.load_state_dict(
                        attention_source.input_layernorm.state_dict()
                    )
                    layer.prefix_norm.load_state_dict(attention_source.input_layernorm.state_dict())
                    layer.mlp.load_state_dict(source.mlp.state_dict())
                    layer.post_attention_layernorm.load_state_dict(
                        source.post_attention_layernorm.state_dict()
                    )
                stream.append(layer)
            self.streams.append(stream)

    @property
    def layers(self):
        """Expose attention dimensions to the shared block-mask builder."""
        return [layer for stream in self.streams for layer in stream]

    def _maybe_init_rotary_emb(self, device=None):
        if not hasattr(self, "rotary_emb"):
            self.rotary_emb = Qwen3VLTextRotaryEmbedding(self.config, device=device)

    def forward(self, noise_embedding, target_hidden, position_ids, attention_mask=None):
        """Return raw tap predictions; prefix states are true target states during training."""
        self._maybe_init_rotary_emb(noise_embedding.device)
        rotary = self.rotary_emb(noise_embedding, position_ids)
        prefixes = target_hidden.split(self.config.hidden_size, dim=-1)
        hidden = noise_embedding
        predictions = []
        for stream, prefix in zip(self.streams, prefixes):
            for layer in stream:
                hidden = layer(hidden, prefix, rotary, attention_mask)
            predictions.append(hidden)
        return torch.cat(predictions, dim=-1)
