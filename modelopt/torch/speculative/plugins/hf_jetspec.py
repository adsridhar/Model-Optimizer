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

"""Separate JetSpec++ drafter and adapter training for Cosmos3 / Qwen3VL targets."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from ..dflash.conversion import JetSpecDMRegistry
from .dflash_flex_attention import build_draft_block_mask
from .hf_dflash import (
    HFDFlashModel,
    _block_positions,
    _dense_additive_mask,
    _draft_mask_mod,
    _gather_rows,
    _multimodal_forward_kwargs,
)
from .modeling_jetspec import JetSpecAdapter, jetspec_training_tokens

__all__ = ["HFJetSpecModel"]


def _adapter_distance(predicted, truth, weights):
    predicted, truth = predicted.float(), truth.float()
    distance = F.smooth_l1_loss(predicted, truth, reduction="none").mean(-1)
    distance = distance + 1 - F.cosine_similarity(predicted, truth, dim=-1)
    return (distance * weights).sum()


@JetSpecDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFJetSpecModel(HFDFlashModel):
    """Reuse DFlash training for the drafter; regress target taps in the adapter stage."""

    def modify(self, config):
        """Validate the supported target and initialize one independently trained stage."""
        arch = config.dflash_architecture_config
        stage = arch.get("jetspec_stage", "drafter")
        if stage not in ("drafter", "adapter"):
            raise ValueError("jetspec_stage must be 'drafter' or 'adapter'")
        if getattr(self.config, "model_type", None) not in ("qwen3_vl", "cosmos3_omni"):
            raise ValueError("JetSpec training currently supports only Cosmos3 / Qwen3VL")
        if config.dflash_offline or not config.dflash_freeze_base_model:
            raise ValueError("JetSpec training requires an online, frozen target")
        if config.dflash_draft_attention != "causal" or config.dflash_block_size < 2:
            raise ValueError("JetSpec requires causal draft attention and block_size >= 2")
        if config.dflash_attention_sink or config.dflash_swa_window_size is not None:
            raise ValueError("JetSpec training currently requires full attention without sinks")
        if stage == "adapter":
            taps = arch.get("target_layer_ids")
            depths = arch.get("adapter_depths")
            if (
                not taps
                or not depths
                or len(depths) != len(taps)
                or taps != sorted(set(taps))
                or any(type(d) is not int or d < 1 or d > t + 1 for t, d in zip(taps, depths))
                or taps[-1] >= self.config.text_config.num_hidden_layers - 1
            ):
                raise ValueError(
                    "Adapter requires ordered target_layer_ids and valid adapter_depths"
                )
            if config.dflash_init_checkpoint or config.dflash_use_torch_compile:
                raise ValueError(
                    "Adapter initializes from target layers; draft warm-start/compile is unsupported"
                )
        self.jetspec_stage = stage
        self._noise_excluded_ids = tuple(
            value
            for name in (
                "pad_token_id",
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            )
            if (value := getattr(self.config, name, None)) is not None
        )
        super().modify(config)

    def _build_draft_module(self, dflash_config):
        if self.jetspec_stage == "adapter":
            return JetSpecAdapter(
                self._base_model, self.target_layer_ids, dflash_config.adapter_depths
            )
        return super()._build_draft_module(dflash_config)

    def _build_noise_embedding(self, input_ids, anchor_positions, block_keep_mask, n_blocks):
        tokens = jetspec_training_tokens(
            input_ids, anchor_positions, self.dflash_block_size, self._noise_excluded_ids
        )
        return self._base_model_embeddings(tokens)

    def _build_draft_attention_mask(
        self, seq_len, anchor_positions, block_keep_mask, n_blocks, dtype, device, window=None
    ):
        if self.jetspec_stage == "drafter":
            return super()._build_draft_attention_mask(
                seq_len, anchor_positions, block_keep_mask, n_blocks, dtype, device, window
            )
        block_size = self.dflash_block_size
        rule = _draft_mask_mod(seq_len, anchor_positions, block_keep_mask, block_size, None, True)

        def adapter_rule(b, h, q, kv):
            return rule(b, h, q, kv) & (
                (kv < seq_len) | ((kv - seq_len) % block_size < q % block_size)
            )

        bsz, q_len = anchor_positions.shape[0], n_blocks * block_size
        if self.dflash_use_flex_attention:
            return build_draft_block_mask(
                adapter_rule,
                bsz,
                q_len,
                seq_len + q_len,
                device,
                head_dims={layer.self_attn.head_dim for layer in self.dflash_module.layers},
            )
        return _dense_additive_mask(adapter_rule, bsz, q_len, seq_len + q_len, dtype, device)

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs
    ):
        """Run DFlash distillation or independently regress the adapter's target taps."""
        if self.jetspec_stage == "drafter":
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                **kwargs,
            )
        seq_len = input_ids.shape[1]
        if seq_len % self.dflash_block_size:
            raise ValueError("training_seq_len must be divisible by dflash_block_size")
        mm_kwargs = _multimodal_forward_kwargs(kwargs)
        with torch.no_grad():
            if position_ids is None:
                position_ids = self.model.compute_3d_position_ids(
                    input_ids=input_ids,
                    inputs_embeds=None,
                    attention_mask=attention_mask,
                    **{
                        k: v
                        for k, v in mm_kwargs.items()
                        if k in ("image_grid_thw", "video_grid_thw", "mm_token_type_ids")
                    },
                )
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=input_ids.device)[None].expand_as(
                    input_ids
                )
            # Share positions with the target, without projecting its states to vocabulary logits.
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
                **mm_kwargs,
            )
            target_hidden = self._target_hidden(output.hidden_states)
            # Some collators include a leading text-position plane before the three RoPE planes.
            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                position_ids = position_ids[1:]

        mask = torch.ones_like(input_ids, dtype=torch.float32)
        if attention_mask is not None:
            mask *= attention_mask
        if labels is not None:
            mask *= labels != -100
        anchors, keep = self._sample_anchor_positions(seq_len, mask, input_ids.device)
        if anchors.shape[1] == 0 or not keep.any():
            return ModelOutput(loss=self._draft_zero_loss(input_ids.device))
        positions, inside = _block_positions(anchors, self.dflash_block_size, 0, seq_len)
        flat_positions = positions.flatten(1)
        tokens = input_ids.gather(1, flat_positions)
        embedding = self._base_model_embeddings(tokens)
        draft_positions = position_ids.gather(
            -1, flat_positions.expand(*position_ids.shape[:-2], -1, -1)
        )
        full_positions = torch.cat([position_ids, draft_positions], dim=-1)
        attn_mask = self._build_draft_attention_mask(
            seq_len, anchors, keep, anchors.shape[1], target_hidden.dtype, input_ids.device
        )
        predicted = self.dflash_module(embedding, target_hidden, full_positions, attn_mask)
        truth = _gather_rows(target_hidden, flat_positions)
        weights = (keep[..., None] & inside).flatten(1) * mask.gather(1, flat_positions)
        # Bound fp32 loss temporaries to a single block rather than all anchors x all taps.
        loss = predicted.new_zeros((), dtype=torch.float32)
        for start in range(0, predicted.shape[1], self.dflash_block_size):
            end = start + self.dflash_block_size
            loss = loss + checkpoint(
                _adapter_distance,
                predicted[:, start:end],
                truth[:, start:end],
                weights[:, start:end],
                use_reentrant=False,
            )
        return ModelOutput(loss=loss / weights.sum().clamp(min=1))

    def pseudo_speculative_generate(self, *args, **kwargs):
        """Reject the mask-only DFlash validator for token-input JetSpec checkpoints."""
        raise NotImplementedError(
            "JetSpec training requires a separate refinement-aware validation runner"
        )

    def get_exporter(self):
        """Export stage-specific weights and metadata."""
        from modelopt.torch.export.plugins.hf_spec_export import JetSpecExporter

        return JetSpecExporter(self)
