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

"""Single-turn multimodal policies for Cosmos3 Super and a Qwen3.5 draft."""

import copy
import math

import torch
from torch.nn.functional import pad

from modelopt.torch.utils.plugins.transformers_dataset import VisionLanguageDataCollator

__all__ = [
    "MultimodalSparsePolicyCollator",
    "batch_multimodal_inputs",
    "normalize_messages",
    "process_multimodal_messages",
    "retarget_visual_tokens",
]


def normalize_messages(example):
    """Resolve LLaVA image placeholders without changing the response text."""
    messages = copy.deepcopy(example["messages"])
    if [m["role"] for m in messages].count("assistant") != 1 or messages[-1]["role"] != "assistant":
        raise ValueError("Multimodal external drafts currently require one final assistant turn.")
    # Text-only examples can discuss literal HTML/XML <image> tags.
    if not example.get("images"):
        return messages
    images = iter(example["images"])
    for message in messages:
        if message["role"] == "assistant":
            continue
        content = message["content"]
        if not isinstance(content, str) or "<image>" not in content:
            continue
        parts = content.split("<image>")
        content = []
        for index, part in enumerate(parts):
            if index:
                image = next(images, None)
                if image is None:
                    raise ValueError("An <image> placeholder has no corresponding image path.")
                content.append({"type": "image", "image": image})
            if part:
                content.append({"type": "text", "text": part})
        message["content"] = content
    if next(images, None) is not None:
        raise ValueError("Image paths remain without corresponding <image> placeholders.")
    return messages


def retarget_visual_tokens(config, tokenizer):
    """Keep Qwen's visual placeholders consistent with its swapped language vocabulary."""
    ids = {}
    vocab = tokenizer.get_vocab()
    for name, token in (
        ("image_token_id", "<|image_pad|>"),
        ("video_token_id", "<|video_pad|>"),
        ("vision_start_token_id", "<|vision_start|>"),
        ("vision_end_token_id", "<|vision_end|>"),
    ):
        if token not in vocab:
            raise ValueError(f"Target tokenizer is missing {token}.")
        ids[name] = tokenizer.convert_tokens_to_ids(token)
    for name, value in ids.items():
        setattr(config, name, value)


def process_multimodal_messages(collator, messages):
    """Honor per-record FPS without leaking the override into subsequent examples."""
    video_processor = collator.processor.video_processor
    default_fps = video_processor.fps
    rates = {
        part.get("fps") if part.get("fps") is not None else default_fps
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "video"
    }
    if len(rates) > 1:
        raise ValueError("Videos in one example must use the same FPS.")
    fps = next(iter(rates), default_fps)
    if fps is not None and (
        not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
    ):
        raise ValueError("Video FPS must be a finite positive number.")
    try:
        video_processor.fps = fps
        return VisionLanguageDataCollator.__call__(collator, [{"messages": messages}])
    finally:
        video_processor.fps = default_fps


def batch_multimodal_inputs(samples, pad_token_id):
    """Right-pad processed conversations and concatenate their visual inputs."""
    length = max(sample["input_ids"].shape[1] for sample in samples)
    media_keys = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
    sequence_keys = {
        "input_ids",
        "attention_mask",
        "assistant_masks",
        "mm_token_type_ids",
        "token_type_ids",
    }
    if len(samples) == 1 and samples[0].keys() <= media_keys | sequence_keys:
        return dict(samples[0])
    batch = {}
    for key in set().union(*(sample.keys() for sample in samples)):
        if key in media_keys:
            batch[key] = torch.cat([sample[key] for sample in samples if key in sample])
        elif key in sequence_keys:
            value = pad_token_id if key == "input_ids" else 0
            batch[key] = torch.cat(
                [
                    pad(sample[key], (0, length - sample[key].shape[1]), value=value)
                    for sample in samples
                ]
            )
        else:
            raise ValueError(f"Unsupported Qwen processor output: {key}.")
    return batch


class MultimodalSparsePolicyCollator(VisionLanguageDataCollator):
    """Place response-indexed policies onto the draft's own visual-token sequence."""

    def __init__(self, processor, tokenizer, train_len, chat_template=None):
        """Use the draft's visual preprocessing with the shared target vocabulary."""
        # Dynamic lengths avoid padding the full-vocabulary logits to 8192 for short replies.
        super().__init__(
            processor=processor, train_len=None, answer_only_loss=True, chat_template=chat_template
        )
        if self.max_prompt_tokens is not None or self.max_assistant_tokens is not None:
            raise ValueError("Text truncation is incompatible with cached response policies.")
        self.max_length = train_len
        # Reconstruct to refresh cached visual-token ids as well as the tokenizer.
        self.processor = self.processor.__class__(
            image_processor=self.processor.image_processor,
            video_processor=self.processor.video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template or tokenizer.chat_template,
        )
        self.tokenizer = tokenizer
        self._post_process_chat_template()
        self.processor.chat_template = chat_template or tokenizer.chat_template
        self._post_process_tokenizer()
        if hasattr(self, "_cached_assistant_marker_specs"):
            del self._cached_assistant_marker_specs
        self._verify_generation_tags()

    def process_messages(self, messages):
        """Process one conversation for length checks before collecting teacher policies."""
        return process_multimodal_messages(self, messages)

    def __call__(self, features):
        """Process media and verify response ids before attaching teacher policies."""
        # Each record keeps its own video sampling rate before sequences are padded.
        samples = [self.process_messages(feature["messages"]) for feature in features]
        length = max(sample["input_ids"].shape[1] for sample in samples)
        if length > self.max_length:
            raise ValueError(
                f"Draft multimodal sequence has {length} tokens, exceeding "
                f"training_seq_len={self.max_length}; re-filter the corpus for both processors."
            )
        batch = batch_multimodal_inputs(samples, self.tokenizer.pad_token_id)
        ids = batch["input_ids"]
        mask = batch.pop("assistant_masks").bool()
        shape = (*ids.shape, features[0]["teacher_topk_tok"].shape[-1])
        tok = torch.zeros(shape, dtype=torch.long)
        prob = torch.zeros(shape, dtype=torch.float)
        for row, feature in enumerate(features):
            if not torch.equal(ids[row, mask[row]], feature["gen_ids"]):
                raise ValueError(
                    "Cached teacher response tokens differ from the draft response tokens."
                )
            if not mask[row].any() or mask[row, 0]:
                raise ValueError("The response must have a nonempty preceding prompt.")
            tok[row, mask[row]] = feature["teacher_topk_tok"]
            prob[row, mask[row]] = feature["teacher_topk_prob"]
        batch.update(
            loss_mask=mask.float(),
            teacher_topk_tok=tok,
            teacher_topk_prob=prob,
            # HF selects the same logit positions for every row; unscored rows stay masked.
            response_positions=mask.any(dim=0).nonzero().flatten(),
        )
        return batch
