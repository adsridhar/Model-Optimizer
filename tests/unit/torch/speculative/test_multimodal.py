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

"""Response alignment across different visual sequence lengths."""

import copy
import json

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessorFast,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from modelopt.torch.speculative.external.multimodal import (
    MultimodalSparsePolicyCollator,
    normalize_messages,
    process_multimodal_messages,
)
from modelopt.torch.speculative.external.sparse_data import SparsePolicyDataset
from modelopt.torch.utils.plugins.transformers_dataset import VisionLanguageDataCollator


@pytest.mark.parametrize("images", [None, []])
def test_text_image_tags_are_not_media_placeholders(images):
    """Literal HTML/XML tags in text-only records must survive normalization."""
    record = {
        "messages": [
            {"role": "user", "content": "Explain <image>flickr</image> in this XML."},
            {"role": "assistant", "content": "The <image> element stores the source."},
        ]
    }
    if images is not None:
        record["images"] = images
    assert normalize_messages(record) == record["messages"]
    record["images"] = ["actual-image.png"]
    record["messages"][0]["content"] += "<image>"
    with pytest.raises(ValueError):
        normalize_messages(record)


@pytest.fixture
def processors(tmp_path):
    """Save two offline processors with different image resolutions."""
    special = [
        "[UNK]",
        "[PAD]",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    words = [*special, "user", "assistant", "Describe", "red", "car"]
    backend = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=special[2:],
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' }}"
        "{% if m['content'] is string %}{{ m['content'] }}{% else %}"
        "{% for c in m['content'] %}{% if c['type'] == 'image' %}"
        "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
        "{% elif c['type'] == 'video' %}{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
        "{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}"
        "{{ '<|im_end|>\\n' }}{% endfor %}"
    )
    for name, pixels in [("teacher", 4096), ("draft", 1024)]:
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessorFast(
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                min_pixels=pixels,
                max_pixels=pixels,
            ),
            video_processor=Qwen3VLVideoProcessor(
                fps=2, size={"shortest_edge": 1024, "longest_edge": 4096}
            ),
            tokenizer=tokenizer,
            chat_template=tokenizer.chat_template,
        )
        processor.save_pretrained(tmp_path / name)
    return tokenizer


@pytest.mark.parametrize("batch_size", [1, 2])
def test_visual_policy_cache_aligns_response_and_rejects_changed_tokens(
    tmp_path, processors, batch_size
):
    """Use real processors: teacher visual positions must not determine draft positions."""
    tokenizer = processors
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "red").save(image)
    messages = normalize_messages(
        {
            "messages": [
                {"role": "user", "content": "Describe<image>"},
                {"role": "assistant", "content": "red car"},
            ],
            "images": [str(image)],
        }
    )
    teacher = VisionLanguageDataCollator(
        processor=str(tmp_path / "teacher"),
        train_len=None,
        answer_only_loss=True,
    )([{"messages": messages}])
    mask = teacher["assistant_masks"][0].bool()
    gen = teacher["input_ids"][0, mask].tolist()
    first = int(mask.nonzero()[0])
    record = {
        "messages": messages,
        "prompt_ids": teacher["input_ids"][0, :first].tolist(),
        "gen_ids": gen,
        "topk_tok": [[token] for token in gen],
        "topk_lp": [[0.0] for _ in gen],
    }
    path = tmp_path / "policy.jsonl"
    path.write_text(json.dumps(record) + "\n")
    dataset = SparsePolicyDataset([str(path)], max_length=128)
    collator = MultimodalSparsePolicyCollator(str(tmp_path / "draft"), tokenizer, 128)
    features = [dataset[0]]
    if batch_size == 2:
        second = copy.deepcopy(dataset[0])
        second["messages"][0]["content"][0]["text"] = "Describe Describe Describe"
        second["messages"][-1]["content"] = "red"
        processed = collator.process_messages(second["messages"])
        second["gen_ids"] = processed["input_ids"][processed["assistant_masks"].bool()]
        second["teacher_topk_tok"] = second["gen_ids"][:, None]
        second["teacher_topk_prob"] = torch.ones(len(second["gen_ids"]), 1)
        features.append(second)
    batch = collator(features)
    assert batch["input_ids"].shape[1] < teacher["input_ids"].shape[1]
    for row, feature in enumerate(features):
        mask = batch["loss_mask"][row].bool()
        assert torch.equal(batch["input_ids"][row, mask], feature["gen_ids"])
        assert torch.equal(batch["teacher_topk_tok"][row, mask, 0], feature["gen_ids"])
        assert not batch["teacher_topk_prob"][row, ~mask].any()
    assert torch.equal(
        batch["response_positions"], batch["loss_mask"].any(dim=0).nonzero().flatten()
    )
    changed = dataset[0]
    changed["gen_ids"][0] = tokenizer.unk_token_id
    with pytest.raises(ValueError, match="response tokens differ"):
        collator([changed])


def test_video_fps_is_honored_and_does_not_leak(tmp_path, processors):
    """Both policy collection and draft preprocessing must sample the requested frames."""
    teacher = VisionLanguageDataCollator(
        str(tmp_path / "teacher"), train_len=None, answer_only_loss=True
    )
    draft = MultimodalSparsePolicyCollator(str(tmp_path / "draft"), processors, 256)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": torch.zeros(96, 32, 32, 3, dtype=torch.uint8), "fps": 4}
            ],
        },
        {"role": "assistant", "content": "red car"},
    ]
    for collator in (teacher, draft):
        result = process_multimodal_messages(collator, messages)
        assert result["video_grid_thw"][0, 0] == 8
        assert collator.processor.video_processor.fps == 2
        without_fps = copy.deepcopy(messages)
        del without_fps[0]["content"][0]["fps"]
        result = process_multimodal_messages(collator, without_fps)
        assert result["video_grid_thw"][0, 0] == 4
        invalid = copy.deepcopy(messages)
        invalid[0]["content"][0]["fps"] = 0
        with pytest.raises(ValueError, match="FPS"):
            process_multimodal_messages(collator, invalid)
        assert collator.processor.video_processor.fps == 2
