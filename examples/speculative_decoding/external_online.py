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

"""Online sparse-policy distillation for Cosmos3 Super and Qwen3.5."""

import json
import os
from array import array
from pathlib import Path

import torch
from eagle_utils import EagleTrainerWithAccLog
from transformers import TrainerCallback

from modelopt.torch.speculative.external.multimodal import (
    MultimodalSparsePolicyCollator,
    batch_multimodal_inputs,
    normalize_messages,
    process_multimodal_messages,
)
from modelopt.torch.speculative.plugins.hf_external import HFExternalDraftModel
from modelopt.torch.utils.plugins.transformers_dataset import VisionLanguageDataCollator

__all__ = ["OnlineConversationDataset", "OnlineExternalDraftTrainer", "OnlineMultimodalCollator"]


class OnlineConversationDataset(torch.utils.data.Dataset):
    """Index raw JSONL rows without imposing an Arrow schema on mixed media content."""

    def __init__(self, path, limit=-1):
        self.path = Path(path)
        self.offsets = array("Q")
        before = self.path.stat()
        with self.path.open("rb") as stream:
            while limit < 0 or len(self.offsets) < limit:
                offset = stream.tell()
                if not stream.readline():
                    break
                self.offsets.append(offset)
        after = self.path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Online input changed while indexing it.")
        if not self.offsets:
            raise ValueError("Online input is empty.")
        self.fingerprint = {
            "path": str(self.path.resolve()),
            "size": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "limit": limit,
        }

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            if (stat.st_size, stat.st_mtime_ns) != (
                self.fingerprint["size"],
                self.fingerprint["mtime_ns"],
            ):
                raise ValueError("Online input changed after indexing it.")
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())


class OnlineMultimodalCollator:
    """Prepare separate visual sequences and verify their assistant tokens match."""

    def __init__(
        self, teacher_processor, draft_processor, tokenizer, max_length, chat_template=None
    ):
        self.teacher = VisionLanguageDataCollator(
            teacher_processor, train_len=None, answer_only_loss=True, chat_template=chat_template
        )
        self.draft = MultimodalSparsePolicyCollator(
            draft_processor, tokenizer, max_length, chat_template=chat_template
        )
        if (
            self.teacher.max_prompt_tokens is not None
            or self.teacher.max_assistant_tokens is not None
        ):
            raise ValueError("Text truncation is incompatible with response alignment.")
        self.processor = self.draft.processor
        self.max_length = max_length

    def __call__(self, records):
        teacher, draft = [], []
        for record in records:
            messages = normalize_messages(record)
            target = process_multimodal_messages(self.teacher, messages)
            sample = self.draft.process_messages(messages)
            if max(target["input_ids"].shape[1], sample["input_ids"].shape[1]) > self.max_length:
                continue
            target_response = target["input_ids"][target["assistant_masks"].bool()]
            draft_response = sample["input_ids"][sample["assistant_masks"].bool()]
            if not target_response.numel() or not torch.equal(target_response, draft_response):
                raise ValueError("Online teacher/draft response tokens differ or are empty.")
            for inputs in (target, sample):
                mask = inputs["assistant_masks"].bool().squeeze(0)
                positions = mask.nonzero().flatten()
                if mask[0] or not torch.equal(
                    positions, torch.arange(positions[0], positions[-1] + 1)
                ):
                    raise ValueError(
                        "Expected one contiguous assistant response with a preceding prompt."
                    )
            teacher.append(target)
            draft.append(sample)
        skipped = len(records) - len(draft)
        if not draft:
            # Every DDP rank must run backward, including ranks whose whole batch is too long.
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello"},
            ]
            draft = [self.draft.process_messages(messages)]
        batch = batch_multimodal_inputs(draft, self.draft.tokenizer.pad_token_id)
        mask = batch.pop("assistant_masks").bool()
        batch["response_positions"] = mask.any(dim=0).nonzero().flatten()
        batch["loss_mask"] = mask.float() if teacher else torch.zeros_like(mask, dtype=torch.float)
        batch["teacher_inputs"] = (
            batch_multimodal_inputs(teacher, self.teacher.tokenizer.pad_token_id)
            if teacher
            else None
        )
        batch["online_skipped"] = skipped
        batch["online_valid"] = len(teacher)
        return batch


class _OnlineBatchCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        # Trainer restores the old save interval from checkpoint state after reading args.
        state.compute_steps(args, state.max_steps)

    def on_step_begin(self, args, state, control, **kwargs):
        self.trainer.online_valid = 0

    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        valid = torch.tensor(self.trainer.online_valid, device=args.device)
        # All ranks enter this collective, regardless of local length filtering.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(valid)
        if valid.item() == 0:
            # AdamW's momentum and weight decay must not update a wholly filtered batch.
            for param in model.parameters():
                param.grad = None


class OnlineExternalDraftTrainer(EagleTrainerWithAccLog):
    """Score a frozen teacher before each draft forward, retaining only sparse policies."""

    def __init__(self, *args, teacher, top_k, top_p, **kwargs):
        if top_k < 1 or not 0 <= top_p <= 1:
            raise ValueError("Online policies require top_k >= 1 and top_p between 0 and 1.")
        self.teacher = teacher.requires_grad_(False).eval()
        self.top_k, self.top_p = top_k, top_p
        self.online_skipped = 0
        self.online_valid = 0
        super().__init__(*args, **kwargs)
        self.teacher.to(self.args.device)
        self.add_callback(_OnlineBatchCallback(self))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        target = inputs.pop("teacher_inputs")
        self.online_skipped += inputs.pop("online_skipped")
        self.online_valid += inputs.pop("online_valid")
        ids, mask = inputs["input_ids"], inputs["loss_mask"].bool()
        shape = (*ids.shape, self.top_k)
        tok = torch.zeros(shape, dtype=torch.long, device=ids.device)
        prob = torch.zeros(shape, dtype=torch.float, device=ids.device)
        if target is not None:
            target = dict(target)
            teacher_mask = target.pop("assistant_masks").bool()
            union = teacher_mask.any(dim=0).nonzero().flatten()
            with torch.no_grad():
                logits = self.teacher(**target, use_cache=False, logits_to_keep=union - 1).logits
                for row in range(ids.shape[0]):
                    positions = teacher_mask[row].nonzero().flatten()
                    predictors = logits[row].index_select(0, torch.searchsorted(union, positions))
                    policy = HFExternalDraftModel._deployment_probs(
                        predictors, self.top_k, self.top_p
                    )
                    selected = (
                        predictors.float()
                        .topk(min(self.top_k, predictors.shape[-1]), dim=-1)
                        .indices
                    )
                    if selected.shape[-1] != self.top_k:
                        raise ValueError("Online top_k exceeds the teacher vocabulary.")
                    tok[row, mask[row]] = selected
                    prob[row, mask[row]] = policy.gather(-1, selected)
                del logits, predictors, policy
        inputs.update(teacher_topk_tok=tok, teacher_topk_prob=prob)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def log(self, logs, *args, **kwargs):
        logs = {**logs, "online_skipped_local": self.online_skipped}
        return super().log(logs, *args, **kwargs)
