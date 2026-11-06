# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU tests for the external (standalone) draft plugin.

These tests require a CUDA GPU. CPU-only tests are in tests/unit/.

The base lm_head is attached with ``object.__setattr__`` so it stays off the draft's
module tree and out of its optimiser and gradients. The cost of that is that
``model.cuda()`` does not move it, which no CPU test can observe: every tensor is on
the same device there. That is the class of bug these tests exist for.
"""

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama, get_tiny_qwen3_5_vl_offline

import modelopt.torch.speculative as mtsp

SEQ_LEN = 16


def _external_draft(**overrides):
    model = get_tiny_llama()
    mtsp.convert(
        model,
        [("external", {"external_loss": "tvd_deploy", **overrides})],
    )
    return model


class TestExternalDraftDevicePlacement:
    """The base lm_head lives off the module tree, so it needs its own device handling."""

    def test_teacher_logits_with_base_head_left_on_cpu(self):
        """Attaching a CPU head then moving the draft to GPU must still train."""
        draft = _external_draft()
        base = get_tiny_llama()
        # Deliberately attach before .cuda(), the order main.py uses.
        draft.attach_base_lm_head(
            base.get_output_embeddings(), getattr(getattr(base, "model", None), "norm", None)
        )
        draft = draft.cuda()

        hidden = torch.randn(
            1, SEQ_LEN, draft.config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        logits = draft._teacher_logits(hidden, {})
        assert logits.device.type == "cuda"
        assert logits.shape == (1, SEQ_LEN, draft.config.vocab_size)

    def test_base_head_stays_out_of_the_draft_parameters(self):
        """Gradient isolation: the teacher must not appear in the draft's optimiser."""
        draft = _external_draft()
        base = get_tiny_llama()
        draft.attach_base_lm_head(base.get_output_embeddings())
        draft = draft.cuda()

        param_ids = {id(p) for p in draft.parameters()}
        assert all(id(p) not in param_ids for p in base.get_output_embeddings().parameters())
        assert not any("_base_lm_head" in k for k in draft.state_dict())


def test_dense_path_rejects_a_missing_teacher_head():
    """Without the head the dense objective cannot be formed; fail loudly, not silently."""
    draft = _external_draft().cuda()
    with pytest.raises(RuntimeError, match="Base lm_head is not attached"):
        draft._teacher_logits(
            torch.randn(1, SEQ_LEN, draft.config.hidden_size, device="cuda", dtype=torch.bfloat16),
            {},
        )


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_qwen35_visual_encoder_frozen_merger_updates(video, batch_size):
    """The real multimodal forward must train the merger and LM while freezing the encoder."""
    draft = get_tiny_qwen3_5_vl_offline().float().cuda()
    mtsp.convert(draft, [("external", {"external_freeze_vision_encoder": True})])
    draft.train()
    visual = draft.model.visual
    assert not visual.training and visual.merger.training
    encoder = [p for name, p in visual.named_parameters() if not name.startswith("merger.")]
    before = [p.detach().clone() for p in visual.merger.parameters()]
    optimizer = torch.optim.SGD([p for p in draft.parameters() if p.requires_grad], lr=0.1)
    ids = torch.tensor([[8, 3, 2 if video else 1, 4, 9, 10]], device="cuda").repeat(batch_size, 1)
    mask = torch.tensor([[0, 0, 0, 0, 1, 1]], device="cuda").repeat(batch_size, 1)
    if batch_size == 2:
        mask[1] = torch.tensor([0, 0, 0, 1, 1, 0], device="cuda")
    positions = mask.any(dim=0).nonzero().flatten()
    types = torch.zeros_like(ids)
    types[:, 2] = 2 if video else 1
    topk = torch.tensor([11, 12, 13, 14], device="cuda").expand(batch_size, 6, 4)
    prob = torch.full((batch_size, 6, 4), 0.25, device="cuda")
    media = {
        "pixel_values_videos" if video else "pixel_values": torch.randn(
            4 * batch_size, 1536, device="cuda"
        ),
        "video_grid_thw" if video else "image_grid_thw": torch.tensor(
            [[1, 2, 2]], device="cuda"
        ).repeat(batch_size, 1),
    }
    result = draft(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        mm_token_type_ids=types,
        teacher_topk_tok=topk,
        teacher_topk_prob=prob,
        loss_mask=mask,
        response_positions=positions,
        use_cache=False,
        **media,
    )
    assert result.logits.shape[1] == len(positions)
    with torch.no_grad():
        full = draft.original_cls.forward(
            draft,
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=types,
            use_cache=False,
            **media,
        )
        expected, _ = draft.compute_sparse_loss(full.logits, topk, prob, mask)
    assert torch.allclose(result.loss, expected, atol=1e-5)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in encoder)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in visual.merger.parameters())
    assert any(p.grad is not None for p in draft.model.language_model.parameters())
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, visual.merger.parameters()))
