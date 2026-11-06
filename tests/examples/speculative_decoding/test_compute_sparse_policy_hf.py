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

"""Exercise automatic torchrun partitioning through the real sparse-policy CLI."""

import gzip
import hashlib
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_qwen3_5_vl_offline, get_tiny_tokenizer
from PIL import Image
from safetensors.torch import load_file
from transformers import (
    Qwen2VLImageProcessorFast,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
    TrainingArguments,
)

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.external.multimodal import (
    MultimodalSparsePolicyCollator,
    normalize_messages,
    process_multimodal_messages,
    retarget_visual_tokens,
)


@pytest.fixture
def policy_data(tmp_path):
    """Save a tiny teacher/processor and heterogeneous-length image conversations."""
    tokenizer = get_tiny_tokenizer()
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|image_pad|>",
                "<|video_pad|>",
            ]
        }
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' }}"
        "{% if m['content'] is string %}{{ m['content'] }}{% else %}"
        "{% for c in m['content'] %}{% if c['type'] == 'image' %}"
        "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
        "{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}"
        "{{ '<|im_end|>\\n' }}{% endfor %}"
    )
    model_dir = tmp_path / "teacher"
    model = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(
            text_config={
                "vocab_size": len(tokenizer),
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 32,
                "deepstack_visual_indexes": [],
            },
            image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
            vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        )
    )
    model.save_pretrained(model_dir)
    Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorFast(
            patch_size=16, temporal_patch_size=2, merge_size=2, min_pixels=1024, max_pixels=1024
        ),
        video_processor=Qwen3VLVideoProcessor(),
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
    ).save_pretrained(model_dir)
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "red").save(image)
    records = [
        {
            "id": f"sample-{index}",
            "messages": [
                {"role": "user", "content": "Describe " * (index + 1) + "the image <image>"},
                {"role": "assistant", "content": "A red car."},
            ],
            "images": [str(image)],
        }
        for index in range(6)
    ]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "policies"
    root = Path(__file__).resolve().parents[3]
    return model_dir, source, output, root


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
@pytest.mark.parametrize("launch_mode", ["cli", "slurm_worker"])
def test_torchrun_partitions_multimodal_policies(tmp_path, policy_data, launch_mode):
    """Two workers produce disjoint shards, including batched scoring and resume."""
    model_dir, source, output, root = policy_data
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "examples/speculative_decoding/collect_hidden_states/compute_sparse_policy_hf.py",
        "--model",
        str(model_dir),
        "--input-data",
        str(source),
        "--output-dir",
        str(output),
        "--multimodal",
        "--draft-processor",
        str(model_dir),
        "--answer-only-loss",
        "--max-seq-len",
        "256",
        "--top-k",
        "4",
        "--debug-max-num-conversations",
        "3",
        "--total-conversations",
        "6",
    ]
    reference_command = command.copy()
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    if launch_mode == "slurm_worker":
        dataset_root = output / "cosmos3"
        dataset_root.mkdir(parents=True)
        stat = source.stat()
        (dataset_root / "run.json").write_text(
            json.dumps(
                {
                    "input": str(source),
                    "input_size": stat.st_size,
                    "input_mtime_ns": stat.st_mtime_ns,
                    "model": str(model_dir),
                    "versions": {},
                    "visual_env": {},
                    "model_config_sha256": hashlib.sha256(
                        (model_dir / "config.json").read_bytes()
                    ).hexdigest(),
                }
            )
        )
        interrupted = dataset_root / "staging" / "part-0000"
        interrupted.mkdir(parents=True)
        (interrupted / "partial.txt").write_text("interrupted attempt")
        env.update(
            {
                "SLURM_JOB_ID": "smoke",
                "SLURM_ARRAY_TASK_ID": "0",
                "REPO_ROOT": str(root),
                "POLICY_DATASET": "cosmos3",
                "INPUT_DATA": str(source),
                "OUTPUT_ROOT": str(output),
                "PARTITION_GROUPS": "1",
                "TOTAL_CONVERSATIONS": "6",
                "BATCH_SIZE": "2",
                "NUM_GPUS": "2",
                "MODEL_PATH": str(model_dir),
                "DRAFT_PROCESSOR": str(model_dir),
                "MAX_SEQ_LEN": "256",
                "TOP_K": "4",
                "TOP_P": "0.95",
            }
        )
        command = [
            *command[:5],
            "--no-python",
            "bash",
            "examples/speculative_decoding/scripts/generate_external_policies.sbatch",
            "--worker",
        ]
    result = subprocess.run(
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    seen = []
    for rank in range(2):
        shards = list(output.rglob(f"policy-{rank:03d}-*.jsonl.gz"))
        assert len(shards) == 1
        with gzip.open(shards[0], "rt") as stream:
            rows = [json.loads(line) for line in stream]
            assert [row["id"] for row in rows] == [f"sample-{i}" for i in range(rank, 6, 2)]
            for row in rows:
                seen.append(row["id"])
                assert row["gen_ids"] and len(row["topk_tok"]) == len(row["gen_ids"])
    assert sorted(seen) == [f"sample-{i}" for i in range(6)]

    if launch_mode == "slurm_worker":
        assert len(list(dataset_root.glob("policy/part-*/.complete"))) == 2
        assert len(list(dataset_root.glob("incomplete/*/work/partial.txt"))) == 1
        before = {p: p.stat().st_mtime_ns for p in dataset_root.rglob("policy-*.jsonl.gz")}
        resumed = subprocess.run(
            command, cwd=root, env=env, capture_output=True, text=True, timeout=60
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert before == {p: p.stat().st_mtime_ns for p in dataset_root.rglob("policy-*.jsonl.gz")}

        baseline_output = tmp_path / "baseline"
        reference_command[reference_command.index("--output-dir") + 1] = str(baseline_output)
        baseline = subprocess.run(
            reference_command, cwd=root, env=env, capture_output=True, text=True, timeout=120
        )
        assert baseline.returncode == 0, baseline.stdout + baseline.stderr
        for rank in range(2):
            with gzip.open(
                next(baseline_output.glob(f"policy-{rank:03d}-*.jsonl.gz")), "rt"
            ) as stream:
                expected = [json.loads(line) for line in stream]
            with gzip.open(
                next(dataset_root.rglob(f"policy-{rank:03d}-*.jsonl.gz")), "rt"
            ) as stream:
                actual = [json.loads(line) for line in stream]
            for left, right in zip(expected, actual, strict=True):
                for key in ("id", "prompt_ids", "gen_ids", "messages", "topk_tok"):
                    assert left[key] == right[key]
                torch.testing.assert_close(
                    torch.tensor(left["topk_lp"]).exp(),
                    torch.tensor(right["topk_lp"]).exp(),
                    rtol=1e-4,
                    atol=1e-6,
                )


def run_online_training(command, root, env, launch_mode):
    """Exercise either one local agent or two one-GPU agents sharing rendezvous."""
    if launch_mode == "single_agent":
        return subprocess.run(
            command, cwd=root, env=env, capture_output=True, text=True, timeout=180
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = []
    logs = []
    try:
        for gpu in range(2):
            worker_env = {
                **env,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "REPO_ROOT": str(root),
                "NUM_NODES": "2",
                "NUM_GPUS": "1",
                "SLURM_JOB_ID": "online-test",
                "RDZV_HOST": "127.0.0.1",
                "RDZV_PORT": str(port),
            }
            processes.append(
                subprocess.Popen(
                    [
                        "bash",
                        "examples/speculative_decoding/scripts/train_external_online_worker.sh",
                        *command[5:],
                    ],
                    cwd=root,
                    env=worker_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 180
        for process in processes:
            output, _ = process.communicate(timeout=max(1, deadline - time.monotonic()))
            logs.append(output)
        return subprocess.CompletedProcess(
            command, max(abs(p.returncode) for p in processes), "\n".join(logs), ""
        )
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
@pytest.mark.parametrize("launch_mode", ["single_agent", "two_agents"])
def test_online_external_training_and_resume(tmp_path, policy_data, monkeypatch, launch_mode):
    """Real DDP training must isolate the teacher and resume a standalone draft checkpoint."""
    model_dir, source, _, root = policy_data
    processor = Qwen3VLProcessor.from_pretrained(model_dir)
    draft = get_tiny_qwen3_5_vl_offline(vocab_size=len(processor.tokenizer))
    draft_dir = tmp_path / "draft"
    draft.save_pretrained(draft_dir)
    processor.save_pretrained(draft_dir)
    # Different visual lengths must preserve response alignment online and offline.
    processor.image_processor = Qwen2VLImageProcessorFast(
        patch_size=16, temporal_patch_size=2, merge_size=2, min_pixels=4096, max_pixels=4096
    )
    processor.save_pretrained(model_dir)
    monkeypatch.syspath_prepend(str(root / "examples/speculative_decoding"))
    monkeypatch.syspath_prepend(str(root / "examples/speculative_decoding/collect_hidden_states"))
    online = importlib.import_module("external_online")
    producer = importlib.import_module("compute_sparse_policy_hf")
    monkeypatch.delitem(sys.modules, "common", raising=False)
    teacher = Qwen3VLForConditionalGeneration.from_pretrained(model_dir).cuda().eval()
    draft_cls = type(draft)
    retarget_visual_tokens(draft.config, processor.tokenizer)
    mtsp.convert(draft, [("external", {"external_freeze_vision_encoder": True})])
    draft = draft.cuda().eval()
    collator = online.OnlineMultimodalCollator(model_dir, draft_dir, processor.tokenizer, 128)
    trainer = online.OnlineExternalDraftTrainer(
        model=draft,
        teacher=teacher,
        top_k=4,
        top_p=0.95,
        args=TrainingArguments(
            output_dir=str(tmp_path / "parity"), report_to=[], remove_unused_columns=False
        ),
    )
    records = [json.loads(line) for line in source.read_text().splitlines()]
    features = []
    with torch.no_grad():
        for entry in records[:2]:
            messages = normalize_messages(entry)
            inputs = process_multimodal_messages(collator.teacher, messages)
            mask = inputs.pop("assistant_masks")
            first, end = producer.scored_spans(mask)[0]
            logits = teacher(
                **{key: value.cuda() for key, value in inputs.items()}, use_cache=False
            ).logits[0]
            record = producer.policy_record(
                entry["id"], inputs["input_ids"][0], first, end, logits, 4, 0.95
            )
            features.append(
                {
                    "messages": messages,
                    "gen_ids": torch.tensor(record["gen_ids"]),
                    "teacher_topk_tok": torch.tensor(record["topk_tok"]),
                    "teacher_topk_prob": torch.tensor(record["topk_lp"]).exp(),
                }
            )
        expected = MultimodalSparsePolicyCollator(draft_dir, processor.tokenizer, 128)(features)
        expected_loss = draft(**{key: value.cuda() for key, value in expected.items()}).loss
        actual_loss = trainer.compute_loss(draft, trainer._prepare_inputs(collator(records[:2])))
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-6)
    del teacher, trainer
    draft = draft.cpu()
    for record in records[::2]:
        record["messages"][0]["content"] = "Describe " * 300 + "<image>"
    records[1]["messages"][0]["content"] = "Explain <image>flickr</image> in XML."
    records[1].pop("images")
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "online"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "examples/speculative_decoding/main.py",
        "--config",
        "examples/speculative_decoding/recipes/cosmos3_super_qwen35_external_online.yaml",
        f"model.model_name_or_path={model_dir}",
        f"draft_model_name_or_path={draft_dir}",
        f"data.vlm_processor={draft_dir}",
        f"data.data_path={source}",
        f"training.output_dir={output}",
        "training.training_seq_len=128",
        "training.gradient_accumulation_steps=1",
        "training.learning_rate=0.001",
        "training.warmup_steps=0",
        "training.weight_decay=0.1",
        "training.max_steps=2",
        "training.save_steps=1",
        "training.logging_steps=1",
        "training.report_to=[]",
        "training.dataloader_num_workers=0",
        "external.external_top_k=4",
    ]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
    result = run_online_training(command, root, env, launch_mode)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((output / "checkpoint-2" / "trainer_state.json").read_text())
    assert state["global_step"] == 2
    command[command.index("training.max_steps=2")] = "training.max_steps=4"
    command[command.index("training.save_steps=1")] = "training.save_steps=2"
    resumed = run_online_training(command, root, env, launch_mode)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = json.loads((output / "checkpoint-4" / "trainer_state.json").read_text())
    assert state["global_step"] == 4
    assert state["save_steps"] == 2
    assert not (output / "checkpoint-3").exists()
    assert any(row.get("online_skipped_local", 0) > 0 for row in state["log_history"])
    empty_steps = [
        row["step"] for row in state["log_history"] if row.get("loss") == 0 and row["step"] > 1
    ]
    assert empty_steps
    checked_empty_updates = 0
    for step in empty_steps:
        previous_path = output / f"checkpoint-{step - 1}" / "model.safetensors"
        if not previous_path.exists():
            continue
        checked_empty_updates += 1
        previous = load_file(previous_path)
        current = load_file(output / f"checkpoint-{step}" / "model.safetensors")
        for name in current:
            torch.testing.assert_close(current[name], previous[name], rtol=0, atol=0)
    assert checked_empty_updates > 0
    restored = draft_cls.from_pretrained(output)
    assert not any("teacher" in key for key in restored.state_dict())
    for name, param in restored.model.visual.named_parameters():
        if not name.startswith("merger."):
            torch.testing.assert_close(
                param, draft.model.visual.get_parameter(name), rtol=0, atol=0
            )
