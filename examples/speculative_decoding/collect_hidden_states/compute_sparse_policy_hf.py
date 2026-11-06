# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dump a base model's top-k deployment policy for external draft training.

The counterpart to ``compute_hidden_states_hf.py``: instead of a hidden state per
token it stores the base's truncated next-token distribution, which is what the
TVD objectives actually consume. The dump is far smaller and needs no base
lm_head at training time; feed it to ``main.py`` via ``data.sparse_data_path``.

Output is gzipped JSONL, one conversation per line, matching the schema
``modelopt.torch.speculative.external.sparse_data`` reads.
"""

import argparse
import gzip
import json
import os
import sys
from itertools import islice
from pathlib import Path

import torch
from common import (
    add_answer_only_loss_args,
    load_conversations,
    load_tokenizer,
    tokenize_with_loss_mask,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from modelopt.torch.speculative.external.multimodal import (
    MultimodalSparsePolicyCollator,
    batch_multimodal_inputs,
    normalize_messages,
    process_multimodal_messages,
)
from modelopt.torch.speculative.plugins.hf_external import HFExternalDraftModel
from modelopt.torch.speculative.utils import load_vlm_or_llm
from modelopt.torch.utils.plugins.transformers_dataset import VisionLanguageDataCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump a base model's top-k deployment policy over conversation responses."
    )
    parser.add_argument("--model", type=str, required=True, help="Base (target) model.")
    parser.add_argument(
        "--input-data", type=Path, required=True, help=".jsonl file or directory of .jsonl files."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Conversations per teacher forward per GPU; batching requires --multimodal.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help=(
            "Skip conversations longer than this. Keep it <= training.training_seq_len: the "
            "loader right-truncates to that length, so a record whose context alone exceeds "
            "it carries no scored position."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Width of the stored policy. Must match external.external_top_k at training time.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus applied after top-k, matching external.external_top_p.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2000,
        help="Records per output shard. One record per supervised span, so a multi-turn "
        "conversation contributes one per assistant turn.",
    )
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=int(os.environ.get("RANK", 0)),
        help="Input partition rank; defaults to torchrun's RANK, otherwise 0.",
    )
    parser.add_argument(
        "--dp-world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", 1)),
        help="Number of input partitions; defaults to torchrun's WORLD_SIZE, otherwise 1.",
    )
    # compute_hidden_states_hf.py spells this --trust_remote_code; accept both so the
    # dumpers can be driven from one script.
    parser.add_argument(
        "--trust-remote-code", "--trust_remote_code", action="store_true", dest="trust_remote_code"
    )
    parser.add_argument("--debug-max-num-conversations", type=int, default=None)
    parser.add_argument(
        "--total-conversations",
        type=int,
        help="Total input records before sharding; enables progress percentage and ETA.",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Score single-turn Cosmos3 Super image/video examples.",
    )
    parser.add_argument(
        "--draft-processor",
        help="Qwen3.5 processor used to check draft lengths and response alignment.",
    )
    add_answer_only_loss_args(parser)
    return parser.parse_args()


def deployment_policy(logits, top_k: int, top_p: float):
    """``(ids, log_probs)`` of width ``top_k`` under the filter ``tvd_deploy`` applies to the draft.

    Reuses that filter so the two cannot drift. Slots the nucleus drops hold ``-inf``,
    which ``sparse_data`` decodes to zero probability.
    """
    probs = HFExternalDraftModel._deployment_probs(logits, top_k, top_p)
    ids = logits.float().topk(min(top_k, logits.shape[-1]), dim=-1).indices
    return ids, probs.gather(-1, ids).log()


def scored_spans(loss_mask):
    """Contiguous runs of supervised positions, as half-open ``(start, end)`` pairs.

    A multi-turn conversation masks each assistant turn separately, so the supervised
    positions are disjoint; treating everything after the first as the response would
    score the later user turns too.
    """
    spans, run_start, prev = [], None, None
    for i in loss_mask.squeeze(0).nonzero().flatten().tolist():
        if run_start is None:
            run_start = i
        elif i != prev + 1:
            spans.append((run_start, prev + 1))
            run_start = i
        prev = i
    if run_start is not None:
        spans.append((run_start, prev + 1))
    # Position 0 has no preceding context, so it can carry no policy. Without
    # --answer-only-loss the mask covers the whole sequence and would start at 0.
    return [(max(a, 1), b) for a, b in spans if b > max(a, 1)]


def policy_record(
    conversation_id,
    ids,
    first: int,
    end: int,
    logits,
    top_k: int,
    top_p: float,
    source=None,
    response_only=False,
):
    """Build one shard record for a single supervised span.

    Args:
        first: index of the span's first supervised token (>= 1); everything before it is
            context.
        end: one past the span's last supervised token.
        logits: the base's logits over the whole sequence.
    """
    # Position t of the stored policy is the distribution token ids[t] was drawn from,
    # conditioned on ids[:t] -- that is the model's logits at t - 1. Dropping this shift
    # still yields a falling loss curve, against the wrong target.
    predictors = logits if response_only else logits[first - 1 : end - 1]
    tok, lp = deployment_policy(predictors, top_k, top_p)
    record = {
        "id": str(conversation_id),
        "prompt_ids": ids[:first].tolist(),
        "gen_ids": ids[first:end].tolist(),
        "topk_tok": tok.cpu().tolist(),
        "topk_lp": lp.float().cpu().tolist(),
    }
    if source is not None:
        record["source"] = source
    return record


def multimodal_policy_records(model, samples, pad_token_id, top_k, top_p):
    """Score padded conversations at their own response predictor positions."""
    batch = batch_multimodal_inputs([sample[2] for sample in samples], pad_token_id)
    positions = [torch.arange(spans[0][0] - 1, spans[0][1] - 1) for _, _, _, spans in samples]
    union = torch.cat(positions).unique(sorted=True)
    batch.update(use_cache=False, logits_to_keep=union)
    with torch.inference_mode():
        logits = model(
            **{
                key: value.to(model.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
        ).logits
    records = []
    for row, (entry, messages, inputs, spans) in enumerate(samples):
        first, end = spans[0]
        predictors = logits[row].index_select(
            0, torch.searchsorted(union, positions[row]).to(logits.device)
        )
        record = policy_record(
            entry.get("conversation_id") or entry.get("uuid") or entry["id"],
            inputs["input_ids"].squeeze(0),
            first,
            end,
            predictors,
            top_k,
            top_p,
            entry.get("source") or entry.get("dataset"),
            response_only=True,
        )
        record["messages"] = messages
        records.append(record)
    return records


def multimodal_conversations(path, rank, world_size):
    """Stream heterogeneous message content without converting it to an Arrow schema."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid data-parallel rank/world size.")
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"No JSONL input files found at {path}.")
    index = 0
    for source in paths:
        with source.open() as stream:
            for line in stream:
                if index % world_size == rank:
                    yield json.loads(line)
                index += 1


def main(args: argparse.Namespace) -> None:
    if args.dp_world_size < 1 or not 0 <= args.dp_rank < args.dp_world_size:
        raise ValueError("Invalid data-parallel rank/world size.")
    if args.total_conversations is not None and args.total_conversations < 0:
        raise ValueError("--total-conversations must be nonnegative.")
    if args.batch_size < 1 or (args.batch_size > 1 and not args.multimodal):
        raise ValueError("--batch-size must be positive; values above 1 require --multimodal.")
    device_map = "auto"
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device_map = {"": f"cuda:{local_rank}"}
    if args.multimodal and (not args.draft_processor or not args.answer_only_loss):
        raise ValueError("--multimodal requires --draft-processor and --answer-only-loss.")
    if args.top_k < 1:
        raise ValueError(
            f"--top-k must be at least 1, got {args.top_k}: the dump stores a "
            "fixed-width policy and a zero-width row carries no teacher."
        )
    if args.multimodal:
        dataset = multimodal_conversations(args.input_data, args.dp_rank, args.dp_world_size)
        if args.debug_max_num_conversations is not None:
            dataset = islice(dataset, args.debug_max_num_conversations)
    else:
        dataset = load_conversations(args.input_data, args.dp_rank, args.dp_world_size)
    if not args.multimodal and args.debug_max_num_conversations is not None:
        # After DP sharding this rank may hold fewer rows than the debug cap, and
        # select() raises on an out-of-range index rather than clamping.
        dataset = dataset.select(range(min(args.debug_max_num_conversations, len(dataset))))

    loader = load_vlm_or_llm if args.multimodal else AutoModelForCausalLM.from_pretrained
    model = loader(
        args.model, dtype="auto", device_map=device_map, trust_remote_code=args.trust_remote_code
    )
    print(f"DP#{args.dp_rank}/{args.dp_world_size}: teacher device {model.device}", flush=True)
    model.eval()
    if args.multimodal:
        if model.config.model_type != "qwen3_vl" or not args.answer_only_loss:
            raise ValueError(
                "Multimodal policy collection requires Cosmos3 Super and --answer-only-loss."
            )
        collator = VisionLanguageDataCollator(
            processor=args.model,
            train_len=None,
            answer_only_loss=True,
            chat_template=args.chat_template.read_text() if args.chat_template else None,
        )
        if collator.max_prompt_tokens is not None or collator.max_assistant_tokens is not None:
            raise ValueError("Text truncation is incompatible with cached response policies.")
        draft_collator = MultimodalSparsePolicyCollator(
            args.draft_processor, collator.tokenizer, args.max_seq_len
        )
    else:
        tokenizer = load_tokenizer(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A second, shorter export overwrites only the shard names it reaches, so the extra
    # shards from the first run stay visible to the trainer's glob and it reads records
    # from both. Refuse rather than silently mix two dumps.
    existing = sorted(args.output_dir.glob(f"policy-{args.dp_rank:03d}-*.jsonl.gz"))
    if existing:
        raise FileExistsError(
            f"{args.output_dir} already holds {len(existing)} shard(s) for DP rank "
            f"{args.dp_rank}, starting with {existing[0].name}. Remove them or pass a "
            "fresh --output-dir; a shorter run here would leave the earlier dump's "
            "extra shards behind and training would read both."
        )
    num_skipped = num_invalid = num_success = 0
    shard_idx = 0
    written_in_shard = 0
    shard_file = None

    def open_shard(idx: int):
        # Write-then-rename: a partial shard left by an interrupted run would otherwise
        # be picked up by the trainer's glob and decoded as damaged input.
        tmp = args.output_dir / f"policy-{args.dp_rank:03d}-{idx:05d}.jsonl.gz.tmp"
        # Exclusive create: the existing-shard check above is a point-in-time test, so
        # two exports for the same rank can both pass it and then write the same file.
        # "xt" makes the second one fail here instead of interleaving records.
        try:
            return gzip.open(tmp, "xt"), tmp
        except FileExistsError:
            raise FileExistsError(
                f"{tmp} already exists: another export is writing DP rank "
                f"{args.dp_rank} into {args.output_dir}, or a previous one was killed. "
                "Remove the .tmp files or use a fresh --output-dir."
            ) from None

    completed = False
    total = (
        len(range(args.dp_rank, args.total_conversations, args.dp_world_size))
        if args.total_conversations is not None
        else None
    )
    if args.debug_max_num_conversations is not None:
        total = (
            min(total, args.debug_max_num_conversations)
            if total is not None
            else args.debug_max_num_conversations
        )
    progress = tqdm(total=total, desc=f"DP#{args.dp_rank} dumping policy")
    pending = []

    def write_record(record):
        nonlocal shard_file, shard_tmp, shard_idx, written_in_shard
        shard_file.write(json.dumps(record) + "\n")
        written_in_shard += 1
        if written_in_shard >= args.shard_size:
            shard_file.close()
            os.replace(shard_tmp, shard_tmp.with_suffix(""))
            shard_idx += 1
            written_in_shard = 0
            shard_file, shard_tmp = open_shard(shard_idx)

    def flush_pending():
        nonlocal num_success
        if not pending:
            return
        records = multimodal_policy_records(
            model, pending, collator.tokenizer.pad_token_id, args.top_k, args.top_p
        )
        for record in records:
            write_record(record)
        num_success += len(pending)
        pending.clear()
        progress.set_postfix(
            written=num_success, skipped=num_skipped, invalid=num_invalid, refresh=False
        )

    try:
        shard_file, shard_tmp = open_shard(shard_idx)
        for entry in dataset:
            progress.update(1)
            conversation_id = entry.get("conversation_id") or entry.get("uuid") or entry.get("id")
            if conversation_id is None:
                raise ValueError("Each conversation needs a conversation_id, uuid or id.")
            conversations = entry.get("messages") or entry.get("conversations")
            if not conversations or not isinstance(conversations, list):
                num_invalid += 1
                progress.set_postfix(
                    written=num_success, skipped=num_skipped, invalid=num_invalid, refresh=False
                )
                continue

            if args.multimodal:
                conversations = normalize_messages(entry)
                model_inputs = process_multimodal_messages(collator, conversations)
                loss_mask = model_inputs.pop("assistant_masks").squeeze(0)
                input_ids = model_inputs["input_ids"]
            else:
                input_ids, loss_mask = tokenize_with_loss_mask(
                    tokenizer, conversations, args.answer_only_loss
                )
                model_inputs = {"input_ids": input_ids}
            if input_ids.shape[1] <= 10 or input_ids.shape[1] > args.max_seq_len:
                num_skipped += 1
                progress.set_postfix(
                    written=num_success, skipped=num_skipped, invalid=num_invalid, refresh=False
                )
                continue
            if args.multimodal:
                draft_inputs = draft_collator.process_messages(conversations)
                if draft_inputs["input_ids"].shape[1] > args.max_seq_len:
                    num_skipped += 1
                    progress.set_postfix(
                        written=num_success, skipped=num_skipped, invalid=num_invalid, refresh=False
                    )
                    continue
                teacher_response = input_ids[loss_mask.unsqueeze(0).bool()]
                draft_response = draft_inputs["input_ids"][draft_inputs["assistant_masks"].bool()]
                if not torch.equal(teacher_response, draft_response):
                    raise ValueError(f"Teacher/draft response tokens differ for {conversation_id}.")

            # The supervised span is the response; everything before it is context.
            # One record per supervised span: a multi-turn conversation masks each
            # assistant turn separately, and each is scored against the context before it.
            spans = scored_spans(loss_mask)
            if not spans:
                num_invalid += 1
                progress.set_postfix(
                    written=num_success, skipped=num_skipped, invalid=num_invalid, refresh=False
                )
                continue
            if args.multimodal:
                if len(spans) != 1:
                    raise ValueError("Expected exactly one assistant response span.")
                pending.append((entry, conversations, model_inputs, spans))
                if len(pending) == args.batch_size:
                    flush_pending()
                continue
            ids = input_ids.squeeze(0)

            with torch.inference_mode():
                logits = model(
                    **{
                        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                        for k, v in model_inputs.items()
                    }
                ).logits.squeeze(0)
            for turn, (span_start, span_end) in enumerate(spans):
                record = policy_record(
                    f"{conversation_id}-{turn}" if len(spans) > 1 else conversation_id,
                    ids,
                    span_start,
                    span_end,
                    logits,
                    args.top_k,
                    args.top_p,
                    entry.get("source") or entry.get("dataset"),
                    response_only=args.multimodal,
                )
                write_record(record)
            del logits
            num_success += 1
            progress.set_postfix(
                written=num_success, skipped=num_skipped, invalid=num_invalid, refresh=False
            )
        flush_pending()
        completed = True
    finally:
        progress.set_postfix(
            written=num_success, skipped=num_skipped, invalid=num_invalid, refresh=False
        )
        progress.close()
        if shard_file is not None:
            shard_file.close()
            # Publish the last shard only on a clean pass. Renaming it after a failure
            # hands the trainer a truncated export that looks complete.
            if written_in_shard and completed:
                os.replace(shard_tmp, shard_tmp.with_suffix(""))
            else:
                Path(shard_tmp).unlink(missing_ok=True)
        if not completed:
            done = sorted(args.output_dir.glob(f"policy-{args.dp_rank:03d}-*.jsonl.gz"))
            print(
                f"Export failed after {num_success} conversations. {len(done)} completed "
                f"shard(s) remain in {args.output_dir} and are a partial dump; remove "
                "them before training or re-exporting.",
                file=sys.stderr,
            )

    if num_skipped:
        print(f"Skipped {num_skipped} conversations due to length constraints.")
    if num_invalid:
        print(f"Skipped {num_invalid} conversations without a usable response span.")
    print(f"Wrote {num_success} conversations to {args.output_dir}")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
