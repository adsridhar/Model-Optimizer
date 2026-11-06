#!/usr/bin/env bash
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
#
# Usage: bash submit_external_policy_gen.sh [--dry-run] [cosmos3|llava|both]
set -euo pipefail
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
selection="${1:-cosmos3}"
case "$selection" in cosmos3|llava|both) ;; *) echo "Expected cosmos3, llava, or both" >&2; exit 1 ;; esac
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/../../scratch.adsridhar_other_1/external_draft}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/../models/cosmos3-super-reasoner_vbf16-final}"
DRAFT_PROCESSOR="${DRAFT_PROCESSOR:-Qwen/Qwen3.5-0.8B}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TOP_K="${TOP_K:-20}"
TOP_P="${TOP_P:-0.95}"
[[ -x "$PYTHON_BIN" && -d "$MODEL_PATH" ]] || { echo "Missing repo .venv or teacher checkpoint" >&2; exit 1; }
[[ -z "${VLM_MAX_PROMPT_TOKENS:-}" && -z "${VLM_MAX_ASSISTANT_TOKENS:-}" ]] || {
    echo "Unset VLM_MAX_PROMPT_TOKENS and VLM_MAX_ASSISTANT_TOKENS for cached policies." >&2; exit 1;
}
datasets=("$selection")
if [[ "$selection" == both ]]; then datasets=(cosmos3 llava); fi
prepared_inputs=()
prepared_groups=()
prepared_counts=()
for POLICY_DATASET in "${datasets[@]}"; do
    case "$POLICY_DATASET" in
        cosmos3)
            INPUT_DATA="${COSMOS_DATA:-$REPO_ROOT/../../scratch.adsridhar_other_1/cosmos3_super_dflash_train_full_8192_partial.jsonl}"
            groups_default=128 ;;
        llava)
            INPUT_DATA="${LLAVA_DATA:-$REPO_ROOT/../LlamaFactory/data/llava_instruct_gqa_sft.jsonl}"
            groups_default=16 ;;
    esac
    PARTITION_GROUPS="${GEN_PARTITION_GROUPS:-$groups_default}"
    for name in NUM_GPUS MAX_SEQ_LEN BATCH_SIZE TOP_K PARTITION_GROUPS; do
        [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be positive" >&2; exit 1; }
    done
    export REPO_ROOT OUTPUT_ROOT MODEL_PATH DRAFT_PROCESSOR NUM_GPUS MAX_SEQ_LEN BATCH_SIZE TOP_K TOP_P
    export POLICY_DATASET INPUT_DATA PARTITION_GROUPS DRY_RUN
    TOTAL_CONVERSATIONS=$("$PYTHON_BIN" - <<'PY'
import fcntl
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path

env = os.environ
source = Path(env["INPUT_DATA"]).resolve()
stat = source.stat()
config = {
    "input": str(source), "input_size": stat.st_size, "input_mtime_ns": stat.st_mtime_ns,
    "model": str(Path(env["MODEL_PATH"]).resolve()), "draft_processor": env["DRAFT_PROCESSOR"],
    "model_config_sha256": hashlib.sha256((Path(env["MODEL_PATH"]) / "config.json").read_bytes()).hexdigest(),
    "max_seq_len": int(env["MAX_SEQ_LEN"]), "top_k": int(env["TOP_K"]), "top_p": float(env["TOP_P"]),
    "partitions": int(env["PARTITION_GROUPS"]) * int(env["NUM_GPUS"]),
    "visual_env": {name: env.get(name) for name in (
        "VLM_MIN_PIXELS", "VLM_MAX_PIXELS", "VLM_VIDEO_MIN_PIXELS", "VLM_VIDEO_MAX_PIXELS")},
    "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "torchcodec", "av")},
}
if not 0 <= config["top_p"] <= 1:
    raise SystemExit("TOP_P must be between 0 and 1.")
manifest = Path(env["OUTPUT_ROOT"]).resolve() / env["POLICY_DATASET"] / "run.json"
dry = env["DRY_RUN"] == "1"
lock = None
if not dry:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lock = (manifest.parent / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
previous = json.loads(manifest.read_text()) if manifest.exists() else None
if not previous and (manifest.parent / "policy").exists() and any((manifest.parent / "policy").rglob("policy-*")):
    raise SystemExit("Existing policies have no run manifest; use a new OUTPUT_ROOT.")
if previous and all(previous.get(key) == value for key, value in config.items()):
    total = previous["total_conversations"]
else:
    if previous:
        raise SystemExit(f"Settings or input changed; use a new OUTPUT_ROOT instead of mixing policies: {manifest}")
    total = 0
    last = b""
    with source.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            total += chunk.count(b"\n")
            last = chunk[-1:]
    total += bool(last and last != b"\n")
    if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
        raise SystemExit("Input changed while counting records.")
if not total:
    raise SystemExit("Input is empty.")
config["total_conversations"] = total
if not dry and not previous:
    tmp = manifest.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n")
    tmp.replace(manifest)
print(total)
PY
)
    prepared_inputs+=("$INPUT_DATA")
    prepared_groups+=("$PARTITION_GROUPS")
    prepared_counts+=("$TOTAL_CONVERSATIONS")
done

previous_job=""
for index in "${!datasets[@]}"; do
    POLICY_DATASET="${datasets[$index]}"
    INPUT_DATA="${prepared_inputs[$index]}"
    PARTITION_GROUPS="${prepared_groups[$index]}"
    TOTAL_CONVERSATIONS="${prepared_counts[$index]}"
    export POLICY_DATASET INPUT_DATA PARTITION_GROUPS
    export TOTAL_CONVERSATIONS
    pending=()
    for ((group=0; group<PARTITION_GROUPS; group++)); do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            printf -v name 'part-%04d' "$((group * NUM_GPUS + gpu))"
            if [[ ! -f "$OUTPUT_ROOT/$POLICY_DATASET/policy/$name/.complete" ]]; then
                pending+=("$group")
                break
            fi
        done
    done
    if [[ "${#pending[@]}" == 0 ]]; then echo "$POLICY_DATASET already complete"; continue; fi
    if [[ "${#pending[@]}" == "$PARTITION_GROUPS" ]]; then
        array="0-$((PARTITION_GROUPS - 1))%1"
    else
        array="$(IFS=,; echo "${pending[*]}")%1"
    fi
    log_dir="$OUTPUT_ROOT/logs/$POLICY_DATASET"
    command=(sbatch --parsable --chdir="$REPO_ROOT" --export=ALL
        --gpus-per-node="$NUM_GPUS" --cpus-per-task="$((NUM_GPUS * 4))"
        --array="$array" --output="$log_dir/%A_%a.out")
    if [[ -n "$previous_job" ]]; then command+=(--dependency="afterok:$previous_job"); fi
    command+=("$REPO_ROOT/examples/speculative_decoding/scripts/generate_external_policies.sbatch")
    echo "$POLICY_DATASET: $TOTAL_CONVERSATIONS inputs, $((PARTITION_GROUPS * NUM_GPUS)) partitions; ${#pending[@]} groups pending, at most one node active."
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "${command[@]}"; printf '\n'
        previous_job='<cosmos3-array-job-id>'
    else
        mkdir -p "$log_dir"
        submission=$("${command[@]}")
        previous_job="${submission%%;*}"
        echo "Submitted $POLICY_DATASET array: $previous_job; logs: $log_dir"
    fi
done
