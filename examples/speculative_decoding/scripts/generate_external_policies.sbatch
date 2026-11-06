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
# Submit through submit_external_policy_gen.sh so partition counts and input totals are fixed.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --partition=h200-nvl@qs1/genoa2d24g2l/8gpu-256cpu-2304gb,h200@cr+mp/viking@cr+mp/8gpu-224cpu-2048gb,h200-nvl@ts1/genoa2d24g2l/8gpu-256cpu-2304gb,h200-nvl@qs1/x14dbgap/8gpu-288cpu-2304gb
#SBATCH --exclude=smc-522ga-0030,4u8g-gen-0277
#SBATCH --qos=batch-short
#SBATCH --time=04:00:00
#SBATCH --job-name=cosmos3-external-policies
#SBATCH --signal=TERM@120

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
source examples/speculative_decoding/scripts/activate_external_draft.sh
: "${SLURM_JOB_ID:?Submit with sbatch through submit_external_policy_gen.sh.}"
: "${POLICY_DATASET:?Missing dataset selection.}"
: "${INPUT_DATA:?Missing input path.}"
: "${OUTPUT_ROOT:?Missing output root.}"
: "${PARTITION_GROUPS:?Missing partition count.}"
: "${TOTAL_CONVERSATIONS:?Missing input record count.}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TOP_K="${TOP_K:-20}"
TOP_P="${TOP_P:-0.95}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/../models/cosmos3-super-reasoner_vbf16-final}"
DRAFT_PROCESSOR="${DRAFT_PROCESSOR:-Qwen/Qwen3.5-0.8B}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-30}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

if [[ "${1:-}" != --worker ]]; then
    [[ "${SLURM_ARRAY_TASK_ID:-0}" -lt "$PARTITION_GROUPS" ]] || {
        echo "Array index exceeds PARTITION_GROUPS=$PARTITION_GROUPS" >&2; exit 1;
    }
    exec "$EXTERNAL_DRAFT_PYTHON" -m torch.distributed.run \
        --standalone --nproc_per_node="$NUM_GPUS" --no-python \
        bash "$REPO_ROOT/examples/speculative_decoding/scripts/generate_external_policies.sbatch" --worker
fi

partition_rank=$((SLURM_ARRAY_TASK_ID * NUM_GPUS + LOCAL_RANK))
partition_world=$((PARTITION_GROUPS * NUM_GPUS))
printf -v partition_name 'part-%04d' "$partition_rank"
dataset_root="$OUTPUT_ROOT/$POLICY_DATASET"
staging="$dataset_root/staging/$partition_name"
published="$dataset_root/policy/$partition_name"
mkdir -p "$dataset_root/locks" "$dataset_root/staging" "$dataset_root/policy"
exec 9>"$dataset_root/locks/$partition_name.lock"
flock -n 9 || { echo "$partition_name is already running" >&2; exit 1; }

validate_inputs() {
"$EXTERNAL_DRAFT_PYTHON" - "$dataset_root/run.json" <<'PY'
import importlib.metadata
import hashlib
import json
import os
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
stat = Path(config["input"]).stat()
if (stat.st_size, stat.st_mtime_ns) != (config["input_size"], config["input_mtime_ns"]):
    raise SystemExit("Input changed after submission; use a new output root.")
for package, version in config["versions"].items():
    if importlib.metadata.version(package) != version:
        raise SystemExit(f"{package} changed after submission; use a consistent environment.")
for name, value in config["visual_env"].items():
    if os.environ.get(name) != value:
        raise SystemExit(f"{name} changed after submission.")
if hashlib.sha256((Path(config["model"]) / "config.json").read_bytes()).hexdigest() != config["model_config_sha256"]:
    raise SystemExit("Teacher config changed after submission.")
PY
}
validate_inputs

if [[ -f "$published/.complete" ]]; then
    echo "$partition_name already complete; skipping"
    exit 0
fi
[[ ! -e "$published" ]] || { echo "Inspect incomplete published directory: $published" >&2; exit 1; }
if [[ -e "$staging" ]]; then
    mkdir -p "$dataset_root/incomplete"
    archive=$(mktemp -d "$dataset_root/incomplete/$partition_name.XXXXXX")
    mv -- "$staging" "$archive/work"
    echo "Preserved interrupted attempt in $archive"
fi
mkdir -p "$staging"
echo "$partition_name: GPU=$LOCAL_RANK, partition=$partition_rank/$partition_world"
"$EXTERNAL_DRAFT_PYTHON" -u \
    examples/speculative_decoding/collect_hidden_states/compute_sparse_policy_hf.py \
    --model "$MODEL_PATH" --input-data "$INPUT_DATA" --output-dir "$staging" \
    --multimodal --draft-processor "$DRAFT_PROCESSOR" --answer-only-loss \
    --batch-size "$BATCH_SIZE" --max-seq-len "$MAX_SEQ_LEN" --top-k "$TOP_K" --top-p "$TOP_P" \
    --dp-rank "$partition_rank" --dp-world-size "$partition_world" \
    --total-conversations "$TOTAL_CONVERSATIONS" \
    2>&1 | tee "$staging/worker.log"
validate_inputs
touch "$staging/.complete"
mv -- "$staging" "$published"
echo "$partition_name complete: $published"
