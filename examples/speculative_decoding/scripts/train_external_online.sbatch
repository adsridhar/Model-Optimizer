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
# Submit from the repository root; rerun the same command to resume the last checkpoint.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --partition=h200-nvl@qs1/genoa2d24g2l/8gpu-256cpu-2304gb,h200@cr+mp/viking@cr+mp/8gpu-224cpu-2048gb,h200-nvl@ts1/genoa2d24g2l/8gpu-256cpu-2304gb,h200-nvl@qs1/x14dbgap/8gpu-288cpu-2304gb
#SBATCH --exclude=smc-522ga-0030,4u8g-gen-0277
#SBATCH --qos=batch-short
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --job-name=cosmos3-external-online
#SBATCH --output=external-online-%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch.}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
source examples/speculative_decoding/scripts/activate_external_draft.sh
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
NUM_NODES="${SLURM_NNODES:-${NUM_NODES:-1}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/../../scratch.adsridhar_other_1/external_draft/online/cosmos3}"
SAVE_STEPS="${SAVE_STEPS:-250}"
for argument in "$@"; do
    case "$argument" in
        training.output_dir=*) OUTPUT_DIR="${argument#*=}" ;;
        training.save_steps=*) SAVE_STEPS="${argument#*=}" ;;
        training.save_total_limit=5) ;;
        training.save_total_limit=*) echo "External online jobs retain 5 checkpoints." >&2; exit 1 ;;
    esac
done
for name in NUM_GPUS NUM_NODES SAVE_STEPS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be positive" >&2; exit 1; }
done
export REPO_ROOT NUM_GPUS NUM_NODES OUTPUT_DIR SAVE_STEPS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR/logs"
exec 9>"$OUTPUT_DIR/.online-training.lock"
flock -n 9 || { echo "Another chained job is using $OUTPUT_DIR" >&2; exit 1; }

if [[ -f "$OUTPUT_DIR/.training-complete" ]]; then
    echo "Training is already complete in $OUTPUT_DIR; skipping model loading."
    exit 0
fi

echo "Training on $NUM_NODES nodes x $NUM_GPUS GPUs; checkpoint every $SAVE_STEPS optimizer steps; retaining 5 checkpoints."
training_args=(examples/speculative_decoding/main.py
    --config examples/speculative_decoding/recipes/cosmos3_super_qwen35_external_online.yaml
    training.output_dir="$OUTPUT_DIR" training.save_steps="$SAVE_STEPS" training.save_total_limit=5 "$@")
if (( NUM_NODES == 1 )); then
    "$EXTERNAL_DRAFT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" "${training_args[@]}"
else
    # One launcher per node; only this batch coordinator owns the output lock.
    export RDZV_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
    export RDZV_PORT="${RDZV_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
    srun --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
        bash examples/speculative_decoding/scripts/train_external_online_worker.sh "${training_args[@]}"
fi
touch "$OUTPUT_DIR/.training-complete"
