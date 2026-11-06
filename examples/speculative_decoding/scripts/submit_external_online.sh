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

# Submit the complete resume-enabled afterany chain before training starts.
set -euo pipefail
usage() {
    cat <<'USAGE'
Usage: bash submit_external_online.sh --runs N --num-gpus N --output-dir PATH [options] [training.key=value ...]
  --nodes N        Nodes per allocation (default: 1); --num-gpus is GPUs per node.
  --save-steps N    Optimizer steps between checkpoints (default: 250). Five checkpoints are retained.
  --dry-run        Print every submission without writing files or submitting jobs.
USAGE
}
RUNS=""
NUM_GPUS=""
NUM_NODES=1
OUTPUT_DIR=""
SAVE_STEPS=250
DRY_RUN=0
training_args=()
while (( $# )); do
    case "$1" in
        --runs|--num-gpus|--nodes|--output-dir|--save-steps)
            (( $# >= 2 )) || { usage >&2; exit 1; }
            case "$1" in
                --runs) RUNS=$2 ;;
                --num-gpus) NUM_GPUS=$2 ;;
                --nodes) NUM_NODES=$2 ;;
                --output-dir) OUTPUT_DIR=$2 ;;
                --save-steps) SAVE_STEPS=$2 ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        training.output_dir=*|training.save_steps=*|training.save_total_limit=*)
            echo "Use --output-dir/--save-steps; save_total_limit is fixed at 5." >&2; exit 1 ;;
        --*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
        *=*) training_args+=("$1"); shift ;;
        *) echo "Expected a training.key=value override: $1" >&2; exit 1 ;;
    esac
done
for name in RUNS NUM_GPUS NUM_NODES SAVE_STEPS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "--runs, --nodes, --num-gpus, and --save-steps must be positive integers" >&2; exit 1; }
done
(( NUM_GPUS <= 8 )) || { echo "Each H200 node supports at most 8 GPUs." >&2; exit 1; }
[[ -n "$OUTPUT_DIR" ]] || { echo "--output-dir is required" >&2; exit 1; }
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
OUTPUT_DIR=$(realpath -m -- "$OUTPUT_DIR")
export REPO_ROOT NUM_GPUS NUM_NODES OUTPUT_DIR SAVE_STEPS
previous_job=""
echo "Submitting $RUNS allocations, $NUM_NODES nodes x $NUM_GPUS GPUs per node, linked by afterany."
echo "Checkpoint every $SAVE_STEPS optimizer steps; retaining 5 checkpoints. Output: $OUTPUT_DIR"
if (( ! DRY_RUN )); then mkdir -p "$OUTPUT_DIR/logs"; fi
for ((run=1; run<=RUNS; run++)); do
    command=(sbatch --parsable --chdir="$REPO_ROOT" --export=ALL
        --nodes="$NUM_NODES" --ntasks-per-node=1 --mem=0
        --gpus-per-node="$NUM_GPUS" --cpus-per-task="$((NUM_GPUS * 4))"
        --output="$OUTPUT_DIR/logs/%j.out")
    if [[ -n "$previous_job" ]]; then command+=(--dependency="afterany:$previous_job"); fi
    command+=("$REPO_ROOT/examples/speculative_decoding/scripts/train_external_online.sbatch"
        "${training_args[@]}")
    if (( DRY_RUN )); then
        printf '%q ' "${command[@]}"; printf '\n'
        previous_job="<run-$run-job-id>"
    else
        submission=$("${command[@]}")
        previous_job="${submission%%;*}"
        echo "Queued run $run/$RUNS: $previous_job"
    fi
done
