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
# Launched once per node by srun; chain coordination stays in the batch script.
set -euo pipefail
cd "$REPO_ROOT"
source examples/speculative_decoding/scripts/activate_external_draft.sh
exec "$EXTERNAL_DRAFT_PYTHON" -m torch.distributed.run \
    --nnodes="$NUM_NODES" --nproc_per_node="$NUM_GPUS" \
    --rdzv-backend=c10d --rdzv-id="$SLURM_JOB_ID" --rdzv-endpoint="$RDZV_HOST:$RDZV_PORT" \
    "$@"
