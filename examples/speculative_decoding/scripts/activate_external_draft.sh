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

# Source from the repository root. EXTERNAL_DRAFT_CUDA_HOME overrides the shared toolkit.
source .venv/bin/activate
export CUDA_HOME="${EXTERNAL_DRAFT_CUDA_HOME:-/home/scratch.svc_compute_arch/release/cuda_toolkit/public/13.3.1/x86_64/u24.04}"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "Missing CUDA compiler: $CUDA_HOME/bin/nvcc" >&2
    return 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export EXTERNAL_DRAFT_PYTHON="$VIRTUAL_ENV/bin/python"
export PYTHONPATH="$PWD:$PWD/tests:$PWD/examples/speculative_decoding${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/ffmpeg:$VIRTUAL_ENV/lib/python3.12/site-packages/av.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="$VIRTUAL_ENV/.cache/torch-extensions"
export TRITON_CACHE_DIR="$VIRTUAL_ENV/.cache/triton"
export CUDA_CACHE_PATH="$VIRTUAL_ENV/.cache/cuda"
