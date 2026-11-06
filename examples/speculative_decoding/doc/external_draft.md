# External Draft — Training a Standalone Draft Model

An external draft is an ordinary pretrained causal LM trained to propose tokens for a
target model. Unlike the head-based modes (EAGLE, Medusa, DFlash) nothing is grafted onto
the target: it contributes its vocabulary and, on the dense path, its `lm_head`. The result
is a plain HuggingFace checkpoint.

Recipe: [`modelopt_recipes/general/speculative_decoding/external_draft.yaml`](../../../modelopt_recipes/general/speculative_decoding/external_draft.yaml) |
Swap implementation: [`vocab_swap.py`](../../../modelopt/torch/speculative/external/vocab_swap.py)

## Architecture

```text
Target (frozen, never modified)
  │
  ├─ sparse path:  top-k deployment policy  ──► dumped to disk ──┐
  │                                                              │
  └─ dense path:   hidden states ──► target lm_head ──► logits ──┤
                                                                 │
                                                         TVD / soft_ce
                                                                 │
Draft (a complete causal LM, trained)  ──► its own lm_head ──► draft logits
```

**Key components:**

- **No grafted head.** The draft owns its embeddings and `lm_head` and runs standalone at
  serving time. Engines that already support a separate draft model load it with no
  mode-specific support.
- **Acceptance-shaped objectives.** Under rejection sampling the probability a proposed
  token survives verification is `sum(min(p, q))`, so total-variation distance is
  `1 - acceptance`. Training on TVD optimises the serving metric directly.
- **Two data formats.** The TVD objectives need only the target's truncated next-token
  distribution, which is far smaller than a hidden state per token and keeps the target off
  the GPU at training time. `soft_ce` needs full logits and so needs dumped hidden states.
- **Vocabulary coupling.** The draft must index the target's vocabulary. A draft from
  another family can be re-indexed onto it, at the cost of replacing a large fraction of a
  small model.

## Training

### Quick Start

```bash
# Every command in this section reads $SEQ_LEN; change it here and they stay consistent.
SEQ_LEN=4096

torchrun --nproc_per_node 8 main.py \
  --config ../../modelopt_recipes/general/speculative_decoding/external_draft.yaml \
  model.model_name_or_path=$TARGET \
  draft_model_name_or_path=$DRAFT \
  data.sparse_data_path=$POLICY_DIR \
  data.sparse_cache_dir=$CACHE_DIR \
  training.training_seq_len=$SEQ_LEN \
  training.output_dir=$OUTPUT_DIR
```

### Recipe

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.model_name_or_path` | — | The **target**, used for its vocabulary and `lm_head` |
| `draft_model_name_or_path` | — | The draft, the model actually trained |
| `external.external_loss` | `tvd` | `soft_ce`, `tvd`, `tvd_deploy` or `tvd_ce` (see below) |
| `external.external_top_k` | 20 | Serving top-k; must match the dump |
| `external.external_top_p` | 0.95 | Serving top-p; must match the dump |
| `external.external_tvd_alpha` | 0.9 | Weight on the TVD term under `tvd_ce` |
| `external.external_ce_alpha` | 0.1 | Weight on the CE term under `tvd_ce` |
| `external.external_vocab_swap` | false | Re-index the draft onto the target's vocabulary |
| `external.external_report_acc` | true | Log top-1 agreement during training |
| `data.sparse_data_path` | — | Dumped top-k policy; selects `data.mode=sparse` |
| `data.sparse_cache_dir` | — | Where the memmapped cache is built |
| `data.offline_data_path` | — | Dumped hidden states; the alternative to the above |
| `model.use_fake_base_for_offline` | true | Load only the target's `lm_head`, not its layers |

> **Note on `external_top_k` / `external_top_p`:** the stored policy *is* the teacher
> distribution, so a dump narrower than the objective expects charges the draft for mass on
> tokens the teacher actually supported. A top-k mismatch draws a warning under
> `tvd_deploy`; the dump does not record its top-p, so a top-p mismatch goes undetected.

### Objectives

| `external_loss` | Scores | Needs |
|-----------------|--------|-------|
| `soft_ce` | Cross-entropy against the target's soft distribution (the EAGLE offline objective) | hidden states |
| `tvd` | Total-variation distance between the two distributions | either format |
| `tvd_deploy` | The same, with both sides put through the serving filter first | either format |
| `tvd_ce` | `tvd` plus a cross-entropy term on the target's argmax token | either format |

On the sparse path the target side is truncated by the dump whichever objective runs, since
that is all the dump holds; `tvd` and `tvd_deploy` then differ only in whether the *draft*
is put through the filter too. Only `tvd_deploy` applies the same nucleus to both sides.
Pairing `soft_ce` with `data.sparse_data_path` is a hard error, not a quiet degradation.

Report which truncation you trained under. `top_k` and `top_p` settings are not
interchangeable, so a number quoted without them is not comparable to anything.

`tvd_ce` exists because TVD is bounded and so ranks weakly among tokens inside the target's
support; cross-entropy's log penalty has no such ceiling.

### Teacher policy data

`$CONVERSATIONS` should carry responses the **target generated itself**, not the responses
that shipped with the prompts. The objective trains the draft to match the target at the
positions it will meet when serving, and those lie along the target's own samples; scoring
it against another model's text trains it on positions the target would rarely produce.
[Data Synthesis](../README.md#data-synthesis) covers generating them.

```bash
python collect_hidden_states/compute_sparse_policy_hf.py \
  --model $TARGET --input-data $CONVERSATIONS --output-dir $POLICY_DIR \
  --top-k 20 --top-p 0.95 --max-seq-len $SEQ_LEN \
  --answer-only-loss --chat-template ./template.jinja
```

Set `--max-seq-len` to the `training.training_seq_len` you will train at. The producer skips
whole conversations longer than it, scored turns included, and prints how many it skipped. The
collator pads every sample to `training_seq_len`, so over-sizing spends the difference on
padding and under-sizing drops the long conversations from the dump.

On a large corpus the first run spends a long time building the memmapped cache beside the
shards. Under `torchrun` only rank 0 builds it and the others wait, so build it once in a
single process first and point every run at the same `data.sparse_cache_dir`:

```bash
python - <<PY
from modelopt.torch.speculative.external.sparse_data import (
    SparsePolicyDataset,
    discover_shards,
)

SparsePolicyDataset(
    discover_shards("$POLICY_DIR"), max_length=$SEQ_LEN, cache_dir="$CACHE_DIR"
)
PY
```

Use `discover_shards` and the same `cache_dir` the training run will use: a cache keyed on a
different shard list is ignored, and `torchrun` rebuilds it from scratch. A ragged
third-party dump also needs `top_k=<its top-k>` here, since the cache width is otherwise
inferred from the first record and the recipe has no field to set it.

> **Note on `answer_only_loss` and chat templates:** when `--answer-only-loss` is set, the
> chat template must wrap assistant content in `{% generation %}` / `{% endgeneration %}`
> tags. Most published templates do not. Copy the model's own template and add them:
>
> 1. Print `AutoTokenizer.from_pretrained(TARGET).chat_template` to a file
> 2. Wrap the assistant content each branch emits, including a reasoning branch if present
> 3. Pass it with `--chat-template` to **every** dump command for that target
>
> Verify by decoding, not by counting: `tokenizer.decode(record["gen_ids"])` should read as
> the assistant's reply alone. A misplaced wrap still cuts at the wrong boundary, so a
> length check passes while the span is wrong.

The dense alternative dumps hidden states instead. Pass `--no-aux-hidden-states`: this mode
never reads the aux planes and they dominate the dump size.

```bash
python collect_hidden_states/compute_hidden_states_hf.py \
  --model $TARGET --input-data $CONVERSATIONS --output-dir $HIDDEN_DIR \
  --max-seq-len $SEQ_LEN --no-aux-hidden-states --answer-only-loss --chat-template ./template.jinja
```

### Cross-tokenizer vocabulary swap (`external_vocab_swap`)

Speculative decoding needs the draft to propose tokens the target can verify, so the two
must index the same vocabulary. When the target's family has no small member to draft with,
`external.external_vocab_swap=true` rebuilds the draft's embeddings and `lm_head` against
the target's vocabulary before conversion. Rows for tokens both tokenizers spell identically
carry over; the rest are drawn from the draft's own embedding statistics.

The run reports what carried over, typically a minority of the table and nearly all of the
most frequent ids, since the rare tail is where tokenizers disagree.

> **Note on judging a swap:** the match percentage describes the swap, not the draft.
> Overlap figures far apart have behaved similarly, so do not read the coverage number as a
> quality score. Prefer a vocabulary-matched draft where one exists: the replacement matrix
> is `base_vocab_size × hidden_size` parameters, a large fraction of a small draft, and a
> swapped draft stays behind a matched one of similar size.

A swapped draft starts with a large untrained fraction and wants a higher learning rate than
a matched draft of comparable size.

### Checkpoint resume

A resumed draft keeps the embeddings it was saved with. The swap reads
`external_vocab_swap` out of the checkpoint's own ModelOpt state, so re-running with the
flag on does not rewrite rows that are already base-indexed.

### Export

Training writes an ordinary HuggingFace checkpoint, shipped with the **target's** tokenizer
when the vocabulary was swapped, since the draft's rows are indexed by the target's ids from
that point on. Nothing in the saved artifact depends on ModelOpt at load time.

**There is no export step for this mode.** `scripts/export_hf_checkpoint.py` restructures a
grafted head into a servable checkpoint; an external draft has no head to restructure, so the
script rejects it. The training output is the artifact.

`training.output_dir` also holds training residue that `from_pretrained` ignores:
`modelopt_state.pth`, `trainer_state.json`, `training_args.bin` and the `checkpoint-*`
directories. Ship `config.json`, `generation_config.json`, the weights and the tokenizer
files.

## Reference Configuration (Llama-3.2-1B drafting for Qwen3.8-27B)

A cross-tokenizer draft: Llama-3.2-1B-Instruct re-indexed onto Qwen3.8-27B's
248,320-token vocabulary. Trained on a sparse teacher policy dumped once from the target,
8 GPUs, one epoch.

### Teacher policy

The [teacher policy data](#teacher-policy-data) dump, with `--max-seq-len 12288` to match
`training.training_seq_len` below.

### Training

```bash
torchrun --nproc_per_node 8 main.py \
  --config ../../modelopt_recipes/general/speculative_decoding/external_draft.yaml \
  model.model_name_or_path=$TARGET \
  draft_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
  external.external_vocab_swap=true \
  external.external_loss=tvd \
  external.external_top_k=20 external.external_top_p=0.95 \
  data.sparse_data_path=$POLICY_DIR \
  data.sparse_cache_dir=$CACHE_DIR \
  training.training_seq_len=12288 \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=2 \
  training.learning_rate=2.0e-5 \
  training.warmup_steps=250 \
  training.max_grad_norm=1.0 \
  training.num_train_epochs=1 \
  training.seed=42 \
  training.output_dir=$OUTPUT_DIR
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Objective | `tvd` |
| Vocabulary swap | yes (Llama tokenizer onto Qwen's) |
| Sequence length | 12,288 |
| Effective batch | 16 (1 per device × 2 accumulation × 8 GPUs) |
| Learning rate | 2.0e-5 |
| Warmup steps | 250 |
| Epochs | 1 |
| Serving policy | top-k 20, top-p 0.95, temperature 1.0 |

> **Note on the learning rate:** we recommend sweeping to find the best rate for your own
> draft and target.

## Cosmos3 Super → vision-capable Qwen3.5-0.8B

The initial multimodal path supports one final assistant turn per conversation,
DDP (no FSDP/DeepSpeed), and sparse policies collected offline or computed online. Both models process the
original media through their own vision stacks. Qwen's vision encoder is frozen; its `visual.merger`,
language model, embeddings and output head train. Media files are referenced, not
copied into the policy cache.

The environment must support Qwen3.5 and have a working Transformers video decoder.
Check a real video before collection; newer torchvision versions without
`read_video` need a working TorchCodec installation.

Run from the repository root. Collect each source into a separate directory:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/speculative_decoding/collect_hidden_states/compute_sparse_policy_hf.py \
  --model ../models/cosmos3-super-reasoner_vbf16-final \
  --input-data ../../scratch.adsridhar_other_1/cosmos3_super_dflash_train_full_8192_partial.jsonl \
  --output-dir ../../scratch.adsridhar_other_1/external_draft/cosmos3/policy \
  --multimodal --draft-processor Qwen/Qwen3.5-0.8B \
  --max-seq-len 16384 --top-k 20 --top-p 0.95 --answer-only-loss
```

Under `torchrun`, `RANK` and `WORLD_SIZE` select disjoint input partitions and
`LOCAL_RANK` pins each full teacher replica to its GPU. Explicit `--dp-rank` and
`--dp-world-size` override the partition defaults. The debug conversation limit
applies to examined records per worker: `--debug-max-num-conversations 1` tries
two examples with two workers; over-length examples can still be skipped.
Direct `python` invocation remains a single worker with automatic
model placement.

Teacher scoring accepts `--batch-size N` per GPU (default 1). Processed
conversations are right-padded, while each saved policy retains its own unpadded
prompt, response, and media references. Different response positions are aligned
before extracting top-k probabilities. A final smaller batch is also scored.
This option is currently supported only with `--multimodal`.

For Slurm, set `BATCH_SIZE`, for example:

```bash
BATCH_SIZE=2 bash examples/speculative_decoding/scripts/submit_external_policy_gen.sh --dry-run cosmos3
```

Profile before increasing it: long video sequences can be faster at batch size 1.
A real H200 smoke at roughly 12,500 tokens used about 65.5 / 69.1 / 76.1 GiB peak
allocated GPU memory at batch sizes 1 / 2 / 4 respectively. These are small-sample
measurements, not worst-case bounds for the full dataset. Batched BF16 kernels can
slightly change probabilities and near-tied top-k choices relative to single-record
scoring.

Structured image/video messages and LLaVA's `<image>` plus `images` format are
accepted. `<image>` is a media placeholder only when image paths are declared;
text-only HTML/XML examples retain literal `<image>` strings. Both processors honor per-record video `fps`; videos within one example
must use the same rate. Collection checks processed lengths for both models and skips examples
above the limit. It verifies identical response tokens before scoring the teacher.
ChatML assistant boundaries provide the response mask, including reasoning text.
Custom templates must preserve those boundaries. Keep visual preprocessing settings
(including `VLM_MIN_PIXELS` / `VLM_MAX_PIXELS`) identical between collection and
training; do not enable text-truncation environment variables.

For full collection on the reference H200 Slurm configuration, use the submission
wrapper. It scores existing responses; it does not generate new responses:

```bash
bash examples/speculative_decoding/scripts/submit_external_policy_gen.sh --dry-run both
# Submit Cosmos3 first, then queue LLaVA after successful completion:
bash examples/speculative_decoding/scripts/submit_external_policy_gen.sh both
```

The defaults match `../Model-Optimizer/examples/llm_distill/cosmos3_tvd/`:
one node with eight H200s, the same partition list and exclusions, `batch-short`,
and a four-hour limit per array task. At most one node runs at once. Cosmos3 uses
128 array tasks and LLaVA uses 16; each task runs eight independent teacher
replicas through `torchrun`. `GEN_PARTITION_GROUPS` overrides the number of array
tasks; use more, smaller partitions if a task exceeds the time limit.
Partition sizing has not been validated by a full-duration run.

Outputs default to `../../scratch.adsridhar_other_1/external_draft`:

- `<dataset>/policy/part-*/`: completed partitions, consumed recursively by cache creation.
- `<dataset>/staging/`: active partitions, excluded from training.
- `<dataset>/incomplete/`: preserved interrupted attempts, excluded from training.
- `logs/<dataset>/<array-job>_<task>.out`: Slurm output; each partition also has `worker.log`.

Each worker reports percentage, ETA, and written/skipped/invalid counts every
30 seconds. Counts measure examined inputs; over-length examples are skipped.
A `policy/part-*/.complete` marker records completion. Resubmitting the same
command skips completed partitions and restarts interrupted partitions from their
beginning. An interrupted partition has no per-record resume. When using `both`,
a failed Cosmos3 array prevents its dependent LLaVA array from starting; resolve
the failure and submit the missing stages again.

The wrapper records input size/mtime, processor, scoring settings, and environment
versions in `<dataset>/run.json`. Keep the source JSONL and media fixed during
collection. Changed settings or input require a fresh `OUTPUT_ROOT`; do not mix
manual exports into a managed output directory. `--dry-run` counts source records
and previews submission without writing artifacts or submitting jobs. The worker
uses the repository `scripts/activate_external_draft.sh` environment.

Prebuild the cache as described above, then train:

```bash
torchrun --nproc_per_node 8 examples/speculative_decoding/main.py \
  --config examples/speculative_decoding/recipes/cosmos3_super_qwen35_external.yaml
```

The recipe swaps Qwen onto Cosmos3's vocabulary and remaps visual special-token
ids. Each teacher response offset is mapped to Qwen's own response position, so
different visual-token counts do not shift the loss. Training refuses a response
token mismatch or an oversized draft sequence rather than silently truncating it.
The checkpoint includes Qwen's visual processors and the shared target tokenizer.

`training.per_device_train_batch_size` controls the number of examples per GPU.
Variable-length examples are padded to the longest sequence in that batch, and
each response keeps its own teacher-policy alignment. Effective batch size is
per-device batch × GPU count × gradient accumulation steps; profile memory at the
intended sequence length and visual caps before increasing it.
The recipe uses batch size 2 with non-reentrant gradient checkpointing for long
sequences and a 16,384-token sequence limit, matching policy collection. With two
GPUs and two accumulation steps, its effective batch is 8. Use
`max_length=16384` when prebuilding its sparse cache.

For stage 2, collect policies from
`../LlamaFactory/data/llava_instruct_gqa_sft.jsonl` into a separate directory, then
override `draft_model_name_or_path` and `data.vlm_processor` with the stage-1 output,
set the stage-2 policy/cache paths, and choose a fresh `training.output_dir`.
This starts a new optimizer schedule from the trained draft weights.

A real teacher-policy batch has passed a draft backward step with production
visual defaults. Multimodal acceptance, serving compatibility, and full-run
stability still need validation. The existing grafted-head
AR validation script does not validate a separate target/draft pair.

## Online Cosmos3 Super → Qwen3.5 training

The online recipe reads the original JSONL and scores the frozen Cosmos3 teacher
before each draft forward. No teacher-policy files or sparse cache are created.
The teacher and draft use their own visual processors; assistant token IDs are
checked and response positions aligned before the existing sparse loss runs.
Only the draft participates in DDP, optimization, and checkpoint saves. Its vision
encoder stays frozen while `visual.merger` and the language model train.

```bash
source examples/speculative_decoding/scripts/activate_external_draft.sh
torchrun --standalone --nproc_per_node=8 examples/speculative_decoding/main.py \
  --config examples/speculative_decoding/recipes/cosmos3_super_qwen35_external_online.yaml
```

For the same H200 partitions as policy collection, submit an explicit `afterany` chain:

```bash
bash examples/speculative_decoding/scripts/submit_external_online.sh \
  --runs 5 --nodes 1 --num-gpus 8 \
  --output-dir ../../scratch.adsridhar_other_1/external_draft/qwen3.5-cosmos3-super \
  --save-steps 250
```

The submitter queues all requested jobs immediately, each dependent on termination
of its predecessor with `afterany`, matching the neighboring Model-Optimizer runner.
A failure or timeout does not stop later jobs from attempting checkpoint resume.
Each allocation has a four-hour limit and explicitly requests all node host RAM
(`--mem=0`); the previous CPU-derived RAM allocation was insufficient for eight
online teacher/draft workers. All job IDs are printed; cancel all remaining IDs
to stop the chain. After successful training and final export, a small
`.training-complete` marker makes remaining jobs exit without loading models.
Remove that marker only when intentionally extending a completed run in place.

`--runs`, `--num-gpus`, and `--output-dir` are required. `--nodes` defaults to one,
and `--num-gpus` means GPUs per node. Checkpoint retention is five. `--save-steps`
defaults to **250** and is honored after resume. Checkpoints and Slurm logs use
`--output-dir`, with logs under `logs/<job-id>.out`. `--dry-run` previews the whole
chain without submitting or writing artifacts. Stop any local training using the
same output directory before submitting.

For two nodes, one `srun` task on each node starts a torchrun agent. Agents share
one rendezvous endpoint; only the batch coordinator acquires the output lock.
Manifest creation uses global, rather than node-local, rank ordering. Each GPU
still holds a complete teacher and draft, with DDP synchronizing draft gradients.
The Python environment, source JSONL, all media, and checkpoints must be accessible
at the same paths on both nodes. NCCL and two real teacher/draft optimizer steps
passed on two physical H200 nodes with one GPU per node, without saving model
checkpoints. Train/resume also passed with two independent one-GPU agents on one
host; full sixteen-GPU throughput has not been measured.

```bash
bash examples/speculative_decoding/scripts/submit_external_online.sh \
  --runs 5 --nodes 2 --num-gpus 8 \
  --output-dir ../../scratch.adsridhar_other_1/external_draft/qwen3.5-cosmos3-super \
  --save-steps 250 \
  training.gradient_accumulation_steps=4
```

Two nodes x eight GPUs x batch 1 x accumulation 4 keeps the same global batch 64
as one node x eight GPUs x accumulation 8. Keep that global batch when resuming
checkpoint 1000 so the optimizer-step/data-offset mapping remains consistent.
Changing world size is not bitwise reproducible, and new ranks have no matching
saved per-rank RNG state. Additional recipe overrides can follow the submitter
options. A single allocation is also available through
`sbatch examples/speculative_decoding/scripts/train_external_online.sbatch`.

The shared activation script defaults `CUDA_HOME` to the validated public CUDA
13.3.1 toolkit at
`/home/scratch.svc_compute_arch/release/cuda_toolkit/public/13.3.1/x86_64/u24.04`
and adds its compiler to `PATH`. Set `EXTERNAL_DRAFT_CUDA_HOME` before sourcing
activation to select another toolkit. This replaces inherited `CUDA_HOME` values,
including malformed ones; selecting a compiler does not install optional FLA or
causal-convolution kernels.

Start with batch size 1 per GPU and gradient checkpointing. The recipe uses eight
accumulation steps, for an effective batch of 64 with eight GPUs. Each GPU holds
one full frozen teacher plus one draft; measure combined memory before increasing
batch size. Teacher scoring is repeated each epoch.

The loader builds an in-memory byte-offset index of one JSONL file, preserving
heterogeneous message/media structures without copying the dataset. Files and
media must remain fixed throughout training. `online_data.json` records the source
size/mtime, teacher configuration, processors, and scoring settings. A changed
manifest refuses reuse of the output directory. Normal Trainer checkpoint resume
restores the optimizer/scheduler/RNG and skips consumed dataset indices; the teacher
is reloaded from its original checkpoint. The first dataset is indexed again on
startup, including resume.

Examples exceeding 16,384 processed tokens for either model contribute no loss.
An entirely filtered local batch uses a tiny zero-loss draft forward/backward so
DDP ranks issue matching collectives. When an entire accumulation window is filtered
across all ranks, parameters are not updated; the training step and learning-rate
schedule still advance. `online_skipped_local` reports filtered records per rank
since process startup. Malformed media or response-token mismatches fail loudly.

For stage 2, set `data.data_path=../LlamaFactory/data/llava_instruct_gqa_sft.jsonl`,
`draft_model_name_or_path` and `data.vlm_processor` to the stage-1 output, and choose
a fresh `training.output_dir`. The fixed teacher remains Cosmos3 Super. Online
support initially covers this pair, DDP, and the existing sparse losses
(`tvd`, `tvd_deploy`, `tvd_ce`); full-distribution distillation and remote teachers
are outside this path.

## Open Items

### Validated end-to-end

- Sparse teacher-policy training with `tvd`, from dump through training to a loadable
  checkpoint.
- The cross-tokenizer vocabulary swap, on a draft trained this way.

### Covered by tests, not yet exercised in a full run

- `soft_ce`, `tvd_deploy` and `tvd_ce`, and the dense hidden-states data path.
- Resume, with and without a swap.
- Untied output heads, and heads carrying a bias.

### Not yet validated

- Streaming the teacher data from a live `vllm serve`. It is accepted for this mode but has
  not been exercised. The Cosmos3/Qwen3.5 pair has a separate local online path above.
