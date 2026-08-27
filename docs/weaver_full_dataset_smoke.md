# WEAVER × full B1K smoke protocol

This protocol uses the complete `b1k-v3again` metadata/data path. It does not
launch a task-specific evaluator and does not use the old radio scripts. The
materialized full run must use a new directory such as `numeric-full-v6`; do
not resume or overwrite an older `numeric-full` schema.

The gates are deliberately separate:

1. released WEAVER proves it can run on its own DROID/OOD artifact;
2. full B1K metadata and numeric episodes prove the 30 Hz → 5 Hz, 61D → 23D contract;
3. B1K frames prove the released SD3 visual path and named latent cache;
4. `BehaviorPrecomputed` proves the exact history/future/memory batch contract;
5. only after those pass should a transferred 23D model run a forward/tiny-overfit smoke.

## 1. Pull the branch and run the CPU contract gate

```bash
source ~/.bashrc
openpion
cd "$OPENPI_WEAVER_ROOT"

git fetch origin tc-b1k-weaver
git switch tc-b1k-weaver
git pull --ff-only origin tc-b1k-weaver
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/tc-b1k-weaver)"
test -z "$(git status --porcelain)"
git ls-files --error-unmatch \
  scripts/b1k/prepare_weaver_dataset_parallel.py \
  scripts/b1k/prepare_weaver_dataset_parallel_test.py \
  scripts/b1k/export_weaver_frames.py \
  scripts/b1k/cache_weaver_latents.py \
  src/openpi/weaver/frame_cache.py

uv run pytest --confcutdir=src/openpi/weaver \
  src/openpi/weaver/b1k_contract_test.py \
  src/openpi/weaver/candidate_pool_test.py \
  src/openpi/weaver/checkpoint_transfer_test.py \
  src/openpi/weaver/b1k_dataset_adapter_test.py \
  src/openpi/weaver/frame_cache_test.py \
  src/openpi/weaver/latent_cache_test.py \
  src/openpi/weaver/behavior_precomputed_test.py \
  src/openpi/weaver/norm_stats_test.py -q

uv run pytest -q \
  scripts/b1k/prepare_weaver_dataset_test.py \
  scripts/b1k/prepare_weaver_dataset_parallel_test.py
```

Pass: all tests pass; there is no skipped shape, NaN, network access, or GPU
dependency in this gate.

## 2. Probe a deterministic cross-task subset of the full dataset

Choose a new output directory. The writer refuses to overwrite an existing
artifact.

```bash
export B1K_DATASET_ROOT=/inspire/dataset/behavior-1k-2026-challenge-demos/v3again
export WEAVER_WORK_ROOT=/path/to/a/large/writable/weaver-work
export NUMERIC_SMOKE="$WEAVER_WORK_ROOT/numeric-cross-task"

test -d "$B1K_DATASET_ROOT/meta"
test ! -e "$NUMERIC_SMOKE"

uv run python scripts/b1k/prepare_weaver_dataset.py \
  --dataset-root "$B1K_DATASET_ROOT" \
  --output-dir "$NUMERIC_SMOKE" \
  --sample-tasks 4 \
  --episodes-per-task 2 \
  --selection-seed 2026 \
  --min-sampled-frames 52 \
  --dry-run
```

The dry run decodes no video and writes nothing. Pass only if it reports four
distinct tasks/eight episodes, every episode has strictly increasing timestamps,
the maximum 5 Hz timing error is finite and within the adapter threshold, and no
episode is rejected.

Materialize the exact same subset through the parallel entry point. Each worker
opens an independent numeric-only LeRobot view; only the coordinator commits
the final NPZ/JSON pairs and manifest index:

```bash
uv run python scripts/b1k/prepare_weaver_dataset_parallel.py \
  --dataset-root "$B1K_DATASET_ROOT" \
  --output-dir "$NUMERIC_SMOKE" \
  --sample-tasks 4 \
  --episodes-per-task 2 \
  --selection-seed 2026 \
  --min-sampled-frames 52 \
  --workers 2 \
  --worker-threads 1 \
  --manifest-flush-every 2
```

Pass:

- `summary.json` says `processed=8`, has four selected tasks, and has an empty
  `rejected` list;
- each NPZ contains finite `(T,23)` `states`, `(T,23)`
  `state_delta_actions`, and a separate `(T,23)`
  `policy_actions_at_frame`;
- `transition_valid` is true except at the final frame;
- the source policy action is never relabeled as the WEAVER state-delta action;
- the manifest names the two representations and records the dataset metadata
  hash.

The source `next.reward`, `next.terminated`, and `next.truncated` are preserved
in the numeric NPZ and the 5 Hz `rewards` field is the audited half-open sum
over each sampled interval. It is not an endpoint sample and is not silently
treated as a RoboMeter state-progress label. Full-mode windows reserve one
additional real successor so the terminal sentinel never reaches the frozen
reward/critic heads.

BEHAVIOR's LeRobot writer records `next.terminated`, `next.truncated`, and
`next.reward` for every exported simulator step. A success marker may remain
true while the official exporter continues recording observations, actions, and
later progress rewards. Therefore the `[dataset_from_index,
dataset_to_index)` episode offsets in `meta/episodes` are the only authoritative
episode boundary. The adapter preserves all source rows and all three source
labels, audits marker runs and the unrepresented source tail, and never crops or
rejects a row solely because a marker is true. Only the final sampled
observation is invalid for supervised transition training because it has no
later sampled observation; its source tail remains preserved for audit.

Before the actual full-data run, benchmark real application concurrency rather
than CPU affinity alone. Use the same episode list for every run:

```bash
for workers in 1 2 4 8; do
  BENCH_RUNTIME=parallel \
  BENCH_MODE=dry-run \
  WORKERS="$workers" \
  THREADS=1 \
  DATASET_ROOT="$B1K_DATASET_ROOT" \
  scripts/b1k/benchmark_weaver_prepare.sh
done
```

Compare `throughput_source_rows_per_second` and the stage timings in the final
summary. Do not jump directly to 55 workers: every worker opens its own dataset
view, so excess workers can amplify GPFS contention. A dry run excludes NPZ
compression, hashes, fsync, JSON/index writes, frame decode, and SD3 encoding;
it is not a full preparation benchmark.

For the actual full-data preparation, use the parallel command without subset
selection and write to a fresh output directory. Start with the best measured
2/4/8-worker configuration:

```bash
export NUMERIC_FULL="$WEAVER_WORK_ROOT/numeric-full-v6"
test ! -e "$NUMERIC_FULL"

uv run python scripts/b1k/prepare_weaver_dataset_parallel.py \
  --dataset-root "$B1K_DATASET_ROOT" \
  --output-dir "$NUMERIC_FULL" \
  --held-out-task-fraction 0.1 \
  --selection-seed 2026 \
  --split-seed 42 \
  --min-sampled-frames 10 \
  --workers 4 \
  --worker-threads 1 \
  --manifest-flush-every 256
```

The per-episode NPZ/JSON pair is the durable source of truth. The coordinator
checkpoints the rebuildable `manifest.jsonl` every 256 successful episodes and
again during orderly shutdown, avoiding 20,000 full index rewrites and fsyncs.
After an abrupt interruption, the next run scans and byte-validates every
complete pair before rebuilding the index; a JSON-only/NPZ-only, stale, or
corrupt pair still fails closed. `summary.json` and `task_prompts.json` appear
only after a normal complete run and remain mandatory downstream gates.

The default deterministic task-level holdout is 10 of the 100 official task
names; an explicit `--held-out-tasks` list takes precedence. The writer records
the official `meta/tasks.jsonl` prompt catalog and hashes `info.json`,
`tasks.jsonl`, `tasks.parquet`, and episode metadata. It rejects a dataset that
is not the BEHAVIOR-1K v3 contract (30 Hz, 20,000 episodes, 100 tasks, R1Pro,
23-D action, 61-D proprio).

## 3. Reproduce the released WEAVER checkpoint independently

Freeze both Git and Hugging Face revisions in the log. The release currently
documents Python 3.11 plus `uv`.

```bash
export UPSTREAM_WEAVER_ROOT=/path/to/WEAVER
export WEAVER_RELEASE=/path/to/weaver-release
export DROID_SMOKE=/path/to/droid-ood-smoke
export UPSTREAM_OUT="$WEAVER_WORK_ROOT/upstream-release-smoke"

git -C "$UPSTREAM_WEAVER_ROOT" fetch origin
git -C "$UPSTREAM_WEAVER_ROOT" switch --detach 3e41e907
cd "$UPSTREAM_WEAVER_ROOT"
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv pip freeze > "$WEAVER_WORK_ROOT/weaver-pip-freeze.txt"

WEAVER_HF_REV=$(python -c \
  'from huggingface_hub import model_info; print(model_info("arnavkj1995/WEAVER").sha)')
DROID_HF_REV=$(python -c \
  'from huggingface_hub import dataset_info; print(dataset_info("yilin-wu/droid_ood_data").sha)')
printf 'WEAVER_HF_REV=%s\nDROID_HF_REV=%s\n' \
  "$WEAVER_HF_REV" "$DROID_HF_REV" | \
  tee "$WEAVER_WORK_ROOT/upstream-hf-revisions.env"

hf download arnavkj1995/WEAVER \
  --revision "$WEAVER_HF_REV" \
  --include 'WEAVER/*' \
  --local-dir "$WEAVER_RELEASE"

hf download yilin-wu/droid_ood_data \
  --repo-type dataset \
  --revision "$DROID_HF_REV" \
  --include \
    'annotation_rewards/val/0.json' \
    'latents/val/0_sd3.npz' \
    'videos/val/0.mp4' \
    'norm_stats_relabel.json' \
  --local-dir "$DROID_SMOKE"
```

Run a deliberately small generation:

```bash
LEN=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["video_length"])' \
  "$DROID_SMOKE/annotation_rewards/val/0.json")
START=$((LEN-3))
test ! -e "$UPSTREAM_OUT"

CUDA_VISIBLE_DEVICES=0 uv run python -m weaver.generate_views \
  --checkpoint "$WEAVER_RELEASE/WEAVER" \
  --output-dir "$UPSTREAM_OUT" \
  --split val \
  --num-samples 1 \
  --num-videos 1 \
  --start-idx "$START" \
  --use-real-history \
  --overrides \
    dataset.path="$DROID_SMOKE" \
    dataset.annotation_dir=annotation_rewards \
    dataset.norm_stats_path="$WEAVER_RELEASE/WEAVER/norm_stats_relabel.json" \
    model.val_steps=2 \
    eval_horizon=2 \
    eval_bootstrap=2 \
    inference.pyramid_stagger_width=0
```

Pass: CUDA is used; model and EMA load without missing/unexpected keys; exactly
one trajectory is saved; GT/pred arrays have matching finite shapes; prediction
is neither all zero nor constant; the MP4 decodes. Record wall time and
`nvidia-smi` peak memory. This gate says nothing yet about B1K transfer.

## 4. Bridge B1K pixels across two isolated environments, then encode

Do not install LeRobot into the official WEAVER environment. The bridge has two
explicit processes and one byte-verifiable intermediate artifact:

1. OmegaSupreme + pinned LeRobot decode exact absolute rows to named uint8 RGB;
2. official WEAVER Python 3.11 reads only that artifact and runs the SD3 VAE.

### 4A. Decode in the OmegaSupreme/LeRobot environment

Set the pinned LeRobot source checkout used by the Omega environment. Both
checkouts must be committed and clean; otherwise their Git SHAs are false
provenance.

```bash
source ~/.bashrc
openpion
cd "$OPENPI_WEAVER_ROOT"

export LEROBOT_ROOT=/path/to/pinned/lerobot
export LEROBOT_SHA=be542e150aa43c97cf8a47cc518953b749ed71e5
export FRAME_SMOKE="$WEAVER_WORK_ROOT/frame-cross-task"
export OMEGA_SHA="$(git rev-parse HEAD)"

test -z "$(git status --porcelain)"
test "$(git -C "$LEROBOT_ROOT" rev-parse HEAD)" = "$LEROBOT_SHA"
test -z "$(git -C "$LEROBOT_ROOT" status --porcelain)"
test ! -e "$FRAME_SMOKE"
uv pip freeze > "$WEAVER_WORK_ROOT/omega-pip-freeze.txt"

uv run python scripts/b1k/export_weaver_frames.py \
  --dataset-root "$B1K_DATASET_ROOT" \
  --numeric-dir "$NUMERIC_SMOKE" \
  --output-dir "$FRAME_SMOKE" \
  --exporter-git-sha "$OMEGA_SHA" \
  --lerobot-git-sha "$LEROBOT_SHA" \
  --split train \
  --max-episodes 1

uv run python scripts/b1k/export_weaver_frames.py \
  --dataset-root "$B1K_DATASET_ROOT" \
  --numeric-dir "$NUMERIC_SMOKE" \
  --output-dir "$FRAME_SMOKE" \
  --exporter-git-sha "$OMEGA_SHA" \
  --lerobot-git-sha "$LEROBOT_SHA" \
  --split train \
  --max-episodes 1 \
  --verify-cache-only
```

Pass only if each requested LeRobot row reports the exact
`source_absolute_indices` value from the numeric NPZ and the episode index also
matches. The frame manifest must contain two named uint8 THWC arrays, positive
height/width, the numeric semantic and NPZ hashes, absolute-index hash,
Omega/exporter SHA, LeRobot SHA, `backend=pyav`, `return_uint8=true`, tolerance,
and the runtime PyAV version. The second command must be a verified cache hit;
a changed exporter or decoder contract is a stale cache, not a hit.

### 4B. Encode in the frozen official WEAVER environment

Return to the official environment. It must not contain the added LeRobot
dependency graph. Download only the VAE subtree used by upstream
`SD3Encoder(..., subfolder="vae")`.

```bash
cd "$UPSTREAM_WEAVER_ROOT"
source .venv/bin/activate
export PYTHONPATH="$OPENPI_WEAVER_ROOT/src:$UPSTREAM_WEAVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SD3_MODEL=stabilityai/stable-diffusion-3-medium-diffusers
SD3_REV=$(python -c \
  'from huggingface_hub import model_info; print(model_info("stabilityai/stable-diffusion-3-medium-diffusers").sha)')
SD3_SNAPSHOT="$WEAVER_WORK_ROOT/sd3/$SD3_REV"
LATENT_LOG="$WEAVER_WORK_ROOT/b1k-sd3-latent-smoke.log"

test ! -e "$NUMERIC_SMOKE/latents"
hf download "$SD3_MODEL" \
  --revision "$SD3_REV" \
  --include 'vae/*' \
  --local-dir "$SD3_SNAPSHOT"
test -f "$SD3_SNAPSHOT/vae/config.json"
printf 'SD3_REV=%s\n' "$SD3_REV" | tee "$WEAVER_WORK_ROOT/sd3-revision.env"
uv pip freeze > "$WEAVER_WORK_ROOT/weaver-pip-freeze-after-download.txt"
```

The SD3 repository may require accepting its license and authenticating with
Hugging Face first. Preflight numeric/frame linkage without importing or
constructing the official SD3 encoder:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  "$OPENPI_WEAVER_ROOT/scripts/b1k/cache_weaver_latents.py" \
  --checkpoint-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --encoder-snapshot "$SD3_SNAPSHOT" \
  --encoder-revision "$SD3_REV" \
  --adapter-git-sha "$OMEGA_SHA" \
  --numeric-dir "$NUMERIC_SMOKE" \
  --frame-dir "$FRAME_SMOKE" \
  --split train \
  --max-episodes 1 \
  --batch-size 4 \
  --device cuda \
  --resize-mode letterbox \
  --storage-dtype float16 \
  --dry-run
```

Run the same command without `--dry-run`, capturing its structured telemetry:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  "$OPENPI_WEAVER_ROOT/scripts/b1k/cache_weaver_latents.py" \
  --checkpoint-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --encoder-snapshot "$SD3_SNAPSHOT" \
  --encoder-revision "$SD3_REV" \
  --adapter-git-sha "$OMEGA_SHA" \
  --numeric-dir "$NUMERIC_SMOKE" \
  --frame-dir "$FRAME_SMOKE" \
  --split train \
  --max-episodes 1 \
  --batch-size 4 \
  --device cuda \
  --resize-mode letterbox \
  --storage-dtype float16 | tee "$LATENT_LOG"
```

On the confirmed 48 GB GPU, batch 4 is the conservative first measurement; do
not increase it before inspecting `gpu_total_bytes`, `peak_allocated_bytes`,
`peak_reserved_bytes`, and `elapsed_seconds` in the final telemetry object.
Compute remains float32; `storage_dtype=float16` only controls the saved NPZ.

Finally verify the committed cache without constructing SD3:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  "$OPENPI_WEAVER_ROOT/scripts/b1k/cache_weaver_latents.py" \
  --checkpoint-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --encoder-snapshot "$SD3_SNAPSHOT" \
  --encoder-revision "$SD3_REV" \
  --adapter-git-sha "$OMEGA_SHA" \
  --numeric-dir "$NUMERIC_SMOKE" \
  --frame-dir "$FRAME_SMOKE" \
  --split train \
  --max-episodes 1 \
  --batch-size 4 \
  --device cuda \
  --resize-mode letterbox \
  --storage-dtype float16 \
  --verify-cache-only
```

Every selected latent must be a byte-level hit. Pass also requires:

- both latent arrays are finite `(T,P,D)` with identical shapes;
- the latent manifest binds the verified frame semantic SHA and frame NPZ byte
  SHA, so changed decoder/pixels cannot reuse an old latent;
- it records distinct `compute_dtype=float32` and `storage_dtype=float16`, the
  exact combined byte hash of every file under `vae/`, checkpoint config hash,
  model revision, view order, resize contract, NPZ size, and NPZ SHA;
- no NaN/OOM occurs and the telemetry reports the expected GPU name and roughly
  48 GB total memory.

`letterbox` is the initial non-distorting square-to-rectangular policy. Inspect
first/middle/last cached RGB frames before scaling. `FRAME_SMOKE` is a deletable
intermediate bridge after latent verification; numeric and latent artifacts are
the durable inputs to `BehaviorPrecomputed`. Deleting it means this encoder
preflight/verification command must regenerate it before a future re-encode.

## 5. Smoke the WEAVER batch contract

Run this dataset-level gate back in the OmegaSupreme environment:

```bash
source ~/.bashrc
openpion
cd "$OPENPI_WEAVER_ROOT"

uv run python scripts/b1k/smoke_weaver_dataset.py \
  --dataset-dir "$NUMERIC_SMOKE" \
  --split train \
  --horizon 8 \
  --n-history 2 \
  --n-memory-frames 6 \
  --t-memory 5
```

Pass:

- each view is `(10,P,D)`, state/action are `(10,23)`, memory state is `(6,23)`;
- every numeric tensor is finite;
- task text is present;
- there is a real `transition_valid` vector;
- there is no `rewards` key in dynamics-only mode;
- no window or sparse-memory index crosses an episode boundary.

This is a `BehaviorPrecomputed` dataset/window contract. The training entrypoint
collates these mappings, moves tensors with the frozen upstream helper, and
accepts task text directly; upstream `WEAVER.encode_task` performs the CLIP
encoding for full mode. Dynamics mode does not construct or consume task/RM/
critic inputs. `transition_valid` is consumed by the dynamics flow-loss mask;
full mode requires every row to be valid because the frozen RM/critic have no
transition mask.

## 6. What this does and does not prove

After Gates 1–5 pass, the dataset path through numeric adaptation, the
deletable uint8 bridge, SD3 cache, and `BehaviorPrecomputed` is healthy. It does
not prove that the 8D→23D transferred world model predicts B1K well; that is a
separate real-model gate.

On the 48 GB GPU, construct the real 23D model, load the released checkpoint
through the audited nine-key shape transfer, and run both of these explicit
gates:

1. `--mode dynamics`: one real forward/backward step and a 32–128-window
   tiny-overfit. Reward/critic must be genuinely absent from this graph.
2. `--mode full`: one real upstream `WEAVER.forward` step with real B1K reward
   labels and task text, followed by a tiny-overfit/resume checkpoint check.

Only after held-out prediction beats persistence should reward/critic ranking
and same-candidate-pool best-of-N control be tested. Zero-filled fake rewards are
not an acceptable substitute for the full gate.

## 7. Held-out and official evaluator gates

For a transferred or resumed 23D checkpoint the checkpoint mode is mandatory;
do not rely on an identity default:

```bash
uv run python scripts/b1k/eval_weaver_b1k.py \
  --checkpoint-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --checkpoint-dir "$WEAVER_23D_CHECKPOINT" \
  --checkpoint-mode resume \
  --dataset-dir "$NUMERIC_FULL" \
  --latent-root "$NUMERIC_FULL/latents" \
  --output-dir "$WEAVER_WORK_ROOT/heldout-val" \
  --split val \
  --horizon 8 \
  --ema
```

The command fails if held-out episode indices or semantic hashes overlap the
training lineage. Its summary records both `val_steps` and effective NFE,
including pyramid stagger forward passes.

Before calling the official evaluator, create and validate a
`behavior_2026_compliance_v1` manifest and then run:

```bash
uv run python scripts/b1k/check_behavior_compliance.py \
  "$WEAVER_WORK_ROOT/compliance_manifest.json" \
  --submission

python -m omnigibson.eval.eval \
  --task-name "$TASK_ID" \
  --mode public_test \
  --instance-indices 0 1 2 3 4 5 6 7 8 9 \
  --num-rollouts 1 \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --host 127.0.0.1 \
  --port 8000 \
  --output-dir "$OFFICIAL_OUTPUT" \
  --write-video
```

Pass only if the WebSocket server returns one finite 23D controller action per
step, selector telemetry proves every decision used one immutable candidate
pool, no oracle label appears in the online artifact, and official JSON/MP4
outputs remain unmodified and byte-hashed.
