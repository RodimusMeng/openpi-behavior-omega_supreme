# WEAVER B1K: candidate stability and 40/10 transfer gate

This runbook deliberately separates three questions:

1. Is candidate ranking reproducible under controlled rollout noise?
2. Is the frozen `0.10` binary-gripper tolerance supported by a broader policy sample?
3. Does hybrid-action finetuning improve state and latent prediction on exact, unseen episodes?

It does not require OmniGibson. It also does not claim task success or selector usefulness in the simulator.

## 0. Common paths

Run from the `tc-b1k-weaver` checkout after committing and syncing the code. Start in the OpenPI environment; it is the combined JAX policy plus Torch WEAVER runtime used by the successful simulator-free replay.

```bash
openpion
export OPENPI_WEAVER_ROOT=/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/2026_behavior_challenge/tiancheng/openpi-weaver
export WEAVER_ROOT=/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/2026_behavior_challenge/tiancheng/WEAVER
export WEAVER_WORK_ROOT=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/2026_behavior_challenge/tiancheng/weaver-work
export DATASET_ROOT=/inspire/dataset/behavior-1k-2026-challenge-demos/v3again
export NUMERIC_ROOT="$WEAVER_WORK_ROOT/numeric-full-v6"
export WEAVER_RELEASE=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/2026_behavior_challenge/tiancheng/models/weaver-release
export TASK_ENCODER="$WEAVER_WORK_ROOT/task-encoder/32bd64288804d66eefd0ccbe215aa642df71cc41"
export SD3_REV=ea42f8cef0f178587cf766dc8129abd379c90671
export SD3_SNAPSHOT="$WEAVER_WORK_ROOT/sd3/$SD3_REV"
export HYBRID_CONTRACT="$WEAVER_WORK_ROOT/audits/action-semantics-40fit-10held-v3.json"
export HYBRID_NORM="$NUMERIC_ROOT/norm_stats_hybrid_v1.json"
export BEHAVIOR_CKPT=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/2026_behavior_challenge/tiancheng/checkpoints/pi05_b1k_full_abs/broad-v1-full-abs-b128-fsdp8-lr2p5e6-100k-20260831-034635/20000
export WEAVER_CHECKPOINT="$WEAVER_WORK_ROOT/train-smoke-hybrid-4ep-v1"
export WEAVER_CONFIG="$WEAVER_CHECKPOINT/config.yaml"

cd "$OPENPI_WEAVER_ROOT"
export PYTHONPATH="$OPENPI_WEAVER_ROOT:$OPENPI_WEAVER_ROOT/src:$WEAVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

test -f "$NUMERIC_ROOT/manifest.jsonl"
test -f "$TASK_ENCODER/config.json"
test -f "$SD3_SNAPSHOT/vae/config.json"
test -f "$HYBRID_CONTRACT"
test -f "$HYBRID_NORM"
test -d "$BEHAVIOR_CKPT/params"
test -f "$WEAVER_CHECKPOINT/checkpoint.pt"
```

## 1. CPU regression gate

Use the Python 3.11 combined runtime. Always invoke pytest through that Python, not the system `pytest` launcher.

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  scripts/b1k/select_weaver_transfer_cohort_test.py \
  scripts/b1k/audit_weaver_candidate_stability_test.py \
  scripts/b1k/compare_weaver_prediction_summaries_test.py \
  scripts/b1k/eval_weaver_b1k_test.py \
  src/openpi/weaver/prediction_eval_test.py \
  src/openpi/weaver/behavior_precomputed_test.py \
  src/openpi/weaver/steering_runtime_test.py \
  scripts/b1k/train_weaver_b1k_test.py
```

## 2. Candidate/noise attribution and gripper audit

The policy pool is sampled once per observation. WEAVER then scores:

- the same batched pool twice with the same seed;
- every candidate separately under identical noise seeds;
- the same candidates in reverse order under the first seed.

This makes action-induced differences paired against common rollout noise. It also samples both raw policy gripper dimensions and reports distance to the nearest `-1` or `+1` command.

```bash
export STABILITY_JSON="$WEAVER_WORK_ROOT/audits/candidate-stability-step20000-$(date -u +%Y%m%d-%H%M%S).json"

CUDA_VISIBLE_DEVICES=0 python scripts/b1k/audit_weaver_candidate_stability.py \
  --dataset-root "$DATASET_ROOT" \
  --numeric-dir "$NUMERIC_ROOT" \
  --output-json "$STABILITY_JSON" \
  --policy-config pi05_b1k_full_abs \
  --policy-dir "$BEHAVIOR_CKPT" \
  --weaver-config "$WEAVER_CONFIG" \
  --weaver-checkpoint "$WEAVER_CHECKPOINT" \
  --task-encoder-model "$TASK_ENCODER" \
  --hybrid-action-contract "$HYBRID_CONTRACT" \
  --norm-stats "$HYBRID_NORM" \
  --num-tasks 30 \
  --samples-per-episode 5 \
  --num-candidates 4 \
  --scored-observations 1 \
  --rollout-seeds 101 202 303 404 505 \
  --device cuda:0 \
  --gripper-tolerance 0.10
```

The technical gate passes only when same-seed replay and order reversal are exact and no sampled gripper command exceeds the declared tolerance. Inspect `gripper` for P95, P99, maximum, and the fractions above `0.05` and `0.10`. Inspect each scored record's `stability.common_noise` for action-gap versus rollout-noise scale. This remains a simulator-free diagnostic.

## 3. Freeze a task-matched 40/10 cohort

This selects ten tasks, four train episodes per task, and one val episode of the same task. Selection is deterministic, episode-disjoint, semantically hashed, and excludes `turn_on_radio`.

```bash
export COHORT_JSON="$WEAVER_WORK_ROOT/audits/weaver-transfer-40train-10val-seed20260903.json"

python scripts/b1k/select_weaver_transfer_cohort.py \
  --dataset-dir "$NUMERIC_ROOT" \
  --output-json "$COHORT_JSON" \
  --num-tasks 10 \
  --train-per-task 4 \
  --held-out-per-task 1 \
  --held-out-split val \
  --seed 20260903

read -r -a TRAIN_EPISODES <<< "$(python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["train_episode_indices"])' "$COHORT_JSON")"
read -r -a HELDOUT_EPISODES <<< "$(python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["held_out_episode_indices"])' "$COHORT_JSON")"

printf 'train=%s\n' "${TRAIN_EPISODES[*]}"
printf 'held_out=%s\n' "${HELDOUT_EPISODES[*]}"
```

Keep this JSON unchanged for every subsequent run. The training and evaluation scripts independently reject missing IDs, duplicates, and train/evaluation lineage overlap.

## 4. Export the exact 50 two-camera frame sequences

Activate the OpenPI environment because this stage reads LeRobot/BEHAVIOR source data.

```bash
openpion
cd "$OPENPI_WEAVER_ROOT"
export PYTHONPATH="$OPENPI_WEAVER_ROOT:$OPENPI_WEAVER_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export FRAME_ROOT="$WEAVER_WORK_ROOT/frame-transfer-40train-10val-v1"
export ADAPTER_SHA="$(git rev-parse HEAD)"
export LEROBOT_SHA=be542e150aa43c97cf8a47cc518953b749ed71e5

python scripts/b1k/export_weaver_frames.py \
  --dataset-root "$DATASET_ROOT" \
  --numeric-dir "$NUMERIC_ROOT" \
  --output-dir "$FRAME_ROOT" \
  --split train \
  --episode-indices "${TRAIN_EPISODES[@]}" \
  --exporter-git-sha "$ADAPTER_SHA" \
  --lerobot-git-sha "$LEROBOT_SHA"

python scripts/b1k/export_weaver_frames.py \
  --dataset-root "$DATASET_ROOT" \
  --numeric-dir "$NUMERIC_ROOT" \
  --output-dir "$FRAME_ROOT" \
  --split val \
  --episode-indices "${HELDOUT_EPISODES[@]}" \
  --exporter-git-sha "$ADAPTER_SHA" \
  --lerobot-git-sha "$LEROBOT_SHA"
```

## 5. Encode train and val latents on the two GPUs

Activate WEAVER, return to the adapter checkout, and run one split per GPU. These processes write different split directories.

```bash
weaveron
cd "$OPENPI_WEAVER_ROOT"
export PYTHONPATH="$OPENPI_WEAVER_ROOT:$OPENPI_WEAVER_ROOT/src:$WEAVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python scripts/b1k/cache_weaver_latents.py \
  --checkpoint-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --encoder-snapshot "$SD3_SNAPSHOT" \
  --encoder-revision "$SD3_REV" \
  --adapter-git-sha "$ADAPTER_SHA" \
  --numeric-dir "$NUMERIC_ROOT" \
  --frame-dir "$FRAME_ROOT" \
  --split train \
  --episode-indices "${TRAIN_EPISODES[@]}" \
  --batch-size 4 \
  --device cuda:0 \
  --precompute-spatial-size 4 \
  --resize-mode stretch \
  --storage-dtype float16 \
  >"$WEAVER_WORK_ROOT/latent-40train.log" 2>&1 &
PID_TRAIN=$!

python scripts/b1k/cache_weaver_latents.py \
  --checkpoint-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --encoder-snapshot "$SD3_SNAPSHOT" \
  --encoder-revision "$SD3_REV" \
  --adapter-git-sha "$ADAPTER_SHA" \
  --numeric-dir "$NUMERIC_ROOT" \
  --frame-dir "$FRAME_ROOT" \
  --split val \
  --episode-indices "${HELDOUT_EPISODES[@]}" \
  --batch-size 4 \
  --device cuda:1 \
  --precompute-spatial-size 4 \
  --resize-mode stretch \
  --storage-dtype float16 \
  >"$WEAVER_WORK_ROOT/latent-10val.log" 2>&1 &
PID_VAL=$!

wait "$PID_TRAIN"; STATUS_TRAIN=$?
wait "$PID_VAL"; STATUS_VAL=$?
printf 'train_status=%s val_status=%s\n' "$STATUS_TRAIN" "$STATUS_VAL"
test "$STATUS_TRAIN" -eq 0
test "$STATUS_VAL" -eq 0
```

Verify every selected artifact semantically, not only by filename:

```bash
python scripts/b1k/cache_weaver_latents.py \
  --checkpoint-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --encoder-snapshot "$SD3_SNAPSHOT" \
  --encoder-revision "$SD3_REV" \
  --adapter-git-sha "$ADAPTER_SHA" \
  --numeric-dir "$NUMERIC_ROOT" \
  --frame-dir "$FRAME_ROOT" \
  --split train \
  --episode-indices "${TRAIN_EPISODES[@]}" \
  --batch-size 4 \
  --device cuda:0 \
  --precompute-spatial-size 4 \
  --resize-mode stretch \
  --storage-dtype float16 \
  --verify-cache-only

python scripts/b1k/cache_weaver_latents.py \
  --checkpoint-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --encoder-snapshot "$SD3_SNAPSHOT" \
  --encoder-revision "$SD3_REV" \
  --adapter-git-sha "$ADAPTER_SHA" \
  --numeric-dir "$NUMERIC_ROOT" \
  --frame-dir "$FRAME_ROOT" \
  --split val \
  --episode-indices "${HELDOUT_EPISODES[@]}" \
  --batch-size 4 \
  --device cuda:1 \
  --precompute-spatial-size 4 \
  --resize-mode stretch \
  --storage-dtype float16 \
  --verify-cache-only
```

## 6. Dynamics-only transfer on one 4090

The current trainer is single-process. Use the second GPU for independent work rather than pretending this is a two-GPU distributed trainer. The following is a bounded 500-step transfer run with effective batch size four via accumulation.

```bash
export TRAIN_OUT="$WEAVER_WORK_ROOT/train-hybrid-40ep-500step-v1"

CUDA_VISIBLE_DEVICES=0 python scripts/b1k/train_weaver_b1k.py \
  --checkpoint-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --pretrained-dir "$WEAVER_RELEASE/WEAVER" \
  --task-encoder-model "$TASK_ENCODER" \
  --output-dir "$TRAIN_OUT" \
  --dataset-dir "$NUMERIC_ROOT" \
  --norm-stats-path "$HYBRID_NORM" \
  --hybrid-action-contract "$HYBRID_CONTRACT" \
  --mode dynamics \
  --split train \
  --episode-indices "${TRAIN_EPISODES[@]}" \
  --device cuda:0 \
  --seed 20260903 \
  --override training.batch_size=1 \
  --override training.gradient_accumulation_steps=4 \
  --override training.max_steps=500 \
  --override training.warmup_steps=50 \
  --override training.max_lr=1e-5 \
  --override training.min_lr=1e-6 \
  --override dataset.n_workers=0 \
  --override ckpt_freq=100 \
  2>&1 | tee "$WEAVER_WORK_ROOT/train-hybrid-40ep-500step-v1.log"
```

Do not change overrides when resuming this directory; the resume contract intentionally rejects changed model, data, norm, hybrid, or optimizer semantics.

## 7. Exact held-out fresh-versus-finetuned rollout

Run identical val windows on separate GPUs. These are generated future trajectories, not training flow losses. Latent MSE/RMSE is always reported; RGB decode and LPIPS remain optional and expensive.

```bash
export EVAL_ROOT="$WEAVER_WORK_ROOT/eval-hybrid-40train-10val-v1"
mkdir -p "$EVAL_ROOT"

CUDA_VISIBLE_DEVICES=0 python scripts/b1k/eval_weaver_b1k.py \
  --checkpoint-config "$TRAIN_OUT/config.yaml" \
  --checkpoint-dir "$WEAVER_RELEASE/WEAVER" \
  --checkpoint-mode b1k_23d_fresh \
  --task-encoder-model "$TASK_ENCODER" \
  --dataset-dir "$NUMERIC_ROOT" \
  --norm-stats-path "$HYBRID_NORM" \
  --hybrid-action-contract "$HYBRID_CONTRACT" \
  --split val \
  --episode-indices "${HELDOUT_EPISODES[@]}" \
  --output-dir "$EVAL_ROOT/fresh" \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 0 \
  --window-stride 10 \
  --max-windows 100 \
  --horizon 1 --horizon 3 --horizon 5 \
  --nfe 10 \
  --seed 20260903 \
  >"$EVAL_ROOT/fresh.log" 2>&1 &
PID_FRESH=$!

CUDA_VISIBLE_DEVICES=1 python scripts/b1k/eval_weaver_b1k.py \
  --checkpoint-config "$TRAIN_OUT/config.yaml" \
  --checkpoint-dir "$TRAIN_OUT" \
  --checkpoint-mode resume \
  --task-encoder-model "$TASK_ENCODER" \
  --dataset-dir "$NUMERIC_ROOT" \
  --norm-stats-path "$HYBRID_NORM" \
  --hybrid-action-contract "$HYBRID_CONTRACT" \
  --split val \
  --episode-indices "${HELDOUT_EPISODES[@]}" \
  --output-dir "$EVAL_ROOT/trained" \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 0 \
  --window-stride 10 \
  --max-windows 100 \
  --horizon 1 --horizon 3 --horizon 5 \
  --nfe 10 \
  --seed 20260903 \
  >"$EVAL_ROOT/trained.log" 2>&1 &
PID_TRAINED=$!

wait "$PID_FRESH"; STATUS_FRESH=$?
wait "$PID_TRAINED"; STATUS_TRAINED=$?
printf 'fresh_status=%s trained_status=%s\n' "$STATUS_FRESH" "$STATUS_TRAINED"
test "$STATUS_FRESH" -eq 0
test "$STATUS_TRAINED" -eq 0

python scripts/b1k/compare_weaver_prediction_summaries.py \
  --fresh-summary "$EVAL_ROOT/fresh/summary.json" \
  --trained-summary "$EVAL_ROOT/trained/summary.json" \
  --output-json "$EVAL_ROOT/comparison.json"
```

With `CUDA_VISIBLE_DEVICES=1`, that process sees the physical GPU 1 as `cuda:0`; this is why both evaluation commands use `--device cuda:0`.

## Interpretation

- Candidate gate: action gaps must be materially larger than seed-induced score variation; exact replay/order checks must pass.
- Gripper gate: decide the tolerance from P95/P99/max and outlier fractions, not one previously observed value.
- Transfer gate: compare each horizon separately. Require state RMSE improvement over fresh, and check whether the trained model also beats the persistence baseline. Require both camera latent errors to improve, especially at short horizons.
- A 500-step result is a transfer/generalization gate, not a final training recipe.
- None of these simulator-free metrics establishes closed-loop success. Same-state OmniGibson counterfactual ranking remains the next external gate when a compatible simulator host exists.
