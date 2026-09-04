# B1K input-pipeline reform experiments

This branch is based on `tc-b1k-baseline@e969d09`. Keep the original
`pi05_b1k_full_abs` run as the control; enable block shuffling and loader
microbatches through CLI overrides until the server measurements select final
values.

## 1. Audit the timestamp that killed the run

```bash
DATA=/inspire/dataset/behavior-1k-2026-challenge-demos/v3again
PY=.venv/bin/python

CUDA_VISIBLE_DEVICES="" "$PY" scripts/local/audit_b1k_video_timestamps.py \
  --dataset-root "$DATA" \
  --video-key observation.rgb.left_realsense_link_camera_0 \
  --video-path-contains chunk-023/file-004.mp4 \
  --output-json /tmp/b1k_timestamp_chunk023_file004.json
```

Then sample a wider, fixed episode set. This command deliberately requires a
sample limit for `--all-episodes`; it never skips bad rows or changes training
data.

```bash
for CAM in \
  observation.rgb.left_realsense_link_camera_0 \
  observation.rgb.right_realsense_link_camera_0 \
  observation.rgb.zed_link_camera_0; do
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/local/audit_b1k_video_timestamps.py \
    --dataset-root "$DATA" \
    --video-key "$CAM" \
    --episode-count 128 \
    --samples 50000 \
    --seed 42 \
    --output-json "/tmp/b1k_timestamp_${CAM}.json"
done
```

The report contains both the old float32 guard result and a float64 reference.
At timestamps above 8192 seconds one float32 ULP is about 0.977 ms, already
larger than the old 0.5 ms tolerance. Use the float64 distribution to judge
actual synchronization quality. The immediate baseline candidate is 2 ms; do
not widen it further without evidence. LeRobot's guard is strict (`error <
tolerance`), so an error exactly equal to the configured tolerance is counted
as a violation.

## 2. Measure locality with the same samples

The benchmark first chooses complete 16-frame comparison blocks, then changes
only their subdivision and order. Every run below must print the same
`sample_set_sha256`; `sequence_sha256` should differ.

Run this only with the bounded-cache LeRobot change installed. The default
capacity is 8 decoders per process and can be changed with
`--torchcodec-decoder-cache-size`.

```bash
mkdir -p /tmp/b1k_block_ab
for K in 1 2 4 8 16; do
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/local/benchmark_b1k_reader_memory.py \
    --dataset-root "$DATA" \
    --episode-count 128 \
    --mode rgb \
    --video-backend torchcodec \
    --return-uint8 \
    --pattern block-random \
    --block-size "$K" \
    --comparison-block-size 16 \
    --samples 4096 \
    --tail-window 1024 \
    --report-every 256 \
    --seed 42 \
    --output-jsonl "/tmp/b1k_block_ab/k${K}.jsonl"
done
```

Record overall and tail throughput, p50/p95/p99/max sample latency, RSS slope,
anonymous-RSS slope, file descriptors, and both hashes. A 128-sample run is not
long enough to establish memory stability.

## 3. Full DataLoader smoke matrix

After selecting `K`, keep optimizer batch 128 and vary only worker job size and
delivery ordering:

```text
control: K unset, loader microbatch unset, in_order=true
A:       K=8,     loader microbatch=16,    in_order=true
B:       K=8,     loader microbatch=16,    in_order=false
```

The corresponding training overrides are:

```bash
--b1k-block-size 8 \
--loader-microbatch-size 16 \
--no-loader-in-order \
--torchcodec-decoder-cache-size 8
```

`loader_microbatch_size` is only the PyTorch worker job size. OpenPI concatenates
eight 16-sample jobs before JAX sharding, so the optimizer still receives batch
128.

Benchmark the complete path without initializing model weights:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/local/benchmark_b1k_data_loader.py \
  --config-name pi05_b1k_full_abs \
  --video-backend torchcodec \
  --batches 64 \
  --warmup-batches 8 \
  --num-workers 32 \
  --prefetch-factor 2 \
  --block-size 8 \
  --loader-microbatch-size 16 \
  --no-in-order \
  --torchcodec-decoder-cache-size 8 \
  --output-json /tmp/b1k_full_loader_k8_mb16_unordered.json
```

This reports the post-transform, post-collate, post-JAX-sharding throughput. It
does not include model compute, so compare it with the measured model-side upper
bound rather than with end-to-end training throughput directly.

## 4. Four-GPU restore-only smoke

First validate 8-to-4 resharding without taking an optimizer step or writing a
new checkpoint. Use the existing experiment name and set `num_train_steps` equal
to the restored step:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  uv run scripts/train.py pi05_b1k_full_abs \
  --exp-name YOUR_EXISTING_EXPERIMENT \
  --resume \
  --fsdp-devices 4 \
  --num-train-steps 20000
```

Only after the log confirms `Restored train state at step 20000` should a new
experiment or resume run execute training steps.

## Checkpoint capacity warning

`save_interval=2000` limits compute loss, but Orbax may temporarily need space
for both the retained checkpoint and the new one. Verify quota/free space for at
least two approximately 42 GiB checkpoints before resuming. `keep_period=20000`
still preserves milestone checkpoints.
