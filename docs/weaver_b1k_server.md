# WEAVER-guided B1K WebSocket server

`serve_weaver_b1k.py` is the production entry point for the SII GPU run. It
loads the released OpenPI policy, the transferred 23-D WEAVER checkpoint, the
causal action adapter, and train-only normalization statistics. Each replan
uses one frozen candidate pool and scores it with the original WEAVER WM + RM
and (for `advantage`) critic chain. The official WebSocket transport and raw
R1Pro observation keys are unchanged.

The launcher is fail-closed: the policy action horizon must cover the requested
WEAVER imagination horizon at 30 Hz → 5 Hz, and all checkpoints must be
explicitly supplied. A 32-action policy chunk covers five complete 5-Hz
intervals; it cannot be silently padded to H=8.

The downloaded WEAVER release may omit `dataset.n_states` and/or
`dataset.n_actions` when its concrete YAML was serialized. The adapter accepts
that serialization defect only for the frozen released 8-D contract, records
the inferred source field(s) in the transfer report, and still builds the
explicit 23-D BEHAVIOR target. A normal architecture audit remains strict;
arbitrary missing dimensions are never inferred.

## Server

```bash
cd "$OPENPI_ROOT"
export PYTHONPATH="$OPENPI_ROOT/src:$WEAVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TASK_ID=<official_task_name>
test -n "$TASK_ID"

uv run python scripts/b1k/serve_weaver_b1k.py \
  --robot b1k/R1Pro \
  --task "$TASK_ID" \
  --policy-config pi05_b1k_full_rgbd \
  --policy-dir /inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/2026_behavior_challenge/tiancheng/checkpoints/pi05_b1k_full_rgbd/rgbd-perf-w16-20260820-033046/40000 \
  --weaver-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --weaver-checkpoint <WEAVER_23D_CHECKPOINT_DIR> \
  --checkpoint-mode resume \
  --adapter-checkpoint <ACTION_ADAPTER_DIR>/checkpoint.pt \
  --norm-stats "$WEAVER_WORK_ROOT/numeric-full-v4/norm_stats.json" \
  --dataset-dir "$WEAVER_WORK_ROOT/numeric-full-v4" \
  --device cuda \
  --num-candidates 4 \
  --imagination-horizon 5 \
  --criterion advantage \
  --resize-mode letterbox \
  --port 8000
```

`--task` accepts either a bare official task name or `b1k/<task_name>`. The
server reads the natural-language instruction from the preparation output
`task_prompts.json` and checks its dataset hash against `summary.json`.
`--prompt` remains an explicit override; the small in-repository registry is
only a compatibility fallback for older artifacts.

The v4 preparation schema changes terminal cropping, prompt conditioning and
the action-delta convention. Do not reuse an older `numeric-full` directory;
write a new directory such as `numeric-full-v4` and recompute downstream
normalization, latent cache, adapter and WEAVER checkpoints from it.

Use `--imagination-horizon 8` only after the policy checkpoint is verified to
produce at least 48 actions per inference call. The server returns one CPU
`float32` 23-D action per evaluator step and supports the official reset
message.

## Official evaluator

Run the BEHAVIOR v3.9.1 evaluator in another shell with its normal
`RGBDFullResWrapper`, `--mode public_test`, `--num-rollouts 1`, and the official
R1Pro config. The evaluator connects to `127.0.0.1:8000`; do not alter the
camera key names or inject simulator state/reward into the server.

If the BEHAVIOR environment reports `ModuleNotFoundError: h5py`, install it in
that environment before launching OmniGibson:

```bash
behavioron
python -m pip install h5py
python -c 'import h5py; print(h5py.__version__)'
```
