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

## Server

```bash
cd "$OPENPI_ROOT"
export PYTHONPATH="$OPENPI_ROOT/src:$WEAVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

uv run python scripts/b1k/serve_weaver_b1k.py \
  --robot b1k/R1Pro \
  --task b1k/<TASK_NAME> \
  --policy-config <OPENPI_TRAINING_CONFIG> \
  --policy-dir <OPENPI_POLICY_CHECKPOINT> \
  --weaver-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --weaver-checkpoint <WEAVER_23D_CHECKPOINT_DIR> \
  --checkpoint-mode resume \
  --adapter-checkpoint <ACTION_ADAPTER_DIR>/checkpoint.pt \
  --norm-stats <NUMERIC_FULL>/norm_stats.json \
  --dataset-dir <NUMERIC_FULL> \
  --device cuda \
  --num-candidates 4 \
  --imagination-horizon 5 \
  --criterion advantage \
  --resize-mode letterbox \
  --port 8000
```

Use `--imagination-horizon 8` only after the policy checkpoint is verified to
produce at least 48 actions per inference call. The server returns one CPU
`float32` 23-D action per evaluator step and supports the official reset
message.

## Official evaluator

Run the BEHAVIOR v3.9.1 evaluator in another shell with its normal
`RGBDFullResWrapper`, `--mode public_test`, `--num-rollouts 1`, and the official
R1Pro config. The evaluator connects to `127.0.0.1:8000`; do not alter the
camera key names or inject simulator state/reward into the server.
