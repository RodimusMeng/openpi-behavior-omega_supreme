# WEAVER-guided B1K WebSocket server

`serve_weaver_b1k.py` is the production entry point for the SII GPU run. It
loads the pinned BEHAVIOR OpenPI policy, the transferred 23-D WEAVER checkpoint,
the audited hybrid causal-action contract, and train-only normalization statistics. Each replan
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
cd "$OPENPI_WEAVER_ROOT"
export PYTHONPATH="$OPENPI_WEAVER_ROOT:$OPENPI_WEAVER_ROOT/src:$WEAVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TASK_ID=<official_task_name>
test -n "$TASK_ID"

python scripts/b1k/serve_weaver_b1k.py \
  --robot b1k/R1Pro \
  --task "$TASK_ID" \
  --policy-config pi05_b1k_full_abs \
  --policy-dir "$B1K_POLICY_CHECKPOINT" \
  --weaver-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --weaver-checkpoint <WEAVER_23D_CHECKPOINT_DIR> \
  --checkpoint-mode resume \
  --task-encoder-model "$WEAVER_WORK_ROOT/task-encoder/<PINNED_REVISION>" \
  --hybrid-action-contract "$WEAVER_WORK_ROOT/audits/action-semantics-40fit-10held-v3.json" \
  --norm-stats "$WEAVER_WORK_ROOT/numeric-full-v6/norm_stats_hybrid_v1.json" \
  --dataset-dir "$WEAVER_WORK_ROOT/numeric-full-v6" \
  --device cuda:0 \
  --num-candidates 4 \
  --imagination-horizon 5 \
  --criterion advantage \
  --resize-mode stretch \
  --port 8000
```

`--task` accepts either a bare official task name or `b1k/<task_name>`. The
server reads the natural-language instruction from the preparation output
`task_prompts.json` and checks its dataset hash against `summary.json`.
`--prompt` remains an explicit override; the small in-repository registry is
only a compatibility fallback for older artifacts.

The v6 preparation schema changes terminal-label handling, prompt conditioning and
the action-delta convention. Do not reuse an older `numeric-full` directory;
write a new directory such as `numeric-full-v6` and recompute downstream
hybrid normalization, latent cache, and WEAVER checkpoints from it.

Use `--imagination-horizon 8` only after the policy checkpoint is verified to
produce at least 48 actions per inference call. The server returns one CPU
`float32` 23-D action per evaluator step and supports the official reset
message.

## Official evaluator

Run the BEHAVIOR v3.9.1 evaluator in another shell with
`RGBDFullResWrapper`, `--mode public_test`, `--num-rollouts 1`, and the official
R1Pro config. This wrapper is retained for its data-collection-resolution RGB
observations. Although it also exposes `depth_linear`, the active OpenPI and
WEAVER paths consume RGB and proprioception only; depth is not forwarded to
either model. The evaluator connects to `127.0.0.1:8000`; do not alter the
camera key names or inject simulator state/reward into the server.

If the BEHAVIOR environment reports `ModuleNotFoundError: h5py`, install it in
that environment before launching OmniGibson:

```bash
behavioron
python -m pip install h5py
python -c 'import h5py; print(h5py.__version__)'
```

## Simulator-free end-to-end replay

When OmniGibson cannot start, `replay_weaver_pipeline.py` exercises the same
`OnlineWeaverPolicy` object directly from exact LeRobot rows. It loads all
three RGB views required by the absolute-action OpenPI policy, while WEAVER
continues to consume its released two-view mapping. Recorded future state and
reward never enter candidate selection.

The example below leaves both physical GPUs visible. JAX/OpenPI takes logical
GPU 0 by default and WEAVER/Torch is placed explicitly on logical GPU 1. This
is safe because no Omniverse Vulkan runtime is involved.

```bash
openpion
cd "$OPENPI_WEAVER_ROOT"
export PYTHONPATH="$OPENPI_WEAVER_ROOT:$OPENPI_WEAVER_ROOT/src:$WEAVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

python scripts/b1k/replay_weaver_pipeline.py \
  --dataset-root /inspire/dataset/behavior-1k-2026-challenge-demos/v3again \
  --numeric-dir "$WEAVER_WORK_ROOT/numeric-full-v6" \
  --output-json "$WEAVER_WORK_ROOT/audits/simulator-free-replay-step20000.json" \
  --policy-config pi05_b1k_full_abs \
  --policy-dir "$B1K_POLICY_CHECKPOINT/20000" \
  --weaver-config "$WEAVER_RELEASE/WEAVER/config.yaml" \
  --weaver-checkpoint "$WEAVER_WORK_ROOT/train-smoke-hybrid-4ep-v1" \
  --checkpoint-mode resume \
  --task-encoder-model "$WEAVER_WORK_ROOT/task-encoder/32bd64288804d66eefd0ccbe215aa642df71cc41" \
  --hybrid-action-contract "$WEAVER_WORK_ROOT/audits/action-semantics-40fit-10held-v3.json" \
  --norm-stats "$WEAVER_WORK_ROOT/numeric-full-v6/norm_stats_hybrid_v1.json" \
  --num-tasks 3 \
  --exclude-tasks turn_on_radio \
  --seed 20260903 \
  --num-candidates 2 \
  --imagination-horizon 5 \
  --criterion advantage \
  --resize-mode stretch \
  --device cuda:1
```

Omit `--tasks` to sample three task names deterministically from the selected
numeric split. To pin the resulting task set on a later run, pass for example
`--tasks polishing_shoes folding_towels putting_away_dishes`. The JSON report
records the exact episode, absolute LeRobot row, numeric semantic hash,
candidate-pool digest, hybrid representation, rollout shapes/spread, scores,
selected controller action, timings, and Torch memory telemetry.

A passing replay proves plumbing and provenance only. The current one-step
hybrid smoke checkpoint has no decision-usefulness claim, and offline replay
cannot provide counterfactual simulator outcomes for unexecuted candidates.
