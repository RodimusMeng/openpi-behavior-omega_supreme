#!/usr/bin/env bash
set -Eeuo pipefail

# Pure OpenPI RGBD checkpoint evaluation on the official BEHAVIOR-1K evaluator.
# Run it from the BEHAVIOR environment after `behavioron` and after exporting
# OPENPI_ROOT, B1K_ROOT, OPENPI_CONFIG, OPENPI_POLICY_CHECKPOINT, and EVAL_ROOT.

: "${OPENPI_ROOT:?export OPENPI_ROOT=/path/to/openpi}"
: "${B1K_ROOT:?export B1K_ROOT=/path/to/BEHAVIOR-1K}"
: "${OPENPI_CONFIG:?export OPENPI_CONFIG=pi05_b1k_full_rgbd}"
: "${OPENPI_POLICY_CHECKPOINT:?export OPENPI_POLICY_CHECKPOINT=/path/to/checkpoint}"
: "${EVAL_ROOT:?export EVAL_ROOT=/path/to/eval-output}"

TASKS_FILE="${BEHAVIOR_TASKS_FILE:-${B1K_DATASET_ROOT:-}/meta/tasks.jsonl}"
HOST="${POLICY_HOST:-127.0.0.1}"
PORT="${POLICY_PORT:-8000}"
ROBOT="${BEHAVIOR_ROBOT:-b1k/R1Pro}"
NORM_STATS_REPO_ID="${OPENPI_NORM_STATS_REPO_ID:-b1k-v3again}"
INSTANCE_INDICES_TEXT="${BEHAVIOR_INSTANCE_INDICES:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
ROLLOUTS="${BEHAVIOR_NUM_ROLLOUTS:-1}"
WRITE_VIDEO="${BEHAVIOR_WRITE_VIDEO:-0}"
RUN_ID="${BEHAVIOR_EVAL_RUN_ID:-rgbd_full_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$EVAL_ROOT/$RUN_ID"
SERVER_PID=""

die() { echo "[rgbd-eval] ERROR: $*" >&2; exit 2; }
[[ "$OPENPI_CONFIG" == *rgbd* ]] || die "OPENPI_CONFIG must be an RGBD config, got $OPENPI_CONFIG"
[[ -d "$OPENPI_ROOT" ]] || die "OPENPI_ROOT does not exist: $OPENPI_ROOT"
[[ -d "$B1K_ROOT/OmniGibson" ]] || die "B1K_ROOT/OmniGibson not found: $B1K_ROOT/OmniGibson"
[[ -f "$OPENPI_POLICY_CHECKPOINT/model.safetensors" || -d "$OPENPI_POLICY_CHECKPOINT/params" ]] || \
    die "checkpoint is not a supported OpenPI checkpoint directory: $OPENPI_POLICY_CHECKPOINT"
[[ -f "$TASKS_FILE" ]] || die "official task metadata not found: $TASKS_FILE"
[[ "$ROLLOUTS" =~ ^[1-9][0-9]*$ ]] || die "BEHAVIOR_NUM_ROLLOUTS must be a positive integer"
[[ "$PORT" =~ ^[1-9][0-9]*$ ]] || die "POLICY_PORT must be a positive integer"
[[ "$WRITE_VIDEO" == 0 || "$WRITE_VIDEO" == 1 ]] || die "BEHAVIOR_WRITE_VIDEO must be 0 or 1"

mkdir -p "$RUN_ROOT"
TASK_TABLE="$RUN_ROOT/task_table.tsv"
DRIVER_LOG="$RUN_ROOT/driver.log"

# Produce the authoritative task list. No OpenPI registry fallback is used:
# the local registry is only a convenience entry, not the official catalog.
python - "$TASKS_FILE" "${BEHAVIOR_EVAL_TASKS:-}" > "$TASK_TABLE" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
requested = {x.strip() for x in sys.argv[2].split(',') if x.strip()}
rows = []
for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    name = str(row.get('task_name', '')).strip()
    prompt = str(row.get('task', row.get('language_instruction', row.get('prompt', '')))).strip()
    if not name or not prompt:
        raise SystemExit(f'{path}:{line_no}: expected task_name and task/instruction fields')
    if not requested or name in requested:
        rows.append((name, prompt))
if requested and {n for n, _ in rows} != requested:
    missing = sorted(requested - {n for n, _ in rows})
    raise SystemExit(f'requested tasks are absent from official metadata: {missing}')
if not rows:
    raise SystemExit('no tasks selected')
seen = set()
for name, prompt in rows:
    if name in seen:
        raise SystemExit(f'duplicate task_name in official metadata: {name}')
    seen.add(name)
    print(name.replace('\t', ' '), prompt.replace('\t', ' '), sep='\t')
PY

read -r -a INSTANCE_INDICES <<< "$INSTANCE_INDICES_TEXT"
[[ "${#INSTANCE_INDICES[@]}" -gt 0 ]] || die "BEHAVIOR_INSTANCE_INDICES selected no instances"
for idx in "${INSTANCE_INDICES[@]}"; do [[ "$idx" =~ ^[0-9]+$ ]] || die "invalid instance index: $idx"; done

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

port_open() {
  python - "$HOST" "$PORT" <<'PY'
import socket, sys
with socket.socket() as s:
    s.settimeout(0.25)
    raise SystemExit(0 if s.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0 else 1)
PY
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

start_server() {
  local task="$1" prompt="$2" task_root="$3"
  port_open && die "policy port $HOST:$PORT is already occupied"
  (
    cd "$OPENPI_ROOT"
    export PYTHONPATH="$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    exec uv run python scripts/b1k/serve_b1k.py \
      --robot "$ROBOT" \
      --task "b1k/$task" \
      --prompt "$prompt" \
      --repo-id "$NORM_STATS_REPO_ID" \
      --policy.config "$OPENPI_CONFIG" \
      --policy.dir "$OPENPI_POLICY_CHECKPOINT" \
      --control-mode "${POLICY_CONTROL_MODE:-receding_horizon}" \
      --action-horizon "${POLICY_ACTION_HORIZON:-16}" \
      --port "$PORT"
  ) > "$task_root/policy_server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 600); do
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    port_open && return 0
    sleep 1
  done
  return 1
}

printf '[rgbd-eval] run_root=%s\n' "$RUN_ROOT" | tee "$DRIVER_LOG"
printf '[rgbd-eval] tasks_file=%s config=%s checkpoint=%s\n' "$TASKS_FILE" "$OPENPI_CONFIG" "$OPENPI_POLICY_CHECKPOINT" | tee -a "$DRIVER_LOG"
printf '[rgbd-eval] instances=%s rollouts=%s write_video=%s\n' "$INSTANCE_INDICES_TEXT" "$ROLLOUTS" "$WRITE_VIDEO" | tee -a "$DRIVER_LOG"

while IFS=$'\t' read -r task prompt; do
  task_root="$RUN_ROOT/$task"
  mkdir -p "$task_root"
  stop_server
  printf '[rgbd-eval] start task=%s\n' "$task" | tee -a "$DRIVER_LOG"
  if ! start_server "$task" "$prompt" "$task_root"; then
    printf '{"task":"%s","status":"policy_server_failed"}\n' "$task" >> "$RUN_ROOT/status.jsonl"
    printf '[rgbd-eval] policy server failed task=%s; continuing\n' "$task" | tee -a "$DRIVER_LOG"
    stop_server
    continue
  fi

  eval_args=(
    --task-name "$task"
    --robot-config "$B1K_ROOT/OmniGibson/omnigibson/eval/r1pro.yaml"
    --host "$HOST" --port "$PORT"
    --mode public_test
    --instance-indices "${INSTANCE_INDICES[@]}"
    --num-rollouts "$ROLLOUTS"
    --policy websocket
    --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper
    --output-dir "$task_root"
    --headless
  )
  [[ "$WRITE_VIDEO" == 1 ]] && eval_args+=(--write-video) || eval_args+=(--no-write-video)
  set +e
  (cd "$B1K_ROOT/OmniGibson" && python -m omnigibson.eval.eval "${eval_args[@]}" 2>&1 | tee "$task_root/evaluator.log")
  status=${PIPESTATUS[0]}
  set -e
  printf '{"task":"%s","status":"evaluator_exit_%s"}\n' "$task" "$status" >> "$RUN_ROOT/status.jsonl"
  printf '[rgbd-eval] task=%s evaluator_status=%s\n' "$task" "$status" | tee -a "$DRIVER_LOG"
  stop_server
done < "$TASK_TABLE"

python - "$RUN_ROOT" "$TASK_TABLE" "$INSTANCE_INDICES_TEXT" "$ROLLOUTS" "$WRITE_VIDEO" \
  "$OPENPI_CONFIG" "$OPENPI_POLICY_CHECKPOINT" "$NORM_STATS_REPO_ID" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
tasks = [line.split('\t', 1)[0] for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
indices = [int(x) for x in sys.argv[3].split()]
rollouts = int(sys.argv[4])
rows, invalid = [], []
for task in tasks:
    for path in sorted((root / task / 'json').glob('*.json')):
        try:
            row = json.loads(path.read_text())
            if row.get('task') != task or 'success' not in row:
                raise ValueError('missing task/success')
            rows.append(row)
        except Exception as exc:
            invalid.append({'path': str(path), 'error': str(exc)})
successes = sum(bool(r['success']) for r in rows)
q = [float(r.get('q_score', {}).get('final', 0.0)) for r in rows if isinstance(r.get('q_score'), dict)]
by_task = {}
for r in rows:
    by_task.setdefault(r['task'], []).append(r)
per_task = {}
for task in tasks:
    rr = by_task.get(task, [])
    per_task[task] = {
        'episodes': len(rr),
        'successes': sum(bool(r['success']) for r in rr),
        'success_rate': (sum(bool(r['success']) for r in rr) / len(rr)) if rr else None,
        'mean_q_score_final': (sum(float(r.get('q_score', {}).get('final', 0.0)) for r in rr) / len(rr)) if rr else None,
    }
summary = {
    'schema': 'behavior_rgbd_evaluation_v1',
    'policy': {'config': sys.argv[6], 'checkpoint': sys.argv[7], 'norm_stats_asset_id': sys.argv[8]},
    'evaluator': {'mode': 'public_test', 'instance_indices': indices, 'rollouts_per_instance': rollouts,
                  'policy': 'websocket', 'env_wrapper': 'omnigibson.eval.wrappers.RGBDFullResWrapper',
                  'official_default_max_steps': True, 'write_video': sys.argv[5] == '1'},
    'coverage': {'tasks_requested': len(tasks), 'episodes_expected': len(tasks)*len(indices)*rollouts,
                 'episodes_observed': len(rows), 'invalid_result_files': invalid},
    'aggregate': {'successes': successes, 'episodes': len(rows),
                  'success_rate': (successes / len(rows)) if rows else None,
                  'mean_q_score_final': (sum(q) / len(q)) if q else None,
                  'macro_task_success_rate': (sum(x['success_rate'] for x in per_task.values() if x['success_rate'] is not None) /
                                              sum(x['success_rate'] is not None for x in per_task.values())) if any(x['success_rate'] is not None for x in per_task.values()) else None},
    'per_task': per_task,
}
(root / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(root / 'evaluation_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
print(json.dumps(summary['aggregate'], indent=2))
print(f'WROTE {root / "evaluation_summary.json"}')
PY

echo "[rgbd-eval] complete: $RUN_ROOT/evaluation_summary.json"
