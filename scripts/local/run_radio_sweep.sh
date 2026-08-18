#!/usr/bin/env bash
set -Eeuo pipefail

[[ "$HOSTNAME" == behavior-tiancheng--* ]] || {
    echo "This script is only for behavior-tiancheng--* containers." >&2
    exit 2
}

[[ "${CONDA_DEFAULT_ENV:-}" == behavior ]] || {
    echo "Wrong Python environment. Run: behavioron" >&2
    exit 2
}

: "${OPENPI_ROOT:?Run: source ~/.bashrc}"
: "${B1K_ROOT:?Run: source ~/.bashrc}"
: "${CONDA_ROOT:?Run: source ~/.bashrc}"
: "${TC_HDD_ROOT:?Run: source ~/.bashrc}"
: "${EVAL_ROOT:?Run: source ~/.bashrc}"
: "${RADIO_CKPT:?Run: source ~/.bashrc}"

SERVER_SCRIPT="$OPENPI_ROOT/scripts/local/run_radio_server.sh"
EVAL_SCRIPT="$OPENPI_ROOT/scripts/local/run_radio_eval.sh"
HORIZONS=(2 4 8 16)
INSTANCE_INDEX=0
ACTUAL_INSTANCE=301
NUM_ROLLOUTS=4
PROMPT_LABEL=custom
PROMPT_NAME=recovery-v2
SWEEP_ID="${RADIO_SWEEP_ID:-radio_h_recovery_v2_h2-16_n4_20260812}"
HOST="${POLICY_HOST:-127.0.0.1}"
PORT="${POLICY_PORT:-8000}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"

RECOVERY_PROMPT="Turn on the radio by pressing its front power button. Locate the radio, front face, and button. If reachable, stabilize the radio with one hand and press with the other. If the radio moves, tips, or rotates, reassess its current pose. Press it directly if possible; otherwise grasp its handle or body and reposition it until the button is reachable. After a failed attempt, withdraw, observe again, and retry. Continue until the button is pressed."

[[ "$SWEEP_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "RADIO_SWEEP_ID contains invalid characters: $SWEEP_ID" >&2
    exit 2
}

[[ "$SERVER_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "SERVER_WAIT_SECONDS must be a positive integer." >&2
    exit 2
}

for file in "$SERVER_SCRIPT" "$EVAL_SCRIPT"; do
    [[ -x "$file" ]] || {
        echo "Missing executable: $file" >&2
        exit 2
    }
done

for command_name in python ffprobe flock git grep setsid sha256sum ss; do
    command -v "$command_name" >/dev/null || {
        echo "Missing command: $command_name" >&2
        exit 2
    }
done

SWEEP_ROOT="$EVAL_ROOT/sweeps/$SWEEP_ID"
MANIFEST="$SWEEP_ROOT/sweep_manifest.json"
SUMMARY="$SWEEP_ROOT/summary.csv"
mkdir -p "$SWEEP_ROOT"

exec 9>"$SWEEP_ROOT/.sweep.lock"
flock -n 9 || {
    echo "Another process already holds this sweep lock: $SWEEP_ROOT" >&2
    exit 2
}

DRIVER_LOG="$SWEEP_ROOT/driver.log"
exec > >(tee -a "$DRIVER_LOG") 2>&1
echo "[sweep] driver started at $(date --iso-8601=seconds)"
echo "[sweep] driver log: $DRIVER_LOG"

port_is_open() {
    ss -H -ltn "sport = :$PORT" | grep -q .
}

if port_is_open; then
    echo "Port $HOST:$PORT already has a listener. Stop the old server first." >&2
    ss -ltnp "sport = :$PORT" >&2 || true
    exit 2
fi

PROMPT_SHA256="$(printf '%s' "$RECOVERY_PROMPT" | sha256sum | awk '{print $1}')"
OPENPI_HEAD="$(git -C "$OPENPI_ROOT" rev-parse HEAD)"
B1K_HEAD="$(git -C "$B1K_ROOT" rev-parse HEAD)"
SERVER_SHA256="$(sha256sum "$SERVER_SCRIPT" | awk '{print $1}')"
EVAL_SHA256="$(sha256sum "$EVAL_SCRIPT" | awk '{print $1}')"

python - \
    "$MANIFEST" \
    "$SWEEP_ID" \
    "$PROMPT_NAME" \
    "$PROMPT_LABEL" \
    "$RECOVERY_PROMPT" \
    "$PROMPT_SHA256" \
    "$NUM_ROLLOUTS" \
    "$INSTANCE_INDEX" \
    "$ACTUAL_INSTANCE" \
    "$OPENPI_HEAD" \
    "$B1K_HEAD" \
    "$SERVER_SHA256" \
    "$EVAL_SHA256" <<'PY'
import datetime
import json
import pathlib
import sys

(
    manifest_path,
    sweep_id,
    prompt_name,
    prompt_label,
    prompt,
    prompt_sha256,
    num_rollouts,
    instance_index,
    actual_instance,
    openpi_head,
    b1k_head,
    server_sha256,
    eval_sha256,
) = sys.argv[1:]

manifest = {
    "schema_version": 1,
    "sweep_id": sweep_id,
    "horizons": [2, 4, 8, 16],
    "num_rollouts_per_horizon": int(num_rollouts),
    "instance_index": int(instance_index),
    "actual_instance": int(actual_instance),
    "prompt_name": prompt_name,
    "helper_prompt_label": prompt_label,
    "prompt": prompt,
    "prompt_sha256": prompt_sha256,
    "openpi_head": openpi_head,
    "b1k_head": b1k_head,
    "server_helper_sha256": server_sha256,
    "eval_helper_sha256": eval_sha256,
}

path = pathlib.Path(manifest_path)
if path.exists():
    old = json.loads(path.read_text())
    comparable_old = {key: old.get(key) for key in manifest}
    if comparable_old != manifest:
        print("Existing manifest does not match this experiment.", file=sys.stderr)
        for key, value in manifest.items():
            if old.get(key) != value:
                print(
                    f"  {key}: existing={old.get(key)!r} current={value!r}",
                    file=sys.stderr,
                )
        print("Use a new RADIO_SWEEP_ID; do not mix configurations.", file=sys.stderr)
        raise SystemExit(2)
else:
    manifest["created_at"] = datetime.datetime.now().astimezone().isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
PY

SERVER_SESSION_PID=""
SERVER_WAITER_PID=""
SERVER_PID_FILE=""

process_group_is_alive() {
    [[ -n "$SERVER_SESSION_PID" ]] &&
        kill -0 -- "-$SERVER_SESSION_PID" 2>/dev/null
}

stop_server() {
    local stop_status=0

    if process_group_is_alive; then
        echo "[sweep] stopping server process group $SERVER_SESSION_PID"
        kill -TERM -- "-$SERVER_SESSION_PID" 2>/dev/null || true

        for _ in $(seq 1 30); do
            process_group_is_alive || break
            sleep 1
        done

        if process_group_is_alive; then
            echo "[sweep] server did not stop after 30 seconds; sending KILL" >&2
            kill -KILL -- "-$SERVER_SESSION_PID" 2>/dev/null || true
        fi
    fi

    if [[ -n "$SERVER_WAITER_PID" ]]; then
        wait "$SERVER_WAITER_PID" 2>/dev/null || true
    fi

    SERVER_SESSION_PID=""
    SERVER_WAITER_PID=""

    if [[ -n "$SERVER_PID_FILE" ]]; then
        rm -f -- "$SERVER_PID_FILE"
    fi
    SERVER_PID_FILE=""

    for _ in $(seq 1 30); do
        port_is_open || return "$stop_status"
        sleep 1
    done

    echo "Port $HOST:$PORT is still occupied after server cleanup." >&2
    ss -ltnp "sport = :$PORT" >&2 || true
    return 1
}

on_exit() {
    local status=$?
    trap - EXIT INT TERM
    stop_server || status=1
    exit "$status"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

validate_run() {
    local run_dir=$1

    python - "$run_dir" "$NUM_ROLLOUTS" "$ACTUAL_INSTANCE" <<'PY'
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
expected_count = int(sys.argv[2])
expected_instance = int(sys.argv[3])
json_files = sorted((root / "json").glob("*.json"))
video_files = sorted((root / "videos").glob("*.mp4"))

errors = []
if len(json_files) != expected_count:
    errors.append(f"expected {expected_count} JSON files, found {len(json_files)}")
if len(video_files) != expected_count:
    errors.append(f"expected {expected_count} MP4 files, found {len(video_files)}")

rollout_ids = []
json_stems = set()
for path in json_files:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        errors.append(f"invalid JSON {path.name}: {exc}")
        continue

    json_stems.add(path.stem)
    if data.get("task") != "turning_on_radio":
        errors.append(f"{path.name}: wrong task {data.get('task')!r}")
    if data.get("instance_id") != expected_instance:
        errors.append(f"{path.name}: wrong instance_id {data.get('instance_id')!r}")
    if not isinstance(data.get("success"), bool):
        errors.append(f"{path.name}: success is not boolean")

    rollout_id = data.get("rollout_id")
    if not isinstance(rollout_id, int):
        errors.append(f"{path.name}: invalid rollout_id {rollout_id!r}")
    else:
        rollout_ids.append(rollout_id)

    steps = data.get("steps")
    if not isinstance(steps, int) or steps <= 0:
        errors.append(f"{path.name}: invalid steps {steps!r}")

if sorted(rollout_ids) != list(range(expected_count)):
    errors.append(
        f"rollout ids are {sorted(rollout_ids)}, expected {list(range(expected_count))}"
    )

video_stems = {path.stem for path in video_files}
if json_stems != video_stems:
    errors.append(
        f"JSON/video basename mismatch: json={sorted(json_stems)} video={sorted(video_stems)}"
    )

for path in video_files:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        errors.append(f"ffprobe failed for {path.name}: {result.stderr.strip()}")
        continue
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        errors.append(f"invalid video duration for {path.name}: {result.stdout!r}")
        continue
    if duration <= 0:
        errors.append(f"non-positive video duration for {path.name}: {duration}")

if errors:
    for error in errors:
        print(f"validation error: {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"validated_run={root}")
print(f"validated_json_count={len(json_files)}")
print(f"validated_video_count={len(video_files)}")
PY
}

marker_matches() {
    local marker=$1
    local horizon=$2

    python - "$marker" "$SWEEP_ID" "$PROMPT_SHA256" "$horizon" "$NUM_ROLLOUTS" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = {
    "sweep_id": sys.argv[2],
    "prompt_sha256": sys.argv[3],
    "horizon": int(sys.argv[4]),
    "num_rollouts": int(sys.argv[5]),
}
try:
    observed = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if all(observed.get(k) == v for k, v in expected.items()) else 1)
PY
}

find_complete_run() {
    local horizon=$1
    local config_dir="$SWEEP_ROOT/h$(printf '%02d' "$horizon")/$PROMPT_LABEL"
    local run_dir

    [[ -d "$config_dir/runs" ]] || return 1

    while IFS= read -r run_dir; do
        [[ -n "$run_dir" ]] || continue
        [[ -f "$run_dir/sweep_complete.json" ]] || continue
        marker_matches "$run_dir/sweep_complete.json" "$horizon" || continue
        validate_run "$run_dir" >/dev/null 2>&1 || continue
        printf '%s\n' "$run_dir"
        return 0
    done < <(
        find "$config_dir/runs" -mindepth 1 -maxdepth 1 -type d -print |
            sort -r
    )

    return 1
}

start_server() {
    local horizon=$1
    local config_dir="$SWEEP_ROOT/h$(printf '%02d' "$horizon")/$PROMPT_LABEL"
    local attempt_id="$(date +%Y%m%d_%H%M%S)_$$"
    local launch_log="$config_dir/server_launch_${attempt_id}.log"
    local server_config="$config_dir/server_config.env"
    local deadline config_pid config_prompt

    mkdir -p "$config_dir"
    SERVER_PID_FILE="$(mktemp -p "$SWEEP_ROOT" ".server_h${horizon}_pid.XXXXXX")"

    echo "[sweep] starting server: h=$horizon"
    echo "[sweep] launcher log: $launch_log"

    env \
        OPENPI_ROOT="$OPENPI_ROOT" \
        CONDA_ROOT="$CONDA_ROOT" \
        TC_HDD_ROOT="$TC_HDD_ROOT" \
        EVAL_ROOT="$EVAL_ROOT" \
        RADIO_CKPT="$RADIO_CKPT" \
        POLICY_PORT="$PORT" \
        RADIO_SWEEP_ID="$SWEEP_ID" \
        SERVER_SCRIPT="$SERVER_SCRIPT" \
        HORIZON="$horizon" \
        PROMPT_LABEL="$PROMPT_LABEL" \
        RECOVERY_PROMPT="$RECOVERY_PROMPT" \
        SERVER_PID_FILE="$SERVER_PID_FILE" \
        setsid bash -c '
            set -Eeuo pipefail
            printf "%s\n" "$$" > "$SERVER_PID_FILE"
            source "$CONDA_ROOT/etc/profile.d/conda.sh"
            while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
                conda deactivate 2>/dev/null || break
            done
            cd "$OPENPI_ROOT"
            source .venv/bin/activate
            exec "$SERVER_SCRIPT" "$HORIZON" "$PROMPT_LABEL" "$RECOVERY_PROMPT"
        ' >"$launch_log" 2>&1 &

    SERVER_WAITER_PID=$!
    deadline=$((SECONDS + 15))
    until [[ -s "$SERVER_PID_FILE" ]]; do
        kill -0 "$SERVER_WAITER_PID" 2>/dev/null || {
            echo "Server launcher exited before recording its session PID." >&2
            tail -80 "$launch_log" >&2 || true
            return 1
        }
        ((SECONDS < deadline)) || {
            echo "Timed out waiting for the server session PID." >&2
            return 1
        }
        sleep 1
    done

    SERVER_SESSION_PID="$(tr -d '[:space:]' < "$SERVER_PID_FILE")"
    [[ "$SERVER_SESSION_PID" =~ ^[1-9][0-9]*$ ]] || {
        echo "Invalid server session PID: $SERVER_SESSION_PID" >&2
        return 1
    }

    deadline=$((SECONDS + SERVER_WAIT_SECONDS))
    while true; do
        process_group_is_alive || {
            echo "Server process group exited during startup." >&2
            tail -100 "$launch_log" >&2 || true
            return 1
        }

        if [[ -f "$server_config" ]]; then
            config_pid="$(
                bash -c 'source "$1"; printf "%s" "$SERVER_PID"' _ "$server_config" 2>/dev/null || true
            )"
            config_prompt="$(
                bash -c 'source "$1"; printf "%s" "$SERVER_PROMPT"' _ "$server_config" 2>/dev/null || true
            )"

            if [[ "$config_pid" == "$SERVER_SESSION_PID" ]] &&
                [[ "$config_prompt" == "$RECOVERY_PROMPT" ]] &&
                port_is_open
            then
                echo "[sweep] server ready: pid=$SERVER_SESSION_PID port=$HOST:$PORT"
                return 0
            fi
        fi

        ((SECONDS < deadline)) || {
            echo "Server did not become ready within ${SERVER_WAIT_SECONDS}s." >&2
            tail -100 "$launch_log" >&2 || true
            return 1
        }
        sleep 2
    done
}

write_completion_marker() {
    local run_dir=$1
    local horizon=$2

    python - \
        "$run_dir/sweep_complete.json" \
        "$SWEEP_ID" \
        "$PROMPT_SHA256" \
        "$horizon" \
        "$NUM_ROLLOUTS" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = {
    "sweep_id": sys.argv[2],
    "prompt_sha256": sys.argv[3],
    "horizon": int(sys.argv[4]),
    "num_rollouts": int(sys.argv[5]),
    "validated_at": datetime.datetime.now().astimezone().isoformat(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(data, indent=2) + "\n")
temporary.replace(path)
PY
}

write_summary() {
    python - "$SWEEP_ROOT" "$PROMPT_LABEL" "$SUMMARY" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
prompt_label = sys.argv[2]
summary_path = pathlib.Path(sys.argv[3])
rows = []

for horizon in (2, 4, 8, 16):
    runs_root = root / f"h{horizon:02d}" / prompt_label / "runs"
    candidates = sorted(runs_root.glob("*/sweep_complete.json"), reverse=True)
    if not candidates:
        continue
    run_dir = candidates[0].parent
    for json_path in sorted((run_dir / "json").glob("*.json")):
        data = json.loads(json_path.read_text())
        q_score = data.get("q_score")
        if isinstance(q_score, dict):
            q_score = q_score.get("final")
        time_data = data.get("time") or {}
        rows.append(
            {
                "horizon": horizon,
                "run_id": run_dir.name,
                "instance_id": data.get("instance_id"),
                "rollout_id": data.get("rollout_id"),
                "success": data.get("success"),
                "q_score_final": q_score,
                "steps": data.get("steps"),
                "simulator_time": time_data.get("simulator_time"),
                "json": str(json_path),
                "video": str(run_dir / "videos" / f"{json_path.stem}.mp4"),
            }
        )

fieldnames = [
    "horizon",
    "run_id",
    "instance_id",
    "rollout_id",
    "success",
    "q_score_final",
    "steps",
    "simulator_time",
    "json",
    "video",
]
temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
with temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(summary_path)

print(f"summary={summary_path}")
print(f"summary_rows={len(rows)}")
print(f"successes={sum(row['success'] is True for row in rows)}")
PY
}

for horizon in "${HORIZONS[@]}"; do
    if complete_run="$(find_complete_run "$horizon")"; then
        echo "[sweep] h=$horizon already complete; skipping: $complete_run"
        continue
    fi

    echo
    echo "[sweep] ========================================"
    echo "[sweep] h=$horizon, rollouts=$NUM_ROLLOUTS, prompt=$PROMPT_NAME"
    echo "[sweep] ========================================"

    start_server "$horizon"

    config_dir="$SWEEP_ROOT/h$(printf '%02d' "$horizon")/$PROMPT_LABEL"
    eval_wrapper_log="$config_dir/eval_wrapper_$(date +%Y%m%d_%H%M%S)_$$.log"

    set +e
    RADIO_SWEEP_ID="$SWEEP_ID" \
    NUM_ROLLOUTS="$NUM_ROLLOUTS" \
    SERVER_WAIT_SECONDS=30 \
        "$EVAL_SCRIPT" "$horizon" "$PROMPT_LABEL" "$INSTANCE_INDEX" \
        2>&1 | tee "$eval_wrapper_log"
    eval_status=${PIPESTATUS[0]}
    set -e

    run_dir="$(sed -n 's/^RUN_DIR=//p' "$eval_wrapper_log" | head -1)"

    stop_server

    if grep -Eqi 'Token length .* exceeds max length|truncating' \
        "$config_dir/server.log" "$eval_wrapper_log" 2>/dev/null
    then
        echo "Prompt truncation detected at h=$horizon. Sweep stopped." >&2
        exit 1
    fi

    if ((eval_status != 0)); then
        echo "Evaluator infrastructure failed at h=$horizon (status=$eval_status)." >&2
        echo "Log: $eval_wrapper_log" >&2
        exit "$eval_status"
    fi

    [[ -n "$run_dir" && "$run_dir" == "$config_dir"/runs/* ]] || {
        echo "Could not identify the new evaluator run directory." >&2
        echo "Observed RUN_DIR=${run_dir:-<empty>}" >&2
        exit 1
    }

    validate_run "$run_dir"
    write_completion_marker "$run_dir" "$horizon"
    echo "[sweep] h=$horizon complete: $run_dir"
    write_summary
done

write_summary

summary_rows="$(python - "$SUMMARY" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    print(sum(1 for _ in csv.DictReader(stream)))
PY
)"

if [[ "$summary_rows" != 16 ]]; then
    echo "Final validation failed: expected 16 summary rows, found $summary_rows." >&2
    exit 1
fi

python - "$SWEEP_ROOT/sweep_complete.json" "$SWEEP_ID" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = {
    "sweep_id": sys.argv[2],
    "status": "completed",
    "completed_at": datetime.datetime.now().astimezone().isoformat(),
    "validated_results": 16,
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(data, indent=2) + "\n")
temporary.replace(path)
PY

echo
echo "[sweep] all 16 rollouts completed and validated"
echo "[sweep] results: $SWEEP_ROOT"
echo "[sweep] summary: $SUMMARY"