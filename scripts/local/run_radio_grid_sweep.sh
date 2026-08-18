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
: "${TC_HDD_ROOT:?Run: source ~/.bashrc}"
: "${EVAL_ROOT:?Run: source ~/.bashrc}"
: "${RADIO_CKPT:?Run: source ~/.bashrc}"

GRID_SERVER_SCRIPT="$OPENPI_ROOT/scripts/local/run_radio_grid_server.sh"
SERVE_SCRIPT="$OPENPI_ROOT/scripts/b1k/serve_b1k.py"

read -r -a GRID_HORIZONS <<< "${RADIO_GRID_HORIZONS:-8 16}"
read -r -a GRID_FLOW_STEPS <<< "${RADIO_GRID_FLOW_STEPS:-5 10 20}"
read -r -a GRID_INSTANCE_INDICES <<< "${RADIO_GRID_INSTANCE_INDICES:-0 9}"
read -r -a GRID_CONTROL_MODES <<< "${RADIO_GRID_CONTROL_MODES:-receding_horizon}"

GRID_REPEATS="${RADIO_GRID_REPEATS:-4}"
RUN_STRESS="${RADIO_RUN_STRESS:-1}"
RUN_GRID="${RADIO_RUN_GRID:-1}"
STRESS_INSTANCE_INDEX="${RADIO_STRESS_INSTANCE_INDEX:-9}"
STRESS_REPEATS="${RADIO_STRESS_REPEATS:-8}"
STRESS_HORIZON="${RADIO_STRESS_HORIZON:-16}"
STRESS_FLOW_STEPS="${RADIO_STRESS_FLOW_STEPS:-10}"
STRESS_CONTROL_MODE="${RADIO_STRESS_CONTROL_MODE:-receding_horizon}"

PROMPT_LABEL="${RADIO_PROMPT_LABEL:-original}"
CUSTOM_PROMPT="${RADIO_CUSTOM_PROMPT:-}"
SWEEP_ID="${RADIO_SWEEP_ID:-radio_pi05_grid_20260813}"
HOST="${POLICY_HOST:-127.0.0.1}"
PORT="${POLICY_PORT:-8000}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"
DRY_RUN="${RADIO_DRY_RUN:-0}"

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

for value in "$GRID_REPEATS" "$STRESS_REPEATS" "$STRESS_HORIZON" \
    "$STRESS_FLOW_STEPS" "$SERVER_WAIT_SECONDS"; do
    is_positive_integer "$value" || {
        echo "Expected a positive integer, got: $value" >&2
        exit 2
    }
done

for value in "$RUN_STRESS" "$RUN_GRID" "$DRY_RUN"; do
    [[ "$value" == 0 || "$value" == 1 ]] || {
        echo "Expected 0 or 1, got: $value" >&2
        exit 2
    }
done

for value in "${GRID_HORIZONS[@]}" "${GRID_FLOW_STEPS[@]}"; do
    is_positive_integer "$value" || {
        echo "Grid H and flow values must be positive integers: $value" >&2
        exit 2
    }
done

for value in "$STRESS_INSTANCE_INDEX" "${GRID_INSTANCE_INDICES[@]}"; do
    is_nonnegative_integer "$value" && (( value <= 19 )) || {
        echo "Public instance index must be in [0, 19], got: $value" >&2
        exit 2
    }
done

for mode in "$STRESS_CONTROL_MODE" "${GRID_CONTROL_MODES[@]}"; do
    case "$mode" in
        receding_horizon|receding_temporal|temporal_ensemble) ;;
        *)
            echo "Unsupported control mode: $mode" >&2
            exit 2
            ;;
    esac
done

[[ "$PROMPT_LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "RADIO_PROMPT_LABEL contains invalid characters: $PROMPT_LABEL" >&2
    exit 2
}
[[ "$SWEEP_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "RADIO_SWEEP_ID contains invalid characters: $SWEEP_ID" >&2
    exit 2
}
[[ "$PROMPT_LABEL" != custom || -n "$CUSTOM_PROMPT" ]] || {
    echo "RADIO_CUSTOM_PROMPT is required when RADIO_PROMPT_LABEL=custom." >&2
    exit 2
}

for file in "$GRID_SERVER_SCRIPT" "$SERVE_SCRIPT"; do
    [[ -f "$file" ]] || {
        echo "Missing file: $file" >&2
        exit 2
    }
done
[[ -x "$GRID_SERVER_SCRIPT" ]] || {
    echo "Not executable: $GRID_SERVER_SCRIPT" >&2
    exit 2
}

for command_name in python ffprobe flock git grep setsid sha256sum ss uv; do
    command -v "$command_name" >/dev/null || {
        echo "Missing command: $command_name" >&2
        exit 2
    }
done

grep -q 'num_flow_steps: int = 10' "$SERVE_SCRIPT" || {
    echo "serve_b1k.py does not expose num_flow_steps. Apply 0001-enable-num-flow-steps.patch first." >&2
    exit 2
}
grep -q 'sample_kwargs={"num_steps": args.num_flow_steps}' "$SERVE_SCRIPT" || {
    echo "serve_b1k.py does not pass num_flow_steps to create_trained_policy." >&2
    exit 2
}

case "$PROMPT_LABEL" in
    original)
        EFFECTIVE_PROMPT="$(
            "$OPENPI_ROOT/.venv/bin/python" - <<'PY'
from openpi.configs.tasks import TASK_REGISTRY
print(TASK_REGISTRY["b1k"]["turning_on_radio"])
PY
        )"
        ;;
    recovery-v1)
        EFFECTIVE_PROMPT="Turn on the radio. First locate the radio, its front face, and the power button. If the button is directly reachable, stabilize the radio with one hand and press the power button with the other. If the radio has moved, tipped over, rotated, or become unstable, first grasp its handle or body, lift or rotate it until the front button is reachable, stabilize it, and then press the button. If a grasp or press attempt fails, withdraw slightly, observe the current state again, reacquire the radio and button, and retry from the earliest incomplete stage. Do not continue a blocked motion or abandon the task after disturbing the radio. Continue until the power button has been successfully pressed."
        ;;
    custom)
        EFFECTIVE_PROMPT="$CUSTOM_PROMPT"
        ;;
    *)
        echo "Unknown prompt label: $PROMPT_LABEL" >&2
        exit 2
        ;;
esac

actual_instance() {
    printf '%d\n' "$((301 + $1))"
}

printf 'Planned stress: enabled=%s instance=%s(actual=%s) H=%s flow=%s mode=%s repeats=%s\n' \
    "$RUN_STRESS" "$STRESS_INSTANCE_INDEX" "$(actual_instance "$STRESS_INSTANCE_INDEX")" \
    "$STRESS_HORIZON" "$STRESS_FLOW_STEPS" "$STRESS_CONTROL_MODE" "$STRESS_REPEATS"
printf 'Planned grid: enabled=%s H={%s} flow={%s} indices={%s} modes={%s} repeats=%s\n' \
    "$RUN_GRID" "${GRID_HORIZONS[*]}" "${GRID_FLOW_STEPS[*]}" \
    "${GRID_INSTANCE_INDICES[*]}" "${GRID_CONTROL_MODES[*]}" "$GRID_REPEATS"
printf 'Prompt: label=%s text=%s\n' "$PROMPT_LABEL" "$EFFECTIVE_PROMPT"

if [[ "$DRY_RUN" == 1 ]]; then
    stress_count=0
    grid_count=0
    [[ "$RUN_STRESS" == 0 ]] || stress_count="$STRESS_REPEATS"
    if [[ "$RUN_GRID" == 1 ]]; then
        grid_count=$((${#GRID_HORIZONS[@]} * ${#GRID_FLOW_STEPS[@]} * ${#GRID_INSTANCE_INDICES[@]} * ${#GRID_CONTROL_MODES[@]} * GRID_REPEATS))
    fi
    printf 'Dry run only. Planned rollouts: stress=%d grid=%d total=%d\n' \
        "$stress_count" "$grid_count" "$((stress_count + grid_count))"
    exit 0
fi

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
echo "[sweep] root: $SWEEP_ROOT"

PROMPT_SHA256="$(printf '%s' "$EFFECTIVE_PROMPT" | sha256sum | awk '{print $1}')"
OPENPI_HEAD="$(git -C "$OPENPI_ROOT" rev-parse HEAD)"
B1K_HEAD="$(git -C "$B1K_ROOT" rev-parse HEAD)"
OPENPI_DIFF_SHA256="$(git -C "$OPENPI_ROOT" diff --binary --no-ext-diff | sha256sum | awk '{print $1}')"
B1K_DIFF_SHA256="$(git -C "$B1K_ROOT" diff --binary --no-ext-diff | sha256sum | awk '{print $1}')"
DRIVER_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
GRID_SERVER_SHA256="$(sha256sum "$GRID_SERVER_SCRIPT" | awk '{print $1}')"
SERVE_SHA256="$(sha256sum "$SERVE_SCRIPT" | awk '{print $1}')"

python - \
    "$MANIFEST" "$SWEEP_ID" "$PROMPT_LABEL" "$EFFECTIVE_PROMPT" "$PROMPT_SHA256" \
    "$RADIO_CKPT" "$OPENPI_HEAD" "$B1K_HEAD" "$OPENPI_DIFF_SHA256" \
    "$B1K_DIFF_SHA256" "$DRIVER_SHA256" \
    "$GRID_SERVER_SHA256" "$SERVE_SHA256" "$RUN_STRESS" "$RUN_GRID" \
    "$STRESS_INSTANCE_INDEX" "$STRESS_REPEATS" "$STRESS_HORIZON" \
    "$STRESS_FLOW_STEPS" "$STRESS_CONTROL_MODE" "$GRID_REPEATS" \
    "${GRID_HORIZONS[*]}" "${GRID_FLOW_STEPS[*]}" \
    "${GRID_INSTANCE_INDICES[*]}" "${GRID_CONTROL_MODES[*]}" <<'PY'
import datetime
import json
import pathlib
import sys

(
    manifest_path, sweep_id, prompt_label, prompt, prompt_sha256,
    checkpoint, openpi_head, b1k_head, openpi_diff_sha256, b1k_diff_sha256,
    driver_sha256, grid_server_sha256, serve_sha256, run_stress, run_grid,
    stress_index, stress_repeats,
    stress_horizon, stress_flow, stress_mode, grid_repeats, grid_horizons,
    grid_flows, grid_indices, grid_modes,
) = sys.argv[1:]

manifest = {
    "schema_version": 1,
    "sweep_id": sweep_id,
    "policy": "pi05_b1k",
    "checkpoint": checkpoint,
    "prompt_label": prompt_label,
    "prompt": prompt,
    "prompt_sha256": prompt_sha256,
    "openpi_head": openpi_head,
    "b1k_head": b1k_head,
    "openpi_diff_sha256": openpi_diff_sha256,
    "b1k_diff_sha256": b1k_diff_sha256,
    "driver_sha256": driver_sha256,
    "grid_server_sha256": grid_server_sha256,
    "serve_b1k_sha256": serve_sha256,
    "stress": {
        "enabled": bool(int(run_stress)),
        "instance_index": int(stress_index),
        "actual_instance": 301 + int(stress_index),
        "repeats": int(stress_repeats),
        "horizon": int(stress_horizon),
        "flow_steps": int(stress_flow),
        "control_mode": stress_mode,
    },
    "grid": {
        "enabled": bool(int(run_grid)),
        "repeats": int(grid_repeats),
        "horizons": [int(x) for x in grid_horizons.split()],
        "flow_steps": [int(x) for x in grid_flows.split()],
        "instance_indices": [int(x) for x in grid_indices.split()],
        "actual_instances": [301 + int(x) for x in grid_indices.split()],
        "control_modes": grid_modes.split(),
    },
}

path = pathlib.Path(manifest_path)
if path.exists():
    old = json.loads(path.read_text())
    comparable = {key: old.get(key) for key in manifest}
    if comparable != manifest:
        print("Existing manifest does not match this experiment.", file=sys.stderr)
        for key, value in manifest.items():
            if old.get(key) != value:
                print(f"  {key}: existing={old.get(key)!r} current={value!r}", file=sys.stderr)
        print("Use a new RADIO_SWEEP_ID; do not mix configurations.", file=sys.stderr)
        raise SystemExit(2)
else:
    manifest["created_at"] = datetime.datetime.now().astimezone().isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
PY

port_is_open() {
    ss -H -ltn "sport = :$PORT" | grep -q .
}

if port_is_open; then
    echo "Port $HOST:$PORT already has a listener. Stop the old server first." >&2
    ss -ltnp "sport = :$PORT" >&2 || true
    exit 2
fi

SERVER_SESSION_PID=""

process_group_is_alive() {
    [[ -n "$SERVER_SESSION_PID" ]] && kill -0 -- "-$SERVER_SESSION_PID" 2>/dev/null
}

stop_server() {
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
        wait "$SERVER_SESSION_PID" 2>/dev/null || true
    fi
    SERVER_SESSION_PID=""

    for _ in $(seq 1 30); do
        port_is_open || return 0
        sleep 1
    done
    echo "Port $HOST:$PORT remains occupied after cleanup." >&2
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

start_server() {
    local horizon=$1
    local flow_steps=$2
    local control_mode=$3
    local cell_root=$4

    port_is_open && {
        echo "Port $HOST:$PORT is occupied before server start." >&2
        return 1
    }
    mkdir -p "$cell_root"

    echo "[sweep] starting server: H=$horizon flow=$flow_steps mode=$control_mode"
    setsid bash -c '
        set -Eeuo pipefail
        source "$1/.venv/bin/activate"
        exec "$2" "$3" "$4" "$5" "$6" "$7" "$8"
    ' _ "$OPENPI_ROOT" "$GRID_SERVER_SCRIPT" "$horizon" "$flow_steps" \
        "$control_mode" "$PROMPT_LABEL" "$CUSTOM_PROMPT" "$cell_root" \
        >"$cell_root/server_launcher.log" 2>&1 &
    SERVER_SESSION_PID=$!
    printf '%s\n' "$SERVER_SESSION_PID" > "$cell_root/server_session.pid"

    local waited=0
    while (( waited < SERVER_WAIT_SECONDS )); do
        if port_is_open; then
            echo "[sweep] server ready after ${waited}s"
            return 0
        fi
        if ! process_group_is_alive; then
            echo "Server exited before opening port $PORT." >&2
            tail -100 "$cell_root/server_launcher.log" >&2 || true
            [[ -f "$cell_root/server.log" ]] && tail -100 "$cell_root/server.log" >&2 || true
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo "Server did not become ready within ${SERVER_WAIT_SECONDS}s." >&2
    tail -100 "$cell_root/server_launcher.log" >&2 || true
    [[ -f "$cell_root/server.log" ]] && tail -100 "$cell_root/server.log" >&2 || true
    return 1
}

rep_root_path() {
    local stage=$1 mode=$2 horizon=$3 flow_steps=$4 instance_index=$5 rep=$6
    local actual
    actual="$(actual_instance "$instance_index")"
    printf '%s/%s/mode-%s/h%02d/f%02d/%s/instance%s/rep%02d\n' \
        "$SWEEP_ROOT" "$stage" "$mode" "$horizon" "$flow_steps" \
        "$PROMPT_LABEL" "$actual" "$rep"
}

rep_is_complete() {
    local rep_root=$1 stage=$2 mode=$3 horizon=$4 flow_steps=$5
    local instance_index=$6 rep=$7
    local marker="$rep_root/rep_complete.json"
    [[ -f "$marker" ]] || return 1

    python - "$marker" "$stage" "$mode" "$horizon" "$flow_steps" \
        "$instance_index" "$(actual_instance "$instance_index")" "$rep" \
        "$PROMPT_SHA256" <<'PY'
import json
import pathlib
import subprocess
import sys

(
    marker_path, stage, mode, horizon, flow_steps, instance_index,
    actual_instance, rep, prompt_sha256,
) = sys.argv[1:]
marker_file = pathlib.Path(marker_path)
try:
    marker = json.loads(marker_file.read_text())
except Exception:
    raise SystemExit(1)

expected = {
    "stage": stage,
    "control_mode": mode,
    "horizon": int(horizon),
    "flow_steps": int(flow_steps),
    "instance_index": int(instance_index),
    "actual_instance": int(actual_instance),
    "rep": int(rep),
    "prompt_sha256": prompt_sha256,
}
if any(marker.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)

attempt = marker_file.parent / marker.get("attempt_relpath", "")
json_files = sorted((attempt / "json").glob("*.json"))
video_files = sorted((attempt / "videos").glob("*.mp4"))
if len(json_files) != 1 or len(video_files) != 1:
    raise SystemExit(1)
if json_files[0].stem != video_files[0].stem:
    raise SystemExit(1)

try:
    data = json.loads(json_files[0].read_text())
except Exception:
    raise SystemExit(1)
if data.get("task") != "turning_on_radio":
    raise SystemExit(1)
if data.get("instance_id") != int(actual_instance):
    raise SystemExit(1)
if not isinstance(data.get("success"), bool):
    raise SystemExit(1)
if data.get("rollout_id") != 0:
    raise SystemExit(1)
if not isinstance(data.get("steps"), int) or data["steps"] <= 0:
    raise SystemExit(1)

probe = subprocess.run(
    [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_files[0]),
    ],
    text=True,
    capture_output=True,
)
try:
    valid_video = probe.returncode == 0 and float(probe.stdout.strip()) > 0
except ValueError:
    valid_video = False
raise SystemExit(0 if valid_video else 1)
PY
}

cell_needs_work() {
    local stage=$1 mode=$2 horizon=$3 flow_steps=$4 repeats=$5
    shift 5
    local instance_index rep rep_root
    for instance_index in "$@"; do
        for ((rep = 0; rep < repeats; rep++)); do
            rep_root="$(rep_root_path "$stage" "$mode" "$horizon" "$flow_steps" "$instance_index" "$rep")"
            rep_is_complete "$rep_root" "$stage" "$mode" "$horizon" "$flow_steps" \
                "$instance_index" "$rep" || return 0
        done
    done
    return 1
}

rebuild_summary() {
    python - "$SWEEP_ROOT" "$SUMMARY" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
rows = []
for marker_path in root.glob("**/rep_complete.json"):
    try:
        marker = json.loads(marker_path.read_text())
        attempt = marker_path.parent / marker["attempt_relpath"]
        json_files = list((attempt / "json").glob("*.json"))
        if len(json_files) != 1:
            continue
        result = json.loads(json_files[0].read_text())
    except Exception:
        continue
    rows.append({
        "stage": marker.get("stage"),
        "control_mode": marker.get("control_mode"),
        "horizon": marker.get("horizon"),
        "flow_steps": marker.get("flow_steps"),
        "instance_index": marker.get("instance_index"),
        "actual_instance": marker.get("actual_instance"),
        "rep": marker.get("rep"),
        "success": result.get("success"),
        "q_score": json.dumps(result.get("q_score"), ensure_ascii=False, sort_keys=True),
        "steps": result.get("steps"),
        "sim_time": result.get("time"),
        "json_file": str(json_files[0]),
        "run_dir": str(attempt),
    })

rows.sort(key=lambda row: (
    str(row["stage"]), str(row["control_mode"]), int(row["horizon"]),
    int(row["flow_steps"]), int(row["instance_index"]), int(row["rep"]),
))
fields = [
    "stage", "control_mode", "horizon", "flow_steps", "instance_index",
    "actual_instance", "rep", "success", "q_score", "steps", "sim_time",
    "json_file", "run_dir",
]
temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(summary_path)
print(f"[sweep] summary rows={len(rows)} path={summary_path}")
PY
}

run_one_rep() {
    local stage=$1 mode=$2 horizon=$3 flow_steps=$4 instance_index=$5 rep=$6
    local rep_root actual run_id attempt_dir eval_status marker temporary
    rep_root="$(rep_root_path "$stage" "$mode" "$horizon" "$flow_steps" "$instance_index" "$rep")"
    actual="$(actual_instance "$instance_index")"

    if rep_is_complete "$rep_root" "$stage" "$mode" "$horizon" "$flow_steps" \
        "$instance_index" "$rep"; then
        echo "[sweep] skip complete: stage=$stage mode=$mode H=$horizon flow=$flow_steps instance=$actual rep=$rep"
        return 0
    fi

    run_id="$(date +%Y%m%d_%H%M%S)_$$"
    attempt_dir="$rep_root/attempts/$run_id"
    mkdir -p "$attempt_dir"
    {
        printf 'STAGE=%q\n' "$stage"
        printf 'CONTROL_MODE=%q\n' "$mode"
        printf 'HORIZON=%q\n' "$horizon"
        printf 'FLOW_STEPS=%q\n' "$flow_steps"
        printf 'INSTANCE_INDEX=%q\n' "$instance_index"
        printf 'ACTUAL_INSTANCE=%q\n' "$actual"
        printf 'REP=%q\n' "$rep"
        printf 'PROMPT_LABEL=%q\n' "$PROMPT_LABEL"
        printf 'PROMPT_SHA256=%q\n' "$PROMPT_SHA256"
        printf 'STARTED_AT=%q\n' "$(date --iso-8601=seconds)"
    } > "$attempt_dir/eval_config.env"

    echo "[sweep] run: stage=$stage mode=$mode H=$horizon flow=$flow_steps instance=$actual(index=$instance_index) rep=$rep"
    set +e
    (
        cd "$B1K_ROOT"
        python -m omnigibson.eval.eval \
            --task-name turning_on_radio \
            --host "$HOST" \
            --port "$PORT" \
            --robot-config "$B1K_ROOT/OmniGibson/omnigibson/eval/r1pro.yaml" \
            --mode public_test \
            --instance-indices "$instance_index" \
            --num-rollouts 1 \
            --policy websocket \
            --output-dir "$attempt_dir" \
            --write-video \
            --headless
    ) 2>&1 | tee "$attempt_dir/eval.log"
    eval_status=${PIPESTATUS[0]}
    set -e
    if (( eval_status != 0 )); then
        echo "Evaluator failed with status $eval_status; preserved attempt: $attempt_dir" >&2
        return "$eval_status"
    fi

    marker="$rep_root/rep_complete.json"
    temporary="$marker.tmp"
    python - "$attempt_dir" "$temporary" "$stage" "$mode" "$horizon" \
        "$flow_steps" "$instance_index" "$actual" "$rep" "$PROMPT_SHA256" <<'PY'
import datetime
import json
import pathlib
import subprocess
import sys

(
    attempt_dir, marker_path, stage, mode, horizon, flow_steps,
    instance_index, actual_instance, rep, prompt_sha256,
) = sys.argv[1:]
attempt = pathlib.Path(attempt_dir)
json_files = sorted((attempt / "json").glob("*.json"))
video_files = sorted((attempt / "videos").glob("*.mp4"))
errors = []
if len(json_files) != 1:
    errors.append(f"expected 1 JSON, found {len(json_files)}")
if len(video_files) != 1:
    errors.append(f"expected 1 MP4, found {len(video_files)}")

data = None
if len(json_files) == 1:
    try:
        data = json.loads(json_files[0].read_text())
    except Exception as exc:
        errors.append(f"invalid JSON: {exc}")
if data is not None:
    if data.get("task") != "turning_on_radio":
        errors.append(f"wrong task: {data.get('task')!r}")
    if data.get("instance_id") != int(actual_instance):
        errors.append(f"wrong instance: {data.get('instance_id')!r}")
    if data.get("rollout_id") != 0:
        errors.append(f"wrong rollout_id: {data.get('rollout_id')!r}")
    if not isinstance(data.get("success"), bool):
        errors.append("success is not boolean")
    if not isinstance(data.get("steps"), int) or data["steps"] <= 0:
        errors.append(f"invalid steps: {data.get('steps')!r}")
if len(json_files) == 1 and len(video_files) == 1:
    if json_files[0].stem != video_files[0].stem:
        errors.append("JSON/video basename mismatch")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_files[0]),
        ],
        text=True,
        capture_output=True,
    )
    try:
        valid_video = probe.returncode == 0 and float(probe.stdout.strip()) > 0
    except ValueError:
        valid_video = False
    if not valid_video:
        errors.append("invalid or empty video")

if errors:
    for error in errors:
        print(f"validation error: {error}", file=sys.stderr)
    raise SystemExit(1)

marker_file = pathlib.Path(marker_path)
relative_attempt = attempt.relative_to(marker_file.parent)
marker = {
    "schema_version": 1,
    "completed_at": datetime.datetime.now().astimezone().isoformat(),
    "stage": stage,
    "control_mode": mode,
    "horizon": int(horizon),
    "flow_steps": int(flow_steps),
    "instance_index": int(instance_index),
    "actual_instance": int(actual_instance),
    "rep": int(rep),
    "prompt_sha256": prompt_sha256,
    "attempt_relpath": str(relative_attempt),
}
marker_file.write_text(json.dumps(marker, indent=2) + "\n")
PY
    mv "$temporary" "$marker"
    rebuild_summary
}

run_cell() {
    local stage=$1 mode=$2 horizon=$3 flow_steps=$4 repeats=$5
    shift 5
    local indices=("$@")
    local cell_root instance_index rep
    cell_root="$SWEEP_ROOT/$stage/mode-$mode/h$(printf '%02d' "$horizon")/f$(printf '%02d' "$flow_steps")/$PROMPT_LABEL"

    if ! cell_needs_work "$stage" "$mode" "$horizon" "$flow_steps" "$repeats" "${indices[@]}"; then
        echo "[sweep] skip complete cell: stage=$stage mode=$mode H=$horizon flow=$flow_steps"
        return 0
    fi

    start_server "$horizon" "$flow_steps" "$mode" "$cell_root"
    for instance_index in "${indices[@]}"; do
        for ((rep = 0; rep < repeats; rep++)); do
            run_one_rep "$stage" "$mode" "$horizon" "$flow_steps" "$instance_index" "$rep"
        done
    done
    stop_server
}

rebuild_summary

if [[ "$RUN_STRESS" == 1 ]]; then
    run_cell stress "$STRESS_CONTROL_MODE" "$STRESS_HORIZON" \
        "$STRESS_FLOW_STEPS" "$STRESS_REPEATS" "$STRESS_INSTANCE_INDEX"
fi

if [[ "$RUN_GRID" == 1 ]]; then
    for mode in "${GRID_CONTROL_MODES[@]}"; do
        for horizon in "${GRID_HORIZONS[@]}"; do
            for flow_steps in "${GRID_FLOW_STEPS[@]}"; do
                run_cell grid "$mode" "$horizon" "$flow_steps" "$GRID_REPEATS" \
                    "${GRID_INSTANCE_INDICES[@]}"
            done
        done
    done
fi

rebuild_summary
echo "[sweep] complete at $(date --iso-8601=seconds)"
echo "[sweep] summary: $SUMMARY"
