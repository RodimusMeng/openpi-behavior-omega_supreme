#!/usr/bin/env bash
set -Eeuo pipefail

[[ "$HOSTNAME" == behavior-tiancheng--* ]] || {
    echo "This script is only for behavior-tiancheng--* containers." >&2
    exit 2
}

: "${B1K_ROOT:?Run: source ~/.bashrc}"
: "${EVAL_ROOT:?Run: source ~/.bashrc}"

[[ "${CONDA_DEFAULT_ENV:-}" == behavior ]] || {
    echo "Wrong Python environment. Run: behavioron" >&2
    exit 2
}

HORIZON="${1:-8}"
PROMPT_LABEL="${2:-original}"
shift "$(( $# >= 2 ? 2 : $# ))"

if (( $# > 0 )); then
    INSTANCES=("$@")
else
    INSTANCES=(0 1 2)
fi

SWEEP_ID="${RADIO_SWEEP_ID:-radio_h_prompt_20260812}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
HOST="${POLICY_HOST:-127.0.0.1}"
PORT="${POLICY_PORT:-8000}"
WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"

CONFIG_DIR="$EVAL_ROOT/sweeps/$SWEEP_ID/h$(printf '%02d' "$HORIZON")/$PROMPT_LABEL"
SERVER_CONFIG="$CONFIG_DIR/server_config.env"

[[ -f "$SERVER_CONFIG" ]] || {
    echo "Missing server config: $SERVER_CONFIG" >&2
    echo "Start the matching server first." >&2
    exit 2
}

# shellcheck disable=SC1090
source "$SERVER_CONFIG"
[[ "$SERVER_HORIZON" == "$HORIZON" ]] || {
    echo "Server horizon mismatch: server=$SERVER_HORIZON eval=$HORIZON" >&2
    exit 2
}
[[ "$SERVER_PROMPT_LABEL" == "$PROMPT_LABEL" ]] || {
    echo "Server prompt mismatch: server=$SERVER_PROMPT_LABEL eval=$PROMPT_LABEL" >&2
    exit 2
}
kill -0 "$SERVER_PID" 2>/dev/null || {
    echo "Recorded server PID $SERVER_PID is not running." >&2
    exit 2
}

deadline=$((SECONDS + WAIT_SECONDS))
until python - "$HOST" "$PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket() as sock:
    sock.settimeout(1)
    sys.exit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
do
    (( SECONDS < deadline )) || {
        echo "Server did not listen on $HOST:$PORT within ${WAIT_SECONDS}s." >&2
        exit 2
    }
    sleep 2
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$CONFIG_DIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
cp "$SERVER_CONFIG" "$RUN_DIR/"

{
    printf 'MODE=%q\n' public_test
    printf 'INSTANCE_INDICES=%q\n' "${INSTANCES[*]}"
    printf 'NUM_ROLLOUTS=%q\n' "$NUM_ROLLOUTS"
    printf 'EVAL_STARTED_AT=%q\n' "$(date --iso-8601=seconds)"
} > "$RUN_DIR/eval_config.env"

cd "$B1K_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "INSTANCES=${INSTANCES[*]}"
echo "NUM_ROLLOUTS=$NUM_ROLLOUTS"

set +e
python -m omnigibson.eval.eval \
    --task-name turning_on_radio \
    --host "$HOST" \
    --port "$PORT" \
    --mode public_test \
    --instance-indices "${INSTANCES[@]}" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --policy websocket \
    --output-dir "$RUN_DIR" \
    --write-video \
    --headless \
    2>&1 | tee "$RUN_DIR/eval.log"
eval_status=${PIPESTATUS[0]}
set -e

python - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = sorted((root / "json").glob("*.json"))
print(f"result_json_count={len(files)}")
for path in files:
    data = json.loads(path.read_text())
    q_score = data.get("q_score")
    if isinstance(q_score, dict):
        q_score = q_score.get("final")
    print(f"{path.name}\tsuccess={data.get('success')}\tq_score={q_score}")
PY

printf 'evaluator_exit_status=%s\n' "$eval_status"
exit "$eval_status"
