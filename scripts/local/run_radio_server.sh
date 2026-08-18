#!/usr/bin/env bash
set -Eeuo pipefail

[[ "$HOSTNAME" == behavior-tiancheng--* ]] || {
    echo "This script is only for behavior-tiancheng--* containers." >&2
    exit 2
}

: "${OPENPI_ROOT:?Run: source ~/.bashrc}"
: "${TC_HDD_ROOT:?Run: source ~/.bashrc}"
: "${EVAL_ROOT:?Run: source ~/.bashrc}"
: "${RADIO_CKPT:?Run: source ~/.bashrc}"

[[ "${VIRTUAL_ENV:-}" == "$OPENPI_ROOT/.venv" ]] || {
    echo "Wrong Python environment. Run: openpion" >&2
    exit 2
}

HORIZON="${1:-8}"
PROMPT_LABEL="${2:-original}"
CUSTOM_PROMPT="${3:-}"
SWEEP_ID="${RADIO_SWEEP_ID:-radio_h_prompt_20260812}"
PORT="${POLICY_PORT:-8000}"

RECORD="${POLICY_RECORD:-0}"

[[ "$RECORD" == "0" || "$RECORD" == "1" ]] || {
    echo "POLICY_RECORD must be 0 or 1." >&2
    exit 2
}

RECORD_ARGS=()
[[ "$RECORD" == "1" ]] && RECORD_ARGS=(--record)

[[ "$HORIZON" =~ ^[1-9][0-9]*$ ]] || {
    echo "HORIZON must be a positive integer." >&2
    exit 2
}
[[ "$PROMPT_LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "PROMPT_LABEL may contain only letters, numbers, dot, underscore, and hyphen." >&2
    exit 2
}

cd "$OPENPI_ROOT"

REGISTRY_PROMPT="$(
    python - <<'PY'
from openpi.configs.tasks import TASK_REGISTRY
print(TASK_REGISTRY["b1k"]["turning_on_radio"])
PY
)"

PROMPT_ARGS=()
case "$PROMPT_LABEL" in
    original)
        EFFECTIVE_PROMPT="$REGISTRY_PROMPT"
        ;;
    recovery-v1)
        EFFECTIVE_PROMPT="Turn on the radio. First locate the radio, its front face, and the power button. If the button is directly reachable, stabilize the radio with one hand and press the power button with the other. If the radio has moved, tipped over, rotated, or become unstable, first grasp its handle or body, lift or rotate it until the front button is reachable, stabilize it, and then press the button. If a grasp or press attempt fails, withdraw slightly, observe the current state again, reacquire the radio and button, and retry from the earliest incomplete stage. Do not continue a blocked motion or abandon the task after disturbing the radio. Continue until the power button has been successfully pressed."
        PROMPT_ARGS=(--prompt "$EFFECTIVE_PROMPT")
        ;;
    custom)
        [[ -n "$CUSTOM_PROMPT" ]] || {
            echo 'For custom mode, pass the prompt as argument 3.' >&2
            exit 2
        }
        EFFECTIVE_PROMPT="$CUSTOM_PROMPT"
        PROMPT_ARGS=(--prompt "$EFFECTIVE_PROMPT")
        ;;
    *)
        echo "Unknown prompt label: $PROMPT_LABEL" >&2
        echo "Use: original, recovery-v1, or custom" >&2
        exit 2
        ;;
esac

if python - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as sock:
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
then
    echo "Port $PORT already has a listener; stop the old server first." >&2
    ss -ltnp "sport = :$PORT" 2>/dev/null || true
    exit 2
fi

CONFIG_DIR="$EVAL_ROOT/sweeps/$SWEEP_ID/h$(printf '%02d' "$HORIZON")/$PROMPT_LABEL"
mkdir -p "$CONFIG_DIR"

{
    printf 'SERVER_PID=%q\n' "$$"
    printf 'SERVER_HORIZON=%q\n' "$HORIZON"
    printf 'SERVER_PROMPT_LABEL=%q\n' "$PROMPT_LABEL"
    printf 'SERVER_PROMPT=%q\n' "$EFFECTIVE_PROMPT"
    printf 'SERVER_PORT=%q\n' "$PORT"
    printf 'SERVER_STARTED_AT=%q\n' "$(date --iso-8601=seconds)"
    printf 'SERVER_GIT_HEAD=%q\n' "$(git rev-parse HEAD)"
} > "$CONFIG_DIR/server_config.env"

exec > >(tee "$CONFIG_DIR/server.log") 2>&1

echo "CONFIG_DIR=$CONFIG_DIR"
echo "HORIZON=$HORIZON"
echo "PROMPT_LABEL=$PROMPT_LABEL"
printf 'PROMPT=%s\n' "$EFFECTIVE_PROMPT"

exec env \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro \
    --task b1k/turning_on_radio \
    --repo-id turning_on_radio \
    --policy.config pi05_b1k \
    --policy.dir "$RADIO_CKPT" \
    --control-mode receding_horizon \
    --action-horizon "$HORIZON" \
    --port "$PORT" \
    "${RECORD_ARGS[@]}" \
    "${PROMPT_ARGS[@]}"
