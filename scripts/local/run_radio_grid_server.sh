#!/usr/bin/env bash
set -Eeuo pipefail

[[ "$HOSTNAME" == behavior-tiancheng--* ]] || {
    echo "This script is only for behavior-tiancheng--* containers." >&2
    exit 2
}

: "${OPENPI_ROOT:?Run: source ~/.bashrc}"
: "${RADIO_CKPT:?Run: source ~/.bashrc}"

[[ "${VIRTUAL_ENV:-}" == "$OPENPI_ROOT/.venv" ]] || {
    echo "Wrong Python environment. The grid driver should activate $OPENPI_ROOT/.venv." >&2
    exit 2
}

HORIZON="${1:?Usage: run_radio_grid_server.sh HORIZON FLOW_STEPS CONTROL_MODE PROMPT_LABEL CUSTOM_PROMPT CONFIG_DIR}"
FLOW_STEPS="${2:?Missing FLOW_STEPS}"
CONTROL_MODE="${3:?Missing CONTROL_MODE}"
PROMPT_LABEL="${4:?Missing PROMPT_LABEL}"
CUSTOM_PROMPT="${5:-}"
CONFIG_DIR="${6:?Missing CONFIG_DIR}"
PORT="${POLICY_PORT:-8000}"

[[ "$HORIZON" =~ ^[1-9][0-9]*$ ]] || {
    echo "HORIZON must be a positive integer." >&2
    exit 2
}
[[ "$FLOW_STEPS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FLOW_STEPS must be a positive integer." >&2
    exit 2
}
[[ "$PROMPT_LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "PROMPT_LABEL contains invalid characters: $PROMPT_LABEL" >&2
    exit 2
}
case "$CONTROL_MODE" in
    receding_horizon|receding_temporal|temporal_ensemble) ;;
    *)
        echo "Unsupported CONTROL_MODE: $CONTROL_MODE" >&2
        exit 2
        ;;
esac

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
            echo "RADIO_CUSTOM_PROMPT is required for custom prompt mode." >&2
            exit 2
        }
        EFFECTIVE_PROMPT="$CUSTOM_PROMPT"
        PROMPT_ARGS=(--prompt "$EFFECTIVE_PROMPT")
        ;;
    *)
        echo "Unknown prompt label: $PROMPT_LABEL" >&2
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

mkdir -p "$CONFIG_DIR"
STARTED_AT="$(date --iso-8601=seconds)"
CONFIG_HISTORY="$CONFIG_DIR/server_config_$(date +%Y%m%d_%H%M%S)_$$.env"
{
    printf 'SERVER_PID=%q\n' "$$"
    printf 'SERVER_HORIZON=%q\n' "$HORIZON"
    printf 'SERVER_FLOW_STEPS=%q\n' "$FLOW_STEPS"
    printf 'SERVER_CONTROL_MODE=%q\n' "$CONTROL_MODE"
    printf 'SERVER_PROMPT_LABEL=%q\n' "$PROMPT_LABEL"
    printf 'SERVER_PROMPT=%q\n' "$EFFECTIVE_PROMPT"
    printf 'SERVER_PORT=%q\n' "$PORT"
    printf 'SERVER_STARTED_AT=%q\n' "$STARTED_AT"
    printf 'SERVER_GIT_HEAD=%q\n' "$(git rev-parse HEAD)"
    printf 'SERVER_SERVE_SHA256=%q\n' "$(sha256sum scripts/b1k/serve_b1k.py | awk '{print $1}')"
} > "$CONFIG_HISTORY"
cp "$CONFIG_HISTORY" "$CONFIG_DIR/server_config.env"

exec > >(tee -a "$CONFIG_DIR/server.log") 2>&1

echo "CONFIG_DIR=$CONFIG_DIR"
echo "HORIZON=$HORIZON"
echo "FLOW_STEPS=$FLOW_STEPS"
echo "CONTROL_MODE=$CONTROL_MODE"
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
    --control-mode "$CONTROL_MODE" \
    --action-horizon "$HORIZON" \
    --num-flow-steps "$FLOW_STEPS" \
    --port "$PORT" \
    "${PROMPT_ARGS[@]}"

