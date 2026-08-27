#!/usr/bin/env bash
set -Eeuo pipefail

# Isolated benchmark for the B1K -> WEAVER preparation path.
#
# This script requires a new output directory and never reuses existing
# artifacts. It is deliberately a shell harness around the existing
# prepare_weaver_dataset.py interface so that the benchmark measures the real
# reader/adapter/writer.
# It refuses to run while another preparation process is active unless the
# caller explicitly sets ALLOW_PRODUCTION_CONTENTION=1.

usage() {
  cat <<'EOF'
Usage:
  scripts/b1k/benchmark_weaver_prepare.sh

Environment overrides:
  OPENPI_ROOT              OpenPI checkout (default: current git root)
  DATASET_ROOT             B1K dataset root
  WEAVER_WORK_ROOT         work root (default: /tmp/weaver-benchmark)
  BENCH_MODE               dry-run or write (default: dry-run)
  BENCH_RUNTIME            serial or parallel (default: serial)
  BENCH_EPISODES           space-separated episode indices (default: 780..787)
  BENCH_OUT                explicit new, isolated output directory
  WORKERS                  parallel worker count (default: 4)
  MAX_IN_FLIGHT            optional parallel in-flight bound (default: 2*workers)
  CPUSET                   optional taskset CPU list, e.g. 0-7
  NICE_LEVEL               optional nice level, e.g. 10
  THREADS                  optional BLAS/OpenMP thread limit, e.g. 1 or 8
  SAMPLE_SECONDS           process telemetry interval (default: 5)
  ALLOW_PRODUCTION_CONTENTION=1
                           override the safety refusal when a real preparation
                           process is already running; normally do not set it.

Examples:
  BENCH_MODE=dry-run DATASET_ROOT=/path/to/v3again \
    scripts/b1k/benchmark_weaver_prepare.sh

  BENCH_MODE=write DATASET_ROOT=/path/to/v3again CPUSET=0-15 \
    scripts/b1k/benchmark_weaver_prepare.sh

  BENCH_RUNTIME=parallel WORKERS=4 THREADS=1 \
    DATASET_ROOT=/path/to/v3again scripts/b1k/benchmark_weaver_prepare.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

OPENPI_ROOT="${OPENPI_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
DATASET_ROOT="${DATASET_ROOT:-/inspire/dataset/behavior-1k-2026-challenge-demos/v3again}"
WEAVER_WORK_ROOT="${WEAVER_WORK_ROOT:-/tmp/weaver-benchmark}"
BENCH_MODE="${BENCH_MODE:-dry-run}"
BENCH_RUNTIME="${BENCH_RUNTIME:-serial}"
BENCH_EPISODES="${BENCH_EPISODES:-780 781 782 783 784 785 786 787}"
WORKERS="${WORKERS:-4}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-5}"
PROCESS_LIST_PATH="${TMPDIR:-/tmp}/weaver_bench_prepare_ps.$$"
pipeline_pid=""
telemetry_pid=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  if [[ -n "$pipeline_pid" ]]; then
    read -r -a cleanup_pids <<< "$(collect_process_tree "$pipeline_pid")"
    for ((index=${#cleanup_pids[@]} - 1; index >= 0; index--)); do
      kill "${cleanup_pids[index]}" 2>/dev/null || true
    done
    wait "$pipeline_pid" 2>/dev/null || true
  fi
  rm -f "$PROCESS_LIST_PATH" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

collect_process_tree() {
  local frontier="$1"
  local all="$1"
  local next pid children
  while [[ -n "$frontier" ]]; do
    next=""
    for pid in $frontier; do
      children="$(pgrep -P "$pid" 2>/dev/null || true)"
      if [[ -n "$children" ]]; then
        all="$all $children"
        next="$next $children"
      fi
    done
    frontier="$next"
  done
  printf '%s\n' "$all"
}

case "$BENCH_MODE" in
  dry-run|write) ;;
  *) echo "BENCH_MODE must be dry-run or write, got: $BENCH_MODE" >&2; exit 2 ;;
esac
case "$BENCH_RUNTIME" in
  serial|parallel) ;;
  *) echo "BENCH_RUNTIME must be serial or parallel, got: $BENCH_RUNTIME" >&2; exit 2 ;;
esac

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ -n "${MAX_IN_FLIGHT:-}" ]] && ! [[ "$MAX_IN_FLIGHT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_IN_FLIGHT must be a positive integer" >&2
  exit 2
fi

if ! [[ "$SAMPLE_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMPLE_SECONDS must be a positive integer" >&2
  exit 2
fi

if [[ ! -d "$OPENPI_ROOT" ]]; then
  echo "OPENPI_ROOT does not exist: $OPENPI_ROOT" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "DATASET_ROOT does not exist: $DATASET_ROOT" >&2
  exit 2
fi

if pgrep -af 'prepare_weaver_dataset(_parallel)?\.py' >"$PROCESS_LIST_PATH" 2>/dev/null; then
  if [[ "${ALLOW_PRODUCTION_CONTENTION:-0}" != "1" ]]; then
    echo "A prepare_weaver_dataset.py process is already running:" >&2
    cat "$PROCESS_LIST_PATH" >&2
    echo "Refusing to benchmark concurrently. Run after production finishes, or" >&2
    echo "set ALLOW_PRODUCTION_CONTENTION=1 only if you explicitly accept CPU/I/O contention." >&2
    exit 3
  fi
  echo "WARNING: benchmarking while another preparation process is active." >&2
  echo "WARNING: elapsed time and I/O measurements will not be directly comparable." >&2
fi
rm -f "$PROCESS_LIST_PATH" 2>/dev/null || true

if [[ -z "${BENCH_OUT:-}" ]]; then
  stamp="$(date +%Y%m%d-%H%M%S)-$$"
  BENCH_OUT="$WEAVER_WORK_ROOT/${BENCH_MODE}-${stamp}"
fi
if [[ "$BENCH_OUT" != /* ]]; then
  echo "BENCH_OUT must be an absolute path, got: $BENCH_OUT" >&2
  exit 2
fi
if [[ -e "$BENCH_OUT" ]]; then
  echo "BENCH_OUT already exists; refusing to reuse or modify it: $BENCH_OUT" >&2
  exit 4
fi
LOG_DIR="${BENCH_LOG_DIR:-$WEAVER_WORK_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/prepare-${BENCH_MODE}-$(date +%Y%m%d-%H%M%S)-$$.log"
TELEMETRY_PATH="${LOG_PATH%.log}.telemetry.log"

read -r -a EPISODES <<< "$BENCH_EPISODES"
if (( ${#EPISODES[@]} == 0 )); then
  echo "BENCH_EPISODES must contain at least one episode index" >&2
  exit 2
fi

prepare_script="scripts/b1k/prepare_weaver_dataset.py"
if [[ "$BENCH_RUNTIME" == "parallel" ]]; then
  prepare_script="scripts/b1k/prepare_weaver_dataset_parallel.py"
fi

cmd=(
  uv run python "$prepare_script"
  --config-name pi05_b1k_full
  --dataset-root "$DATASET_ROOT"
  --output-dir "$BENCH_OUT"
  --episode-indices "${EPISODES[@]}"
  --target-hz 5
  --min-sampled-frames 10
  --val-fraction 0.1
  --split-seed 42
  --held-out-task-fraction 0.1
  --fail-fast
)
if [[ "$BENCH_RUNTIME" == "parallel" ]]; then
  cmd+=(--workers "$WORKERS" --worker-threads "${THREADS:-1}")
  if [[ -n "${MAX_IN_FLIGHT:-}" ]]; then
    cmd+=(--max-in-flight "$MAX_IN_FLIGHT")
  fi
fi
if [[ "$BENCH_MODE" == "dry-run" ]]; then
  cmd+=(--dry-run)
fi

run_cmd=("${cmd[@]}")
if [[ -n "${CPUSET:-}" ]]; then
  if ! command -v taskset >/dev/null 2>&1; then
    echo "CPUSET was set but taskset is unavailable" >&2
    exit 2
  fi
  run_cmd=(taskset -c "$CPUSET" "${run_cmd[@]}")
fi

export PYTHONUNBUFFERED=1
if [[ -n "${THREADS:-}" ]]; then
  if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "THREADS must be a positive integer" >&2
    exit 2
  fi
  export OMP_NUM_THREADS="$THREADS"
  export MKL_NUM_THREADS="$THREADS"
  export OPENBLAS_NUM_THREADS="$THREADS"
  export NUMEXPR_NUM_THREADS="$THREADS"
fi

start_epoch="$(date +%s)"
if [[ -n "${NICE_LEVEL:-}" ]]; then
  if ! [[ "$NICE_LEVEL" =~ ^-?[0-9]+$ ]]; then
    echo "NICE_LEVEL must be an integer" >&2
    exit 2
  fi
  run_cmd=(nice -n "$NICE_LEVEL" "${run_cmd[@]}")
fi

{
  echo "===== WEAVER prepare benchmark ====="
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "mode=$BENCH_MODE"
  echo "runtime=$BENCH_RUNTIME"
  echo "openpi_root=$OPENPI_ROOT"
  echo "git_revision=$(git -C "$OPENPI_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "dataset_root=$DATASET_ROOT"
  echo "output_dir=$BENCH_OUT"
  echo "episodes=${EPISODES[*]}"
  echo "cpuset=${CPUSET:-unrestricted}"
  echo "threads=${THREADS:-inherited}"
  if [[ "$BENCH_RUNTIME" == "parallel" ]]; then
    echo "workers=$WORKERS"
    echo "max_in_flight=${MAX_IN_FLIGHT:-default}"
  fi
  echo "nice_level=${NICE_LEVEL:-inherited}"
  echo "nproc=$(nproc 2>/dev/null || echo unavailable)"
  awk '/^Cpus_allowed_list:|^Mems_allowed_list:/{print}' /proc/self/status 2>/dev/null || true
  free -h 2>/dev/null || true
  df -h "$DATASET_ROOT" "$WEAVER_WORK_ROOT" 2>/dev/null || true
  echo "log=$LOG_PATH"
  echo "telemetry=$TELEMETRY_PATH"
  printf 'command='
  printf '%q ' "${run_cmd[@]}"
  printf '\n===== task output =====\n'
} | tee "$LOG_PATH"

(
  cd "$OPENPI_ROOT"
  "${run_cmd[@]}" >>"$LOG_PATH" 2>&1
) &
pipeline_pid=$!

{
  echo "===== telemetry ====="
  while kill -0 "$pipeline_pid" 2>/dev/null; do
    now="$(date --iso-8601=seconds)"
    read -r -a tree_pids <<< "$(collect_process_tree "$pipeline_pid")"
    if (( ${#tree_pids[@]} > 0 )); then
      pid_csv="$(IFS=,; echo "${tree_pids[*]}")"
      ps -o pid,ppid,etime,%cpu,%mem,rss,vsz,threads,stat,cmd -p "$pid_csv" || true
      for observed_pid in "${tree_pids[@]}"; do
        if [[ -r "/proc/$observed_pid/io" ]]; then
          awk -v now="$now" -v pid="$observed_pid" \
            '/^read_bytes:|^write_bytes:|^syscr:|^syscw:/{printf "%s pid=%s %s %s\n", now, pid, $1, $2}' \
            "/proc/$observed_pid/io"
        fi
      done
    else
      ps -o pid,ppid,etime,%cpu,%mem,rss,vsz,threads,stat,cmd -p "$pipeline_pid" || true
    fi
    if [[ "$BENCH_MODE" == "write" && -d "$BENCH_OUT" ]]; then
      printf 'artifact_counts '
      printf 'npz='; find "$BENCH_OUT/episodes" -type f -name '*.npz' 2>/dev/null | wc -l
      printf 'json='; find "$BENCH_OUT/episodes" -type f -name '*.json' 2>/dev/null | wc -l
      printf 'bytes='; du -sb "$BENCH_OUT" 2>/dev/null | awk '{print $1}' || true
    fi
    sleep "$SAMPLE_SECONDS"
  done
} >"$TELEMETRY_PATH" 2>&1 &
telemetry_pid=$!

set +e
wait "$pipeline_pid"
status=$?
set -e
kill "$telemetry_pid" 2>/dev/null || true
wait "$telemetry_pid" 2>/dev/null || true
telemetry_pid=""
pipeline_pid=""

end_epoch="$(date +%s)"
elapsed=$((end_epoch - start_epoch))
completed="$(grep -cE '^\[[0-9]+/[0-9]+\]' "$LOG_PATH" 2>/dev/null || true)"
completed_source_rows="$(
  awk '
    /^\[[0-9]+\/[0-9]+\]/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^source=[0-9]+$/) {
          split($i, pair, "=")
          rows += pair[2]
        }
      }
    }
    END { print rows + 0 }
  ' "$LOG_PATH"
)"
echo "status=$status elapsed_seconds=$elapsed completed_episodes=$completed log=$LOG_PATH telemetry=$TELEMETRY_PATH"
if (( elapsed > 0 )) && [[ "$completed" =~ ^[0-9]+$ ]]; then
  awk -v episodes="$completed" -v seconds="$elapsed" \
    'BEGIN { printf "throughput_episodes_per_minute=%.3f\n", episodes * 60.0 / seconds }'
  awk -v rows="$completed_source_rows" -v seconds="$elapsed" \
    'BEGIN { printf "throughput_source_rows_per_second=%.3f\n", rows / seconds }'
fi
if [[ "$BENCH_MODE" == "write" && -d "$BENCH_OUT" ]]; then
  printf 'final_counts '
  printf 'npz='; find "$BENCH_OUT/episodes" -type f -name '*.npz' 2>/dev/null | wc -l
  printf 'json='; find "$BENCH_OUT/episodes" -type f -name '*.json' 2>/dev/null | wc -l
  printf 'bytes='; du -sb "$BENCH_OUT" 2>/dev/null | awk '{print $1}' || true
fi
exit "$status"
