"""Monitor a running B1K training process and emit an auditable report.

This script intentionally uses only the Python standard library. It can run from
either the OpenPI virtual environment or the BEHAVIOR conda environment on Linux.
Raw one-second samples are written to JSONL independently of the lower-frequency
terminal dashboard, avoiding the sampling aliasing caused by ``watch nvidia-smi``.
"""

from __future__ import annotations

import argparse
import collections.abc
import datetime as dt
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any

GIB = 1024**3
MIB = 1024**2
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(
    r"Step\s+(?P<step>\d+):\s+grad_norm=(?P<grad_norm>[-+\d.eE]+),\s+"
    r"loss=(?P<loss>[-+\d.eE]+),\s+param_norm=(?P<param_norm>[-+\d.eE]+)"
)
PROGRESS_RE = re.compile(r"(?P<step>\d+)/(?P<total>\d+)\s+\[[^\]]*?,\s*(?P<seconds>[\d.]+)s/it\]")
LOGGABLE_PROGRESS_RE = re.compile(
    r"Progress on:\s+(?P<step>[\d.]+)(?P<step_unit>[kM]?it)/"
    r"(?P<total>[\d.]+)(?P<total_unit>[kM]?it)\s+rate:(?P<seconds>[\d.]+)s/it"
)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def _read_int(path: Path) -> int | None:
    value = _read_text(path)
    if value is None:
        return None
    value = value.strip()
    return int(value) if value.isdigit() else None


def _proc_cmdline(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
        return None


def _proc_stat(pid: int) -> dict[str, int | str] | None:
    value = _read_text(Path(f"/proc/{pid}/stat"))
    if value is None:
        return None
    close_paren = value.rfind(")")
    if close_paren < 0:
        return None
    fields = value[close_paren + 2 :].split()
    if len(fields) < 20:
        return None
    return {
        "state": fields[0],
        "ppid": int(fields[1]),
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "start_ticks": int(fields[19]),
    }


def _proc_status(pid: int) -> dict[str, int]:
    value = _read_text(Path(f"/proc/{pid}/status"))
    if value is None:
        return {}
    fields: dict[str, int] = {}
    for line in value.splitlines():
        name, separator, raw_value = line.partition(":")
        if not separator:
            continue
        parts = raw_value.split()
        if not parts or not parts[0].isdigit():
            continue
        multiplier = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
        fields[name] = int(parts[0]) * multiplier
    return fields


def _fd_count(pid: int) -> int | None:
    try:
        return len(tuple(Path(f"/proc/{pid}/fd").iterdir()))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def _direct_children(pid: int) -> list[int]:
    value = _read_text(Path(f"/proc/{pid}/task/{pid}/children"))
    if value is None:
        return []
    return [int(child) for child in value.split() if child.isdigit()]


def _comm(pid: int) -> str | None:
    value = _read_text(Path(f"/proc/{pid}/comm"))
    return None if value is None else value.strip()


def _discover_process(match: str) -> int | None:
    candidates: list[tuple[int, int]] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        command = _proc_cmdline(pid)
        if command is None or match not in command or "monitor_b1k_training.py" in command:
            continue
        stat = _proc_stat(pid)
        if stat is not None:
            candidates.append((int(stat["start_ticks"]), pid))
    return max(candidates)[1] if candidates else None


class _ProcessCpuTracker:
    def __init__(self) -> None:
        self._clock_ticks = os.sysconf("SC_CLK_TCK")
        self._previous: dict[int, tuple[int, float]] = {}

    def measure(self, ticks: dict[int, int], now: float) -> float | None:
        cpu_seconds = 0.0
        elapsed_reference: list[float] = []
        measured = False
        for pid, current_ticks in ticks.items():
            previous = self._previous.get(pid)
            if previous is None:
                continue
            previous_ticks, previous_time = previous
            elapsed = now - previous_time
            if elapsed <= 0 or current_ticks < previous_ticks:
                continue
            cpu_seconds += (current_ticks - previous_ticks) / self._clock_ticks
            elapsed_reference.append(elapsed)
            measured = True
        self._previous = {pid: (value, now) for pid, value in ticks.items()}
        if not measured:
            return None
        elapsed = statistics.median(elapsed_reference)
        return 100 * cpu_seconds / elapsed if elapsed > 0 else None


class _HostCpuTracker:
    def __init__(self) -> None:
        self._previous: list[int] | None = None

    def measure(self) -> dict[str, float] | None:
        value = _read_text(Path("/proc/stat"))
        if value is None:
            return None
        first_line = value.splitlines()[0].split()
        if not first_line or first_line[0] != "cpu":
            return None
        current = [int(item) for item in first_line[1:9]]
        previous = self._previous
        self._previous = current
        if previous is None:
            return None
        delta = [new - old for new, old in zip(current, previous, strict=True)]
        total = sum(delta)
        if total <= 0:
            return None
        return {
            "user_pct": 100 * (delta[0] + delta[1]) / total,
            "system_pct": 100 * (delta[2] + delta[5] + delta[6]) / total,
            "idle_pct": 100 * delta[3] / total,
            "iowait_pct": 100 * delta[4] / total,
            "steal_pct": 100 * delta[7] / total,
        }


def _read_cgroup_memory() -> dict[str, int | None]:
    current_paths = (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    )
    limit_paths = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    stat_paths = (
        Path("/sys/fs/cgroup/memory.stat"),
        Path("/sys/fs/cgroup/memory/memory.stat"),
    )
    current = next((value for path in current_paths if (value := _read_int(path)) is not None), None)
    limit = next((value for path in limit_paths if (value := _read_int(path)) is not None), None)
    stats: dict[str, int] = {}
    for path in stat_paths:
        value = _read_text(path)
        if value is None:
            continue
        for line in value.splitlines():
            name, raw_value = line.split(maxsplit=1)
            if raw_value.isdigit():
                stats[name] = int(raw_value)
        break
    return {
        "current_bytes": current,
        "limit_bytes": limit,
        "anon_bytes": stats.get("anon"),
        "file_bytes": stats.get("file"),
    }


def _query_gpus() -> tuple[list[dict[str, int]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return [], str(error)
    gpus: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not all(part.isdigit() for part in parts):
            continue
        index, utilization, memory_used, memory_total = map(int, parts)
        gpus.append(
            {
                "index": index,
                "utilization_pct": utilization,
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
            }
        )
    return gpus, None


def _load_averages() -> list[float] | None:
    value = _read_text(Path("/proc/loadavg"))
    if value is None:
        return None
    return [float(item) for item in value.split()[:3]]


def _collect_sample(
    pid: int,
    *,
    started_wall: float,
    process_cpu: _ProcessCpuTracker,
    worker_cpu: _ProcessCpuTracker,
    host_cpu: _HostCpuTracker,
    expected_workers: int | None,
) -> dict[str, Any]:
    now_wall = time.time()
    now_monotonic = time.monotonic()
    trainer_stat = _proc_stat(pid)
    trainer_status = _proc_status(pid)
    trainer_ticks = {} if trainer_stat is None else {pid: int(trainer_stat["cpu_ticks"])}
    trainer_cpu_pct = process_cpu.measure(trainer_ticks, now_monotonic)

    worker_pids = [child for child in _direct_children(pid) if _comm(child) == "pt_data_worker"]
    worker_ticks: dict[int, int] = {}
    worker_rss = 0
    worker_anon = 0
    worker_fds = 0
    for worker_pid in worker_pids:
        stat = _proc_stat(worker_pid)
        if stat is not None:
            worker_ticks[worker_pid] = int(stat["cpu_ticks"])
        status = _proc_status(worker_pid)
        worker_rss += status.get("VmRSS", 0)
        worker_anon += status.get("RssAnon", 0)
        worker_fds += _fd_count(worker_pid) or 0

    gpus, gpu_error = _query_gpus()
    cgroup = _read_cgroup_memory()
    return {
        "schema_version": 1,
        "timestamp": now_wall,
        "timestamp_iso": dt.datetime.fromtimestamp(now_wall, tz=dt.datetime.now().astimezone().tzinfo).isoformat(),
        "elapsed_s": now_wall - started_wall,
        "trainer": {
            "pid": pid,
            "running": trainer_stat is not None,
            "state": None if trainer_stat is None else trainer_stat["state"],
            "cpu_pct": trainer_cpu_pct,
            "rss_bytes": trainer_status.get("VmRSS"),
            "anon_bytes": trainer_status.get("RssAnon"),
            "threads": trainer_status.get("Threads"),
            "fds": _fd_count(pid),
        },
        "workers": {
            "expected": expected_workers,
            "count": len(worker_pids),
            "pids": worker_pids,
            "cpu_pct": worker_cpu.measure(worker_ticks, now_monotonic),
            "rss_bytes": worker_rss,
            "anon_bytes": worker_anon,
            "fds": worker_fds,
        },
        "cgroup_memory": cgroup,
        "gpus": gpus,
        "gpu_error": gpu_error,
        "host_cpu": host_cpu.measure(),
        "load_average": _load_averages(),
    }


def _linear_slope(
    samples: list[dict[str, Any]], value_getter: collections.abc.Callable[[dict[str, Any]], Any]
) -> float | None:
    points = [
        (float(sample["timestamp"]), float(value)) for sample in samples if (value := value_getter(sample)) is not None
    ]
    if len(points) < 2:
        return None
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def _average(values: collections.abc.Iterable[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return sum(filtered) / len(filtered) if filtered else None


def _longest_all_zero_run(samples: list[dict[str, Any]], sample_interval: float) -> float:
    longest = 0.0
    run_started: float | None = None
    previous_timestamp: float | None = None
    for sample in samples:
        timestamp = float(sample["timestamp"])
        gpus = sample["gpus"]
        all_zero = bool(gpus) and all(gpu["utilization_pct"] == 0 for gpu in gpus)
        gap_too_large = previous_timestamp is not None and timestamp - previous_timestamp > 3 * sample_interval
        if all_zero and not gap_too_large:
            run_started = timestamp if run_started is None else run_started
            longest = max(longest, timestamp - run_started + sample_interval)
        elif all_zero:
            run_started = timestamp
        else:
            run_started = None
        previous_timestamp = timestamp
    return longest


def _summarize_window(samples: list[dict[str, Any]], sample_interval: float) -> dict[str, Any]:
    if not samples:
        return {"samples": 0}
    gpu_values = [gpu["utilization_pct"] for sample in samples for gpu in sample["gpus"]]
    all_zero = [
        bool(sample["gpus"]) and all(gpu["utilization_pct"] == 0 for gpu in sample["gpus"]) for sample in samples
    ]
    all_high = [
        bool(sample["gpus"]) and all(gpu["utilization_pct"] >= 80 for gpu in sample["gpus"]) for sample in samples
    ]
    memory_slope = _linear_slope(samples, lambda sample: sample["cgroup_memory"]["current_bytes"])
    duration = float(samples[-1]["timestamp"]) - float(samples[0]["timestamp"])
    return {
        "samples": len(samples),
        "duration_s": duration,
        "average_gpu_util_pct": _average(gpu_values),
        "all_gpu_zero_pct": 100 * sum(all_zero) / len(all_zero),
        "all_gpu_at_least_80_pct": 100 * sum(all_high) / len(all_high),
        "longest_all_gpu_zero_s": _longest_all_zero_run(samples, sample_interval),
        "cgroup_memory_start_gib": samples[0]["cgroup_memory"]["current_bytes"] / GIB,
        "cgroup_memory_end_gib": samples[-1]["cgroup_memory"]["current_bytes"] / GIB,
        "cgroup_memory_max_gib": max(sample["cgroup_memory"]["current_bytes"] for sample in samples) / GIB,
        "cgroup_memory_slope_gib_per_min": None if memory_slope is None else memory_slope * 60 / GIB,
        "worker_count_min": min(sample["workers"]["count"] for sample in samples),
        "worker_count_max": max(sample["workers"]["count"] for sample in samples),
        "worker_cpu_pct_average": _average(sample["workers"]["cpu_pct"] for sample in samples),
        "worker_rss_gib_max": max(sample["workers"]["rss_bytes"] for sample in samples) / GIB,
        "worker_fds_max": max(sample["workers"]["fds"] for sample in samples),
        "trainer_rss_gib_max": max((sample["trainer"]["rss_bytes"] or 0) for sample in samples) / GIB,
        "host_idle_pct_average": _average(
            None if sample["host_cpu"] is None else sample["host_cpu"]["idle_pct"] for sample in samples
        ),
        "host_iowait_pct_average": _average(
            None if sample["host_cpu"] is None else sample["host_cpu"]["iowait_pct"] for sample in samples
        ),
    }


def _parse_train_log(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    text = ANSI_ESCAPE_RE.sub("", path.read_text(errors="replace")).replace("\r", "\n")
    steps = [
        {
            "step": int(match.group("step")),
            "grad_norm": float(match.group("grad_norm")),
            "loss": float(match.group("loss")),
            "param_norm": float(match.group("param_norm")),
        }
        for match in STEP_RE.finditer(text)
    ]
    progress_matches = list(PROGRESS_RE.finditer(text))
    progress = None
    if progress_matches:
        match = progress_matches[-1]
        progress = {
            "step": int(match.group("step")),
            "total": int(match.group("total")),
            "seconds_per_iteration": float(match.group("seconds")),
        }
    else:
        loggable_matches = list(LOGGABLE_PROGRESS_RE.finditer(text))
        if loggable_matches:
            match = loggable_matches[-1]
            units = {"it": 1, "kit": 1_000, "Mit": 1_000_000}
            progress = {
                "step": round(float(match.group("step")) * units[match.group("step_unit")]),
                "total": round(float(match.group("total")) * units[match.group("total_unit")]),
                "seconds_per_iteration": float(match.group("seconds")),
            }
    return {
        "initialized_data_loader": "Initialized data loader" in text,
        "initialized_train_state": "Initialized train state" in text,
        "resource_exhausted": "RESOURCE_EXHAUSTED" in text,
        "traceback": "Traceback (most recent call last)" in text,
        "checkpoint_save_count": text.count("Finished saving checkpoint"),
        "waiting_for_checkpoint_manager": "Waiting for checkpoint manager to finish" in text,
        "steps": steps,
        "last_step": steps[-1] if steps else None,
        "progress": progress,
    }


def _format_optional(value: Any, suffix: str = "", digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"


def _dashboard(
    pid: int,
    command: str,
    output_dir: Path,
    samples: list[dict[str, Any]],
    *,
    sample_interval: float,
    window_seconds: float,
    train_log: Path | None,
) -> None:
    cutoff = samples[-1]["timestamp"] - window_seconds
    window = [sample for sample in samples if sample["timestamp"] >= cutoff]
    summary = _summarize_window(window, sample_interval)
    current = samples[-1]
    gpus = " ".join(f"{gpu['index']}:{gpu['utilization_pct']}%/{gpu['memory_used_mib']}MiB" for gpu in current["gpus"])
    log_summary = _parse_train_log(train_log)
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    print(f"B1K training monitor  {current['timestamp_iso']}")
    print(f"pid={pid} running={current['trainer']['running']} output={output_dir}")
    print(f"command={command}")
    print(
        f"workers={current['workers']['count']}/{current['workers']['expected'] or '?'} "
        f"cpu={_format_optional(current['workers']['cpu_pct'], '%')} "
        f"rss={current['workers']['rss_bytes'] / GIB:.1f}GiB fds={current['workers']['fds']}"
    )
    print(
        f"cgroup={current['cgroup_memory']['current_bytes'] / GIB:.1f}GiB "
        f"trainer_rss={(current['trainer']['rss_bytes'] or 0) / GIB:.1f}GiB"
    )
    print(f"gpu_now={gpus or current['gpu_error'] or 'unavailable'}")
    print(
        f"rolling={summary.get('duration_s', 0) / 60:.1f}min "
        f"gpu_avg={_format_optional(summary.get('average_gpu_util_pct'), '%')} "
        f"all_zero={_format_optional(summary.get('all_gpu_zero_pct'), '%')} "
        f"longest_zero={_format_optional(summary.get('longest_all_gpu_zero_s'), 's')} "
        f"memory_slope={_format_optional(summary.get('cgroup_memory_slope_gib_per_min'), 'GiB/min', 3)}"
    )
    host = current["host_cpu"]
    if host is not None:
        print(f"host_cpu idle={host['idle_pct']:.1f}% iowait={host['iowait_pct']:.1f}% load={current['load_average']}")
    if log_summary is not None:
        print(f"train_log last_step={log_summary['last_step']} progress={log_summary['progress']}")
    print("Ctrl-C finalizes samples.jsonl, summary.json, and report.md", flush=True)


def _make_summary(
    samples: list[dict[str, Any]],
    *,
    sample_interval: float,
    warmup_seconds: float,
    cooldown_seconds: float,
    window_seconds: float,
    metadata: dict[str, Any],
    train_log: Path | None,
    exit_reason: str,
) -> dict[str, Any]:
    alive = [sample for sample in samples if sample["trainer"]["running"]]
    if alive:
        first_timestamp = alive[0]["timestamp"]
        last_timestamp = alive[-1]["timestamp"]
        steady = [
            sample
            for sample in alive
            if sample["timestamp"] >= first_timestamp + warmup_seconds
            and sample["timestamp"] <= last_timestamp - cooldown_seconds
        ]
        last_steady_timestamp = last_timestamp - cooldown_seconds
        last_window = [sample for sample in steady if sample["timestamp"] >= last_steady_timestamp - window_seconds]
    else:
        steady = []
        last_window = []
    return {
        "schema_version": 1,
        "exit_reason": exit_reason,
        "metadata": metadata,
        "sample_interval_s": sample_interval,
        "warmup_seconds_excluded": warmup_seconds,
        "cooldown_seconds_excluded": cooldown_seconds,
        "overall": _summarize_window(alive, sample_interval),
        "steady": _summarize_window(steady, sample_interval),
        "last_steady_window": _summarize_window(last_window, sample_interval),
        "train_log": _parse_train_log(train_log),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    metadata = summary["metadata"]
    lines = [
        "# B1K training monitor report",
        "",
        f"- Exit reason: `{summary['exit_reason']}`",
        f"- Trainer PID: `{metadata['pid']}`",
        f"- Started monitoring: `{metadata['monitor_started_iso']}`",
        f"- Command: `{metadata['command']}`",
        f"- Sample interval: `{summary['sample_interval_s']} s`",
        f"- Excluded warm-up / cooldown: `{summary['warmup_seconds_excluded']} s` / "
        f"`{summary['cooldown_seconds_excluded']} s`",
        "",
        "## Resource summary",
        "",
        "| Window | Duration | GPU avg | All GPUs zero | Longest zero | Memory slope | Max cgroup | Workers | Max worker RSS | Max FDs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Overall", "overall"),
        ("Steady", "steady"),
        ("Last steady window", "last_steady_window"),
    ):
        window = summary[key]
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    _format_optional(window.get("duration_s", 0) / 60, " min"),
                    _format_optional(window.get("average_gpu_util_pct"), "%"),
                    _format_optional(window.get("all_gpu_zero_pct"), "%"),
                    _format_optional(window.get("longest_all_gpu_zero_s"), " s"),
                    _format_optional(window.get("cgroup_memory_slope_gib_per_min"), " GiB/min", 3),
                    _format_optional(window.get("cgroup_memory_max_gib"), " GiB"),
                    f"{window.get('worker_count_min', 'n/a')}-{window.get('worker_count_max', 'n/a')}",
                    _format_optional(window.get("worker_rss_gib_max"), " GiB"),
                    str(window.get("worker_fds_max", "n/a")),
                )
            )
            + " |"
        )
    lines.extend(("", "## Training log", ""))
    log_summary = summary["train_log"]
    if log_summary is None:
        lines.append("No `--train-log` was supplied; step metrics and checkpoint messages were not captured.")
    else:
        lines.extend(
            (
                f"- Data loader initialized: `{log_summary['initialized_data_loader']}`",
                f"- Train state initialized: `{log_summary['initialized_train_state']}`",
                f"- Last step metrics: `{log_summary['last_step']}`",
                f"- Last progress: `{log_summary['progress']}`",
                f"- Completed checkpoint saves: `{log_summary['checkpoint_save_count']}`",
                f"- RESOURCE_EXHAUSTED observed: `{log_summary['resource_exhausted']}`",
                f"- Traceback observed: `{log_summary['traceback']}`",
            )
        )
    lines.extend(
        (
            "",
            "## Interpretation guardrails",
            "",
            "The steady window excludes configured startup and shutdown/checkpoint margins. A short near-zero memory "
            "slope is evidence for stability, not a proof that a full 50k-step run cannot grow. Inspect `samples.jsonl` "
            "when a summary metric is surprising.",
            "",
        )
    )
    path.write_text("\n".join(lines))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, help="Trainer Python PID. Auto-discovered when omitted.")
    parser.add_argument("--match", default="scripts/b1k/train_b1k.py", help="Command-line substring for discovery.")
    parser.add_argument("--expected-workers", type=int, help="Expected pt_data_worker count.")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--display-interval", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=300.0)
    parser.add_argument("--warmup-seconds", type=float, default=300.0)
    parser.add_argument("--cooldown-seconds", type=float, default=120.0)
    parser.add_argument("--wait-seconds", type=float, default=600.0)
    parser.add_argument("--post-exit-seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-log", type=Path, help="Optional output captured with `2>&1 | tee TRAIN_LOG`.")
    parser.add_argument("--no-dashboard", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.sample_interval <= 0 or args.display_interval <= 0 or args.window_seconds <= 0:
        raise ValueError("Sample, display, and window intervals must be positive")
    if args.warmup_seconds < 0 or args.cooldown_seconds < 0 or args.post_exit_seconds < 0:
        raise ValueError("Warm-up, cooldown, and post-exit intervals must be non-negative")

    pid = args.pid
    wait_started = time.monotonic()
    while pid is None and time.monotonic() - wait_started < args.wait_seconds:
        pid = _discover_process(args.match)
        if pid is None:
            print(f"Waiting for a process matching {args.match!r}...", flush=True)
            time.sleep(min(5.0, args.sample_interval))
    if pid is None:
        raise SystemExit(f"No process matching {args.match!r} appeared within {args.wait_seconds} seconds")

    command = _proc_cmdline(pid)
    if command is None:
        raise SystemExit(f"PID {pid} is not readable or has already exited")
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if args.output_dir is None:
        root = Path(os.environ.get("TC_HDD_ROOT", ".")) / "outputs" / "train-monitor"
        output_dir = root / f"{timestamp}-pid{pid}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "pid": pid,
        "command": command,
        "monitor_started_iso": dt.datetime.now().astimezone().isoformat(),
        "hostname": os.uname().nodename,
        "output_dir": str(output_dir),
        "train_log": None if args.train_log is None else str(args.train_log),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    samples: list[dict[str, Any]] = []
    process_cpu = _ProcessCpuTracker()
    worker_cpu = _ProcessCpuTracker()
    host_cpu = _HostCpuTracker()
    started_wall = time.time()
    next_sample = time.monotonic()
    next_display = next_sample
    exited_at: float | None = None
    exit_reason = "process_exited"

    samples_path = output_dir / "samples.jsonl"
    try:
        with samples_path.open("w", buffering=1) as output_file:
            while True:
                sample = _collect_sample(
                    pid,
                    started_wall=started_wall,
                    process_cpu=process_cpu,
                    worker_cpu=worker_cpu,
                    host_cpu=host_cpu,
                    expected_workers=args.expected_workers,
                )
                samples.append(sample)
                output_file.write(json.dumps(sample, sort_keys=True) + "\n")

                now = time.monotonic()
                if not args.no_dashboard and now >= next_display:
                    _dashboard(
                        pid,
                        command,
                        output_dir,
                        samples,
                        sample_interval=args.sample_interval,
                        window_seconds=args.window_seconds,
                        train_log=args.train_log,
                    )
                    next_display = now + args.display_interval

                if sample["trainer"]["running"]:
                    exited_at = None
                else:
                    exited_at = now if exited_at is None else exited_at
                    if now - exited_at >= args.post_exit_seconds:
                        break

                next_sample += args.sample_interval
                time.sleep(max(0.0, next_sample - time.monotonic()))
    except KeyboardInterrupt:
        exit_reason = "monitor_interrupted"
        print("\nMonitor interrupted; finalizing report...", flush=True)
    finally:
        summary = _make_summary(
            samples,
            sample_interval=args.sample_interval,
            warmup_seconds=args.warmup_seconds,
            cooldown_seconds=args.cooldown_seconds,
            window_seconds=args.window_seconds,
            metadata=metadata,
            train_log=args.train_log,
            exit_reason=exit_reason,
        )
        summary_path = output_dir / "summary.json"
        report_path = output_dir / "report.md"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _write_report(report_path, summary)
        print(f"Raw samples: {samples_path}")
        print(f"Summary:     {summary_path}")
        print(f"Report:      {report_path}")


if __name__ == "__main__":
    main()
