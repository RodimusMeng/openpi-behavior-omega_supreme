"""Benchmark B1K video-reader memory without starting the OpenPI model stack."""

from __future__ import annotations

import dataclasses
import gc
import inspect
import json
import os
from pathlib import Path
import time
from typing import Literal

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tyro

GIB = 1024**3


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str = "b1k-v3again"
    dataset_root: str | None = None
    episode: int = 0
    mode: Literal["none", "rgb", "depth", "rgbd"] = "rgbd"
    pattern: Literal["sequential", "repeat"] = "sequential"
    samples: int = 1956
    report_every: int = 64
    tail_window: int = 512
    tolerance_s: float = 5e-4
    output_jsonl: str | None = None
    max_tail_slope_mib_per_sample: float | None = None


def _read_proc_status() -> dict[str, int]:
    fields: dict[str, int] = {}
    with open("/proc/self/status") as status_file:
        for line in status_file:
            name, separator, value = line.partition(":")
            if not separator:
                continue
            parts = value.split()
            if parts and parts[0].isdigit():
                multiplier = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
                fields[name] = int(parts[0]) * multiplier
    return fields


def _read_first_int(paths: tuple[Path, ...]) -> int | None:
    for path in paths:
        if path.exists():
            value = path.read_text().strip()
            if value.isdigit():
                return int(value)
    return None


def _read_cgroup_memory_stat() -> dict[str, int]:
    paths = (
        Path("/sys/fs/cgroup/memory.stat"),
        Path("/sys/fs/cgroup/memory/memory.stat"),
    )
    for path in paths:
        if not path.exists():
            continue
        stats: dict[str, int] = {}
        for line in path.read_text().splitlines():
            name, value = line.split(maxsplit=1)
            if value.isdigit():
                stats[name] = int(value)
        return stats
    return {}


def _gib(value: int | None) -> float | None:
    return None if value is None else value / GIB


def _format_gib(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _collect_metrics(*, sample: int, elapsed_s: float, label: str) -> dict[str, object]:
    proc_status = _read_proc_status()
    cgroup_stats = _read_cgroup_memory_stat()
    cgroup_current = _read_first_int(
        (
            Path("/sys/fs/cgroup/memory.current"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        )
    )

    metrics: dict[str, object] = {
        "label": label,
        "sample": sample,
        "elapsed_s": elapsed_s,
        "samples_per_s": sample / elapsed_s if sample > 0 and elapsed_s > 0 else None,
        "rss_gib": _gib(proc_status.get("VmRSS")),
        "rss_anon_gib": _gib(proc_status.get("RssAnon")),
        "rss_file_gib": _gib(proc_status.get("RssFile")),
        "peak_rss_gib": _gib(proc_status.get("VmHWM")),
        "threads": proc_status.get("Threads"),
        "file_descriptors": len(tuple(Path("/proc/self/fd").iterdir())),
        "cgroup_current_gib": _gib(cgroup_current),
        "cgroup_anon_gib": _gib(cgroup_stats.get("anon")),
        "cgroup_file_gib": _gib(cgroup_stats.get("file")),
    }
    print(
        f"{label}: sample={sample} "
        f"rss={_format_gib(metrics['rss_gib'])} GiB "
        f"rss_anon={_format_gib(metrics['rss_anon_gib'])} GiB "
        f"cgroup={_format_gib(metrics['cgroup_current_gib'])} GiB "
        f"cgroup_anon={_format_gib(metrics['cgroup_anon_gib'])} GiB "
        f"threads={metrics['threads']} fds={metrics['file_descriptors']} "
        f"rate={metrics['samples_per_s'] or 0.0:.2f} samples/s",
        flush=True,
    )
    return metrics


def _linear_slope_mib_per_sample(records: list[dict[str, object]], key: str, tail_window: int) -> float:
    final_sample = int(records[-1]["sample"])
    first_sample = max(0, final_sample - tail_window)
    points = [
        (float(record["sample"]), float(record[key]))
        for record in records
        if int(record["sample"]) >= first_sample and record[key] is not None
    ]
    if len(points) < 2:
        raise ValueError("At least two memory reports are required to compute a tail slope")

    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0:
        return 0.0
    slope_gib = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return slope_gib * 1024


def _select_video_keys(meta: LeRobotDatasetMetadata, mode: str) -> list[str]:
    all_keys = list(meta.video_keys)
    depth_keys = set(meta.depth_keys)
    rgb_keys = [key for key in all_keys if key not in depth_keys]
    return {
        "none": [],
        "rgb": rgb_keys,
        "depth": [key for key in all_keys if key in depth_keys],
        "rgbd": all_keys,
    }[mode]


def main(args: Args) -> None:
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.report_every <= 0:
        raise ValueError("report_every must be positive")
    if args.tail_window <= 0:
        raise ValueError("tail_window must be positive")

    dataset_root = args.dataset_root or os.environ.get("B1K_DATASET_ROOT")
    if dataset_root is None:
        raise ValueError("Set --dataset-root or B1K_DATASET_ROOT")

    meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=dataset_root)
    selected_keys = _select_video_keys(meta, args.mode)
    supports_video_keys = "video_keys" in inspect.signature(LeRobotDataset.__init__).parameters
    if not supports_video_keys and args.mode != "rgbd":
        raise RuntimeError(f"Installed LeRobot does not support video_keys, so mode={args.mode!r} cannot be selected")

    print(f"repo_id={args.repo_id}", flush=True)
    print(f"dataset_root={dataset_root}", flush=True)
    print(f"episode={args.episode} mode={args.mode} pattern={args.pattern}", flush=True)
    print(f"selected_video_keys={selected_keys}", flush=True)

    output_file = None
    if args.output_jsonl is not None:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("w", buffering=1)

    records: list[dict[str, object]] = []

    def record(*, sample: int, elapsed_s: float, label: str) -> None:
        metrics = _collect_metrics(sample=sample, elapsed_s=elapsed_s, label=label)
        records.append(metrics)
        if output_file is not None:
            output_file.write(json.dumps(metrics, sort_keys=True) + "\n")

    record(sample=0, elapsed_s=0.0, label="before_init")
    dataset_kwargs: dict[str, object] = {}
    if supports_video_keys:
        dataset_kwargs["video_keys"] = selected_keys
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        episodes=[args.episode],
        tolerance_s=args.tolerance_s,
        **dataset_kwargs,
    )
    record(sample=0, elapsed_s=0.0, label="after_init")

    if len(dataset) == 0:
        raise RuntimeError(f"Episode {args.episode} contains no samples")

    read_started = time.perf_counter()
    for sample in range(1, args.samples + 1):
        index = 0 if args.pattern == "repeat" else (sample - 1) % len(dataset)
        item = dataset[index]
        del item

        should_report = sample <= 16 and sample & (sample - 1) == 0
        should_report |= sample % args.report_every == 0 or sample == args.samples
        if should_report:
            record(sample=sample, elapsed_s=time.perf_counter() - read_started, label="reading")

    rss_slope = _linear_slope_mib_per_sample(records, "rss_gib", args.tail_window)
    anon_slope = _linear_slope_mib_per_sample(records, "rss_anon_gib", args.tail_window)
    print(f"tail_rss_slope_mib_per_sample={rss_slope:.6f}", flush=True)
    print(f"tail_anon_slope_mib_per_sample={anon_slope:.6f}", flush=True)

    del dataset
    gc.collect()
    record(
        sample=args.samples,
        elapsed_s=time.perf_counter() - read_started,
        label="after_gc",
    )
    if output_file is not None:
        output_file.close()

    if args.max_tail_slope_mib_per_sample is not None and rss_slope > args.max_tail_slope_mib_per_sample:
        raise SystemExit(
            f"Tail RSS slope {rss_slope:.6f} MiB/sample exceeds limit {args.max_tail_slope_mib_per_sample:.6f}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
