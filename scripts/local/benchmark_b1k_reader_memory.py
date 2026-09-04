"""Benchmark B1K video-reader memory without starting the OpenPI model stack."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Literal

from openpi.training.block_sampler import EpisodeRange

GIB = 1024**3


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str = "b1k-v3again"
    dataset_root: str | None = None
    episode: int = 0
    all_episodes: bool = False
    # Deterministically choose this many episodes instead of using ``episode``.
    episode_count: int | None = None
    mode: Literal["none", "rgb", "depth", "rgbd"] = "rgbd"
    video_backend: Literal["torchcodec", "pyav"] | None = None
    return_uint8: bool = True
    torchcodec_decoder_cache_size: int = 8
    pattern: Literal["sequential", "repeat", "random", "block-random"] = "sequential"
    # Consecutive samples kept together by block-random.
    block_size: int = 1
    # Canonical block used to choose an identical sample set for every block_size A/B.
    comparison_block_size: int = 16
    seed: int = 42
    samples: int = 1956
    report_every: int = 64
    tail_window: int = 512
    tolerance_s: float = 5e-4
    output_jsonl: str | None = None
    max_tail_slope_mib_per_sample: float | None = None


def _select_episode_indices(args: Args, meta: Any) -> list[int] | None:
    if args.all_episodes and args.episode_count is not None:
        raise ValueError("--all-episodes and --episode-count are mutually exclusive")

    total_episodes = int(meta.total_episodes)
    if args.episode_count is not None:
        if not 0 < args.episode_count <= total_episodes:
            raise ValueError(f"episode_count must be in [1, {total_episodes}], got {args.episode_count}")
        return sorted(random.Random(args.seed).sample(range(total_episodes), args.episode_count))
    if args.all_episodes:
        return None
    if not 0 <= args.episode < total_episodes:
        raise ValueError(f"episode must be in [0, {total_episodes}), got {args.episode}")
    return [args.episode]


def _episode_ranges(meta: Any, episode_indices: Sequence[int] | None) -> list[EpisodeRange]:
    selected = range(int(meta.total_episodes)) if episode_indices is None else episode_indices
    ranges: list[EpisodeRange] = []
    relative_start = 0
    for episode_index in selected:
        episode = meta.episodes[int(episode_index)]
        length = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
        if length <= 0:
            raise ValueError(f"Episode {episode_index} has invalid length {length}")
        ranges.append(EpisodeRange(int(episode_index), relative_start, relative_start + length))
        relative_start += length
    return ranges


def _build_block_random_plan(
    ranges: Sequence[EpisodeRange],
    *,
    samples: int,
    block_size: int,
    comparison_block_size: int,
    seed: int,
) -> list[int]:
    """Build a locality-aware plan whose sample set is identical across block sizes.

    Samples are first selected as complete ``comparison_block_size`` blocks. Each
    selected comparison block is then split into smaller blocks and only those
    blocks are shuffled. Consequently K=1/2/4/... changes order, not membership.
    """
    if block_size <= 0 or comparison_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if comparison_block_size % block_size != 0:
        raise ValueError("block_size must divide comparison_block_size")
    if samples % comparison_block_size != 0:
        raise ValueError("samples must be divisible by comparison_block_size for a fair A/B")

    comparison_blocks = [
        list(range(start, start + comparison_block_size))
        for episode_range in ranges
        for start in range(episode_range.start, episode_range.stop - comparison_block_size + 1, comparison_block_size)
    ]
    blocks_needed = samples // comparison_block_size
    if blocks_needed > len(comparison_blocks):
        eligible = len(comparison_blocks) * comparison_block_size
        raise ValueError(f"Requested {samples} samples, but only {eligible} fit complete comparison blocks")

    selection_rng = random.Random(seed)
    selection_rng.shuffle(comparison_blocks)
    selected = comparison_blocks[:blocks_needed]
    blocks = [
        comparison_block[offset : offset + block_size]
        for comparison_block in selected
        for offset in range(0, comparison_block_size, block_size)
    ]
    random.Random(seed ^ 0xB10C).shuffle(blocks)
    return [index for block in blocks for index in block]


def _plan_hash(indices: Sequence[int], *, sort_indices: bool) -> str:
    values = sorted(indices) if sort_indices else indices
    digest = hashlib.sha256()
    for index in values:
        digest.update(int(index).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _latency_summary(latencies_s: Sequence[float]) -> dict[str, float]:
    if not latencies_s:
        raise ValueError("At least one latency is required")
    ordered = sorted(float(value) for value in latencies_s)

    def percentile(q: float) -> float:
        position = q * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {
        "mean_s": statistics.fmean(ordered),
        "p50_s": percentile(0.50),
        "p95_s": percentile(0.95),
        "p99_s": percentile(0.99),
        "max_s": ordered[-1],
    }


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


def _select_video_keys(meta: Any, mode: str) -> list[str]:
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
    if args.torchcodec_decoder_cache_size <= 0:
        raise ValueError("torchcodec_decoder_cache_size must be positive")
    os.environ["LEROBOT_VIDEO_DECODER_CACHE_SIZE"] = str(args.torchcodec_decoder_cache_size)

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

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
    episode_indices = _select_episode_indices(args, meta)
    selected_keys = _select_video_keys(meta, args.mode)
    supports_video_keys = "video_keys" in inspect.signature(LeRobotDataset.__init__).parameters
    supports_return_uint8 = "return_uint8" in inspect.signature(LeRobotDataset.__init__).parameters
    if not supports_video_keys and args.mode != "rgbd":
        raise RuntimeError(f"Installed LeRobot does not support video_keys, so mode={args.mode!r} cannot be selected")
    if args.return_uint8 and not supports_return_uint8:
        raise RuntimeError("Installed LeRobot does not support return_uint8")

    print(f"repo_id={args.repo_id}", flush=True)
    print(f"dataset_root={dataset_root}", flush=True)
    episode_selection = "all" if episode_indices is None else ",".join(str(index) for index in episode_indices)
    print(
        f"episode={episode_selection} mode={args.mode} "
        f"video_backend={args.video_backend or 'auto'} return_uint8={args.return_uint8} pattern={args.pattern}",
        flush=True,
    )
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
    if supports_return_uint8:
        dataset_kwargs["return_uint8"] = args.return_uint8
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        episodes=episode_indices,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend,
        **dataset_kwargs,
    )
    record(sample=0, elapsed_s=0.0, label="after_init")

    if len(dataset) == 0:
        raise RuntimeError(f"Episode selection {episode_selection} contains no samples")

    indices: list[int] | None = None
    if args.pattern == "random":
        if args.samples > len(dataset):
            raise ValueError(
                f"random pattern requested {args.samples} unique samples from dataset length {len(dataset)}"
            )
        indices = random.Random(args.seed).sample(range(len(dataset)), k=args.samples)
    elif args.pattern == "block-random":
        ranges = _episode_ranges(meta, episode_indices)
        if sum(item.stop - item.start for item in ranges) != len(dataset):
            raise RuntimeError("Episode metadata lengths do not match the loaded dataset length")
        indices = _build_block_random_plan(
            ranges,
            samples=args.samples,
            block_size=args.block_size,
            comparison_block_size=args.comparison_block_size,
            seed=args.seed,
        )

    if indices is not None:
        print(f"sequence_sha256={_plan_hash(indices, sort_indices=False)}", flush=True)
        print(f"sample_set_sha256={_plan_hash(indices, sort_indices=True)}", flush=True)

    read_started = time.perf_counter()
    latencies_s: list[float] = []
    for sample in range(1, args.samples + 1):
        if args.pattern == "repeat":
            index = 0
        elif indices is not None:
            index = indices[sample - 1]
        else:
            index = (sample - 1) % len(dataset)
        sample_started = time.perf_counter()
        item = dataset[index]
        latencies_s.append(time.perf_counter() - sample_started)
        del item

        should_report = sample <= 16 and sample & (sample - 1) == 0
        should_report |= sample % args.report_every == 0 or sample == args.samples
        if should_report:
            record(sample=sample, elapsed_s=time.perf_counter() - read_started, label="reading")

    rss_slope = _linear_slope_mib_per_sample(records, "rss_gib", args.tail_window)
    anon_slope = _linear_slope_mib_per_sample(records, "rss_anon_gib", args.tail_window)
    total_elapsed_s = time.perf_counter() - read_started
    tail_latencies = latencies_s[-min(args.tail_window, len(latencies_s)) :]
    summary: dict[str, object] = {
        "label": "summary",
        "samples": args.samples,
        "elapsed_s": total_elapsed_s,
        "samples_per_s": args.samples / total_elapsed_s,
        "latency": _latency_summary(latencies_s),
        "tail_samples": len(tail_latencies),
        "tail_latency": _latency_summary(tail_latencies),
        "tail_samples_per_s": len(tail_latencies) / sum(tail_latencies),
        "tail_rss_slope_mib_per_sample": rss_slope,
        "tail_anon_slope_mib_per_sample": anon_slope,
    }
    if indices is not None:
        summary["sequence_sha256"] = _plan_hash(indices, sort_indices=False)
        summary["sample_set_sha256"] = _plan_hash(indices, sort_indices=True)
    print(f"tail_rss_slope_mib_per_sample={rss_slope:.6f}", flush=True)
    print(f"tail_anon_slope_mib_per_sample={anon_slope:.6f}", flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if output_file is not None:
        output_file.write(json.dumps(summary, sort_keys=True) + "\n")

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
    import tyro

    main(tyro.cli(Args))
