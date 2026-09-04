"""Audit LeRobot video timestamp deviations without decoding RGB arrays.

The audit mirrors the PyAV timestamp path used by ``LeRobotDataset``:

1. add the episode's per-video ``from_timestamp`` to the dataset row timestamp;
2. seek to the preceding keyframe;
3. decode through the largest requested timestamp; and
4. compare each query with the nearest decoded frame timestamp using float32,
   matching LeRobot's current tolerance guard.

This is intentionally a diagnostic tool. It never skips samples and never changes
the training tolerance.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import dataclasses
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

DEFAULT_VIDEO_KEY = "observation.rgb.left_realsense_link_camera_0"


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str = "b1k-v3again"
    dataset_root: str | None = None
    video_key: str = DEFAULT_VIDEO_KEY
    # Match every episode whose relative video path contains this text.
    video_path_contains: str | None = None
    # Explicit episode selection. Mutually exclusive with episode_count/all_episodes/path matching.
    episode_indices: tuple[int, ...] = ()
    # Deterministically sample this many episodes from the full metadata.
    episode_count: int | None = None
    all_episodes: bool = False
    # Deterministically sample this many numeric rows after episode selection. None scans all selected rows.
    samples: int | None = None
    seed: int = 42
    tolerance_s: float = 5e-4
    worst_count: int = 20
    output_json: str | None = None


@dataclasses.dataclass(frozen=True)
class TimestampQuery:
    episode_index: int
    frame_index: int
    local_timestamp_s: float
    shifted_timestamp_s: float
    video_path: str


@dataclasses.dataclass(frozen=True)
class TimestampMatch:
    query: TimestampQuery
    nearest_timestamp_s: float
    error_s: float


def _as_scalar(value: Any) -> int | float:
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalize_path_fragment(value: str) -> str:
    return value.replace("\\", "/")


def _nearest_timestamp_matches(
    queries: Sequence[TimestampQuery], decoded_timestamps_s: Sequence[float], *, dtype: Any = np.float32
) -> list[TimestampMatch]:
    """Return nearest timestamp matches using the requested comparison dtype."""
    if not queries:
        return []
    if not decoded_timestamps_s:
        raise ValueError("decoded_timestamps_s must not be empty")

    decoded = np.asarray(decoded_timestamps_s, dtype=dtype)
    if np.any(decoded[1:] < decoded[:-1]):
        raise ValueError("decoded frame timestamps must be nondecreasing")

    result: list[TimestampMatch] = []
    for query in queries:
        query_ts = dtype(query.shifted_timestamp_s)
        right = int(np.searchsorted(decoded, query_ts, side="left"))
        candidates = []
        if right < decoded.size:
            candidates.append(right)
        if right > 0:
            candidates.append(right - 1)
        nearest = min(candidates, key=lambda index: (abs(float(decoded[index] - query_ts)), index))
        error_s = abs(float(decoded[nearest] - query_ts))
        result.append(
            TimestampMatch(
                query=query,
                nearest_timestamp_s=float(decoded[nearest]),
                error_s=error_s,
            )
        )
    return result


def _summarize_errors(errors_s: Sequence[float], *, tolerance_s: float) -> dict[str, Any]:
    if not errors_s:
        raise ValueError("errors_s must not be empty")
    values = np.asarray(errors_s, dtype=np.float64)
    thresholds_ms = (0.5, 1.0, 2.0, 5.0)
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean() * 1_000),
        "p50_ms": float(np.quantile(values, 0.5) * 1_000),
        "p95_ms": float(np.quantile(values, 0.95) * 1_000),
        "p99_ms": float(np.quantile(values, 0.99) * 1_000),
        "p99_9_ms": float(np.quantile(values, 0.999) * 1_000),
        "max_ms": float(values.max() * 1_000),
        "tolerance_ms": tolerance_s * 1_000,
        "violates_current_tolerance": int(np.count_nonzero(values >= tolerance_s)),
        "counts_strictly_above_ms": {
            str(threshold): int(np.count_nonzero(values > threshold / 1_000)) for threshold in thresholds_ms
        },
    }


def _select_episode_indices(args: Args, meta: Any) -> list[int]:
    selectors = sum(
        (
            args.video_path_contains is not None,
            bool(args.episode_indices),
            args.episode_count is not None,
            args.all_episodes,
        )
    )
    if selectors != 1:
        raise ValueError(
            "Select exactly one of --video-path-contains, --episode-indices, --episode-count, or --all-episodes"
        )

    total_episodes = int(meta.total_episodes)
    if args.video_key not in meta.video_keys:
        raise ValueError(f"Unknown video key {args.video_key!r}; available keys: {list(meta.video_keys)}")

    if args.video_path_contains is not None:
        needle = _normalize_path_fragment(args.video_path_contains)
        selected = [
            episode_index
            for episode_index in range(total_episodes)
            if needle in _normalize_path_fragment(str(meta.get_video_file_path(episode_index, args.video_key)))
        ]
        if not selected:
            raise ValueError(f"No episode uses a {args.video_key!r} video path containing {needle!r}")
        return selected

    if args.episode_indices:
        selected = list(args.episode_indices)
        invalid = [index for index in selected if index < 0 or index >= total_episodes]
        if invalid:
            raise ValueError(f"Episode indices out of range [0, {total_episodes}): {invalid}")
        if len(selected) != len(set(selected)):
            raise ValueError("episode_indices must not contain duplicates")
        return sorted(selected)

    if args.episode_count is not None:
        if not 0 < args.episode_count <= total_episodes:
            raise ValueError(f"episode_count must be in [1, {total_episodes}], got {args.episode_count}")
        return sorted(random.Random(args.seed).sample(range(total_episodes), args.episode_count))

    return list(range(total_episodes))


def _collect_queries(args: Args, meta: Any, episode_indices: Sequence[int], dataset: Any) -> list[TimestampQuery]:
    row_count = len(dataset.hf_dataset)
    if row_count == 0:
        raise ValueError("Selected episodes contain no numeric rows")
    if args.samples is None:
        relative_indices: Sequence[int] = range(row_count)
    else:
        if args.samples <= 0:
            raise ValueError("samples must be positive")
        relative_indices = random.Random(args.seed).sample(range(row_count), min(args.samples, row_count))

    selected_set = set(episode_indices)
    queries = []
    for relative_index in relative_indices:
        row = dataset.hf_dataset[int(relative_index)]
        episode_index = int(_as_scalar(row["episode_index"]))
        if episode_index not in selected_set:
            raise RuntimeError(f"Loaded unexpected episode {episode_index}; selected={sorted(selected_set)}")
        frame_index = int(_as_scalar(row["frame_index"]))
        local_timestamp_s = float(_as_scalar(row["timestamp"]))
        episode = meta.episodes[episode_index]
        shifted_timestamp_s = float(episode[f"videos/{args.video_key}/from_timestamp"]) + local_timestamp_s
        video_path = Path(args.dataset_root) / meta.get_video_file_path(episode_index, args.video_key)
        queries.append(
            TimestampQuery(
                episode_index=episode_index,
                frame_index=frame_index,
                local_timestamp_s=local_timestamp_s,
                shifted_timestamp_s=shifted_timestamp_s,
                video_path=str(video_path),
            )
        )
    return queries


def _decode_timestamp_span(video_path: str, queries: Sequence[TimestampQuery]) -> list[float]:
    """Decode only frame PTS values over the same seek span used by the PyAV reader."""
    import av  # noqa: PLC0415 - keep pure planning/statistics tests independent from FFmpeg.

    first_ts = min(query.shifted_timestamp_s for query in queries)
    last_ts = max(query.shifted_timestamp_s for query in queries)
    loaded_timestamps = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        container.seek(int(first_ts * av.time_base), backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * stream.time_base)
            loaded_timestamps.append(current_ts)
            if current_ts >= last_ts:
                break
    if not loaded_timestamps:
        raise RuntimeError(f"No frame timestamps decoded from {video_path} over [{first_ts}, {last_ts}]")
    return loaded_timestamps


def _run(args: Args) -> dict[str, Any]:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    dataset_root = args.dataset_root or os.environ.get("B1K_DATASET_ROOT")
    if dataset_root is None:
        raise ValueError("Set --dataset-root or B1K_DATASET_ROOT")
    args = dataclasses.replace(args, dataset_root=dataset_root)

    meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=dataset_root)
    episode_indices = _select_episode_indices(args, meta)
    if args.all_episodes and args.samples is None:
        raise ValueError("--all-episodes requires --samples to avoid an accidental full-dataset scan")

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        episodes=episode_indices,
        video_keys=[],
    )
    queries = _collect_queries(args, meta, episode_indices, dataset)
    grouped: dict[str, list[TimestampQuery]] = defaultdict(list)
    for query in queries:
        grouped[query.video_path].append(query)

    float32_matches = []
    float64_matches = []
    for ordinal, (video_path, video_queries) in enumerate(sorted(grouped.items()), start=1):
        print(
            f"[{ordinal}/{len(grouped)}] video={video_path} queries={len(video_queries)} "
            f"span={min(q.shifted_timestamp_s for q in video_queries):.6f}.."
            f"{max(q.shifted_timestamp_s for q in video_queries):.6f}",
            flush=True,
        )
        decoded_timestamps = _decode_timestamp_span(video_path, video_queries)
        float32_matches.extend(_nearest_timestamp_matches(video_queries, decoded_timestamps, dtype=np.float32))
        float64_matches.extend(_nearest_timestamp_matches(video_queries, decoded_timestamps, dtype=np.float64))

    float32_matches.sort(key=lambda match: match.error_s, reverse=True)
    float64_matches.sort(key=lambda match: match.error_s, reverse=True)

    def serialize_worst(matches: Sequence[TimestampMatch]) -> list[dict[str, Any]]:
        return [
            {
                "episode_index": match.query.episode_index,
                "frame_index": match.query.frame_index,
                "video_path": match.query.video_path,
                "local_timestamp_s": match.query.local_timestamp_s,
                "query_timestamp_s": match.query.shifted_timestamp_s,
                "nearest_timestamp_s": match.nearest_timestamp_s,
                "error_ms": match.error_s * 1_000,
            }
            for match in matches[: args.worst_count]
        ]

    result = {
        "repo_id": args.repo_id,
        "dataset_root": dataset_root,
        "video_key": args.video_key,
        "episode_indices": episode_indices,
        "video_count": len(grouped),
        "summary": {
            "current_float32_guard": _summarize_errors(
                [match.error_s for match in float32_matches], tolerance_s=args.tolerance_s
            ),
            "float64_reference": _summarize_errors(
                [match.error_s for match in float64_matches], tolerance_s=args.tolerance_s
            ),
        },
        "worst": {
            "current_float32_guard": serialize_worst(float32_matches),
            "float64_reference": serialize_worst(float64_matches),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main(args: Args) -> None:
    if args.tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")
    if args.worst_count < 0:
        raise ValueError("worst_count must be non-negative")
    result = _run(args)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
