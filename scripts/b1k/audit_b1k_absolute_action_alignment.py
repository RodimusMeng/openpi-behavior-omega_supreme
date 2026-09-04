"""Audit alignment between BEHAVIOR absolute actions and compact robot state.

This is a numeric-only baseline diagnostic. It uses the exact B1K dataset config,
loads no videos, and measures whether action[0] is naturally offset from state_t in
the demonstrations. That distinguishes a data/action-semantics issue from a fresh
replanning discontinuity at inference time.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.configs.robots import ROBOT_REGISTRY
from openpi.configs.robots.base_config import extract_compact_state
from openpi.training import config as _config
from openpi.training import lerobot_compat as _lerobot_compat


GROUPS = {
    "joints": np.asarray(list(range(3, 14)) + list(range(15, 22)), dtype=int),
    "torso": np.arange(3, 7, dtype=int),
    "left_arm": np.arange(7, 14, dtype=int),
    "right_arm": np.arange(15, 22, dtype=int),
    "base": np.arange(0, 3, dtype=int),
    "grippers": np.asarray([14, 22], dtype=int),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_b1k_full_abs")
    parser.add_argument("--robot", default="b1k/R1Pro")
    parser.add_argument("--max-frames", type=int, default=20_000)
    parser.add_argument("--max-state-lag", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        default="outputs/debug/b1k_abs_action_alignment.json",
    )
    return parser.parse_args()


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_int(value: Any) -> int:
    return int(as_numpy(value).reshape(-1)[0])


def episode_index(sample: dict[str, Any]) -> int:
    for key in ("episode_index", "episode/index"):
        if key in sample:
            return scalar_int(sample[key])
    raise KeyError(
        "No episode index in LeRobot sample; keys=" + ",".join(sorted(sample))
    )


def rms(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    return float(np.sqrt(np.mean(vector * vector)))


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def append_group_metrics(
    metrics: dict[str, dict[str, list[float]]],
    name: str,
    vector: np.ndarray,
) -> None:
    for group, indices in GROUPS.items():
        metrics[name][group].append(rms(np.asarray(vector)[indices]))


def main() -> None:
    args = parse_args()
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.max_state_lag < 1:
        raise ValueError("--max-state-lag must be >= 1")

    train_config = _config.get_config(args.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    robot = ROBOT_REGISTRY[args.robot]

    if data_config.repo_id is None or isinstance(data_config.repo_id, list):
        raise ValueError("This audit expects one concrete B1K repo_id")
    if data_config.dataset_root is None:
        raise ValueError("B1K config has no dataset_root")

    metadata = _lerobot_compat.LeRobotDatasetMetadata(
        repo_id=data_config.repo_id,
        root=data_config.dataset_root,
    )
    fps = float(metadata.fps)
    horizon = int(train_config.model.action_horizon)

    # Match create_b1k_dataset, but explicitly disable videos.
    dataset_kwargs = dict(data_config.dataset_kwargs)
    dataset_kwargs["video_keys"] = []
    dataset = data_config.data_cls(
        root=data_config.dataset_root,
        repo_id=data_config.repo_id,
        delta_timestamps={
            robot.action_key: [step / fps for step in range(horizon)],
        },
        **dataset_kwargs,
    )

    dataset_len = len(dataset)
    rng = np.random.default_rng(args.seed)
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sampled_episodes: set[int] = set()
    sampled_indices: list[int] = []
    seen: set[int] = set()

    attempts = 0
    max_attempts = max(args.max_frames * 20, 10_000)
    while len(sampled_indices) < args.max_frames and attempts < max_attempts:
        attempts += 1
        index = int(rng.integers(0, dataset_len - args.max_state_lag))
        if index in seen:
            continue
        seen.add(index)

        current = dataset[index]
        nxt = dataset[index + 1]
        episode = episode_index(current)
        if episode_index(nxt) != episode:
            continue

        state0 = np.asarray(
            extract_compact_state(as_numpy(current["observation.state"]), robot),
            dtype=np.float64,
        )
        state1 = np.asarray(
            extract_compact_state(as_numpy(nxt["observation.state"]), robot),
            dtype=np.float64,
        )
        actions = np.asarray(as_numpy(current[robot.action_key]), dtype=np.float64)
        next_actions = np.asarray(as_numpy(nxt[robot.action_key]), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] < 2 or actions.shape[1] < robot.action_dim:
            raise RuntimeError(f"Bad action shape at index {index}: {actions.shape}")
        if next_actions.ndim != 2 or next_actions.shape[1] < robot.action_dim:
            raise RuntimeError(f"Bad next action shape at index {index + 1}: {next_actions.shape}")

        actions = actions[:, : robot.action_dim]
        next_actions = next_actions[:, : robot.action_dim]
        action0, action1 = actions[0], actions[1]
        next_action0 = next_actions[0]

        append_group_metrics(metrics, "action0_minus_state_t", action0 - state0)
        append_group_metrics(metrics, "action0_minus_state_t+1", action0 - state1)
        append_group_metrics(metrics, "action1_minus_action0", action1 - action0)
        append_group_metrics(metrics, "state_t+1_minus_state_t", state1 - state0)
        append_group_metrics(metrics, "action1_minus_state_t+1", action1 - state1)
        append_group_metrics(metrics, "chunk_overlap_action1_vs_next_action0", action1 - next_action0)

        for lag in range(args.max_state_lag + 1):
            sample = dataset[index + lag]
            if episode_index(sample) != episode:
                break
            lag_state = np.asarray(
                extract_compact_state(as_numpy(sample["observation.state"]), robot),
                dtype=np.float64,
            )
            append_group_metrics(
                metrics,
                f"action0_minus_state_t+{lag}",
                action0 - lag_state,
            )
            if lag < len(actions):
                append_group_metrics(
                    metrics,
                    f"action{lag}_minus_state_t+{lag}",
                    actions[lag] - lag_state,
                )

        sampled_indices.append(index)
        sampled_episodes.add(episode)

    if not sampled_indices:
        raise RuntimeError("Collected zero valid frames")

    report = {
        "schema_version": "b1k_absolute_action_alignment_v1",
        "config": args.config,
        "robot": args.robot,
        "repo_id": data_config.repo_id,
        "dataset_root": str(data_config.dataset_root),
        "fps": fps,
        "action_horizon": horizon,
        "dataset_len": int(dataset_len),
        "sample_count": len(sampled_indices),
        "episode_count": len(sampled_episodes),
        "seed": args.seed,
        "metrics": {
            name: {group: summarize(values) for group, values in groups.items()}
            for name, groups in metrics.items()
        },
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("===== B1K ABSOLUTE ACTION ALIGNMENT =====")
    print(f"config        : {args.config}")
    print(f"dataset       : {data_config.dataset_root}")
    print(f"fps           : {fps:g}")
    print(f"sample_count  : {len(sampled_indices)}")
    print(f"episode_count : {len(sampled_episodes)}")

    print("\n===== JOINT SUMMARY =====")
    for name in (
        "action0_minus_state_t",
        "action0_minus_state_t+1",
        "action1_minus_action0",
        "state_t+1_minus_state_t",
        "action1_minus_state_t+1",
        "chunk_overlap_action1_vs_next_action0",
    ):
        summary = report["metrics"][name]["joints"]
        print(
            f"{name:>38}: median={summary['median']:.6f} "
            f"p90={summary['p90']:.6f} p99={summary['p99']:.6f} max={summary['max']:.6f}"
        )

    print("\n===== ACTION0 -> FUTURE STATE LAG SCAN (JOINTS) =====")
    for lag in range(args.max_state_lag + 1):
        summary = report["metrics"].get(f"action0_minus_state_t+{lag}", {}).get("joints")
        if summary and summary.get("count"):
            print(
                f"lag={lag:2d} ({lag / fps:.4f}s): "
                f"median={summary['median']:.6f} p90={summary['p90']:.6f}"
            )

    print("\n===== COMPONENT action0-state_t =====")
    for group in ("torso", "left_arm", "right_arm", "base", "grippers"):
        summary = report["metrics"]["action0_minus_state_t"][group]
        print(
            f"{group:>10}: median={summary['median']:.6f} "
            f"p90={summary['p90']:.6f} max={summary['max']:.6f}"
        )

    print(f"\nJSON: {output_path}")


if __name__ == "__main__":
    main()
