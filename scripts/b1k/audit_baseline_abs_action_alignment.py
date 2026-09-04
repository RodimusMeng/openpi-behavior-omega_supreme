"""Audit raw B1K absolute-action/state temporal alignment for the baseline policy.

No WEAVER code is imported or used.  This is a numeric-only audit of the same
BEHAVIOR/LeRobot demonstrations used by the baseline pi0.5 policy.  It asks whether
a training chunk's first absolute joint target is normally close to the current
robot state, and whether any fixed temporal offset explains the action/state
relationship.
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
from openpi.training import lerobot_compat as _lerobot_compat


DEFAULT_DATASET_ROOT = "/inspire/dataset/behavior-1k-2026-challenge-demos/v3again"
GROUPS = {
    "joints": np.asarray(list(range(3, 14)) + list(range(15, 22)), dtype=int),
    "torso": np.arange(3, 7, dtype=int),
    "left_arm": np.arange(7, 14, dtype=int),
    "right_arm": np.arange(15, 22, dtype=int),
    "base": np.arange(0, 3, dtype=int),
    "grippers": np.asarray([14, 22], dtype=int),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="b1k-v3again")
    parser.add_argument("--robot", default="b1k/R1Pro")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=20_000)
    parser.add_argument("--max-state-lag", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        default="outputs/debug/baseline_abs_action_alignment.json",
    )
    return parser.parse_args()


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_int(value: Any) -> int:
    return int(_np(value).reshape(-1)[0])


def _episode_index(sample: dict[str, Any]) -> int:
    for key in ("episode_index", "episode/index"):
        if key in sample:
            return _scalar_int(sample[key])
    raise KeyError(
        "No episode index in LeRobot sample; available keys: "
        + ", ".join(sorted(sample))
    )


def _rms(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    return float(np.sqrt(np.mean(vector * vector)))


def _summary(values: list[float]) -> dict[str, float | int]:
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


def _append_group_metrics(
    store: dict[str, dict[str, list[float]]],
    metric: str,
    vector: np.ndarray,
) -> None:
    for group_name, indices in GROUPS.items():
        store[metric][group_name].append(_rms(np.asarray(vector)[indices]))


def main() -> None:
    args = _parse_args()
    if args.action_horizon < 2:
        raise ValueError("--action-horizon must be >= 2")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.max_state_lag < 1:
        raise ValueError("--max-state-lag must be >= 1")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    robot = ROBOT_REGISTRY[args.robot]
    metadata = _lerobot_compat.LeRobotDatasetMetadata(
        repo_id=args.repo_id,
        root=str(dataset_root),
    )
    fps = float(metadata.fps)

    # Same numeric temporal contract as the baseline B1K data loader, but no videos.
    dataset = _lerobot_compat.LeRobotDataset(
        root=str(dataset_root),
        repo_id=args.repo_id,
        delta_timestamps={
            robot.action_key: [step / fps for step in range(args.action_horizon)],
        },
        tolerance_s=5e-4,
        video_keys=[],
    )

    dataset_len = len(dataset)
    if dataset_len <= args.max_state_lag + 1:
        raise RuntimeError(f"Dataset is unexpectedly short: {dataset_len}")

    rng = np.random.default_rng(args.seed)
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sampled_episodes: set[int] = set()
    processed = 0
    attempts = 0
    max_attempts = max(args.max_frames * 20, 10_000)
    seen: set[int] = set()

    while processed < args.max_frames and attempts < max_attempts:
        attempts += 1
        index = int(rng.integers(0, dataset_len - args.max_state_lag))
        if index in seen:
            continue
        seen.add(index)

        current = dataset[index]
        next_sample = dataset[index + 1]
        episode = _episode_index(current)
        if _episode_index(next_sample) != episode:
            continue

        state0 = np.asarray(
            extract_compact_state(_np(current["observation.state"]), robot),
            dtype=np.float64,
        )
        state1 = np.asarray(
            extract_compact_state(_np(next_sample["observation.state"]), robot),
            dtype=np.float64,
        )
        actions = np.asarray(_np(current[robot.action_key]), dtype=np.float64)
        next_actions = np.asarray(_np(next_sample[robot.action_key]), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] < 2 or actions.shape[1] < robot.action_dim:
            raise RuntimeError(f"Unexpected action chunk shape at {index}: {actions.shape}")
        if next_actions.ndim != 2 or next_actions.shape[1] < robot.action_dim:
            raise RuntimeError(f"Unexpected next action chunk shape at {index + 1}: {next_actions.shape}")

        actions = actions[:, : robot.action_dim]
        next_actions = next_actions[:, : robot.action_dim]
        action0, action1 = actions[0], actions[1]
        next_action0 = next_actions[0]

        _append_group_metrics(metrics, "action0_minus_state_t", action0 - state0)
        _append_group_metrics(metrics, "action0_minus_state_t+1", action0 - state1)
        _append_group_metrics(metrics, "action1_minus_action0", action1 - action0)
        _append_group_metrics(metrics, "state_t+1_minus_state_t", state1 - state0)
        _append_group_metrics(metrics, "action1_minus_state_t+1", action1 - state1)
        _append_group_metrics(metrics, "action1_minus_next_action0", action1 - next_action0)

        # Lag scan: which future realized state best matches the current action0?
        for lag in range(args.max_state_lag + 1):
            lagged = dataset[index + lag]
            if _episode_index(lagged) != episode:
                break
            lagged_state = np.asarray(
                extract_compact_state(_np(lagged["observation.state"]), robot),
                dtype=np.float64,
            )
            _append_group_metrics(
                metrics,
                f"action0_minus_state_t+{lag}",
                action0 - lagged_state,
            )
            if lag < len(actions):
                _append_group_metrics(
                    metrics,
                    f"action{lag}_minus_state_t+{lag}",
                    actions[lag] - lagged_state,
                )

        sampled_episodes.add(episode)
        processed += 1

    if processed == 0:
        raise RuntimeError("No valid non-terminal frames were collected")

    report = {
        "schema_version": "baseline_b1k_absolute_action_alignment_v1",
        "repo_id": args.repo_id,
        "dataset_root": str(dataset_root),
        "robot": args.robot,
        "fps": fps,
        "action_horizon": args.action_horizon,
        "dataset_len": int(dataset_len),
        "sample_count": int(processed),
        "episode_count": int(len(sampled_episodes)),
        "seed": args.seed,
        "metrics": {
            metric: {group: _summary(values) for group, values in by_group.items()}
            for metric, by_group in metrics.items()
        },
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("===== BASELINE B1K ABSOLUTE ACTION ALIGNMENT =====")
    print(f"dataset       : {dataset_root}")
    print(f"fps           : {fps:g}")
    print(f"sample_count  : {processed}")
    print(f"episode_count : {len(sampled_episodes)}")

    print("\n===== JOINT SUMMARY =====")
    for metric in (
        "action0_minus_state_t",
        "action0_minus_state_t+1",
        "action1_minus_action0",
        "state_t+1_minus_state_t",
        "action1_minus_state_t+1",
        "action1_minus_next_action0",
    ):
        summary = report["metrics"][metric]["joints"]
        print(
            f"{metric:>34}: "
            f"median={summary['median']:.6f} "
            f"p90={summary['p90']:.6f} "
            f"p99={summary['p99']:.6f} "
            f"max={summary['max']:.6f}"
        )

    print("\n===== ACTION0 -> FUTURE STATE LAG SCAN (JOINTS) =====")
    for lag in range(args.max_state_lag + 1):
        summary = report["metrics"].get(f"action0_minus_state_t+{lag}", {}).get("joints")
        if not summary or not summary.get("count"):
            continue
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
