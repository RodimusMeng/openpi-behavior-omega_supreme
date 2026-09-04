"""Audit temporal alignment between raw BEHAVIOR absolute actions and robot state.

This script intentionally bypasses image loading and model inference.  It reads the
same LeRobot action/state fields used by the B1K training config and measures, in
raw controller units, whether the first action target of a training chunk is close
to the current compact proprio state or to a future state.

The main use is diagnosing receding-horizon seams for absolute-action checkpoints:
if training action[0] is normally close to state_t but fresh policy chunks start far
from state_t, the discontinuity is a deployment/model-sampling problem rather than
an intrinsic property of the demonstration action semantics.
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


def _parse_args() -> argparse.Namespace:
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


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_int(value: Any) -> int:
    arr = _np(value)
    return int(arr.reshape(-1)[0])


def _episode_index(sample: dict[str, Any]) -> int:
    for key in ("episode_index", "episode/index"):
        if key in sample:
            return _scalar_int(sample[key])
    raise KeyError(
        "LeRobot sample has no episode index key; available keys: "
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
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.max_state_lag < 1:
        raise ValueError("--max-state-lag must be >= 1")

    train_config = _config.get_config(args.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    robot = ROBOT_REGISTRY[args.robot]

    if isinstance(data_config.repo_id, list) or data_config.repo_id is None:
        raise ValueError("This audit currently expects one concrete B1K repo_id")
    if data_config.dataset_root is None:
        raise ValueError("B1K data config has no local dataset_root")

    metadata = _lerobot_compat.LeRobotDatasetMetadata(
        repo_id=data_config.repo_id,
        root=data_config.dataset_root,
    )
    fps = float(metadata.fps)
    horizon = int(train_config.model.action_horizon)

    # Use the same dataset constructor contract as create_b1k_dataset, but suppress
    # all videos so this becomes a cheap numeric audit.
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
    if dataset_len <= args.max_state_lag + 1:
        raise RuntimeError(f"Dataset is unexpectedly short: {dataset_len}")

    rng = np.random.default_rng(args.seed)
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    processed_indices: list[int] = []
    sampled_episodes: set[int] = set()

    # Episode boundaries make some random indices unusable.  Oversample attempts
    # rather than assuming contiguous dataset indices always stay in one episode.
    attempts = 0
    max_attempts = max(args.max_frames * 20, 10_000)
    seen: set[int] = set()

    while len(processed_indices) < args.max_frames and attempts < max_attempts:
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

        raw_state = _np(current["observation.state"])
        state0 = np.asarray(extract_compact_state(raw_state, robot), dtype=np.float64)
        actions = np.asarray(_np(current[robot.action_key]), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] < 2 or actions.shape[1] < robot.action_dim:
            raise RuntimeError(
                f"Unexpected action chunk shape at dataset index {index}: {actions.shape}"
            )
        actions = actions[:, : robot.action_dim]
        action0 = actions[0]
        action1 = actions[1]

        next_actions = np.asarray(_np(next_sample[robot.action_key]), dtype=np.float64)
        if next_actions.ndim != 2:
            raise RuntimeError(
                f"Unexpected next action chunk shape at dataset index {index + 1}: {next_actions.shape}"
            )
        next_action0 = next_actions[0, : robot.action_dim]
        state1 = np.asarray(
            extract_compact_state(_np(next_sample["observation.state"]), robot),
            dtype=np.float64,
        )

        # Core same-time and one-step quantities.
        _append_group_metrics(metrics, "action0_minus_state_t", action0 - state0)
        _append_group_metrics(metrics, "action0_minus_state_t+1", action0 - state1)
        _append_group_metrics(metrics, "action1_minus_action0", action1 - action0)
        _append_group_metrics(metrics, "state_t+1_minus_state_t", state1 - state0)
        _append_group_metrics(metrics, "action1_minus_state_t+1", action1 - state1)
        _append_group_metrics(metrics, "chunk_overlap_action1_vs_next_action0", action1 - next_action0)

        # Lag scan.  Two complementary measurements are useful:
        # 1) action0 vs state_{t+k}: which realized state best matches the current command?
        # 2) action[k] vs state_{t+k}: are command/state streams aligned pointwise?
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

        processed_indices.append(index)
        sampled_episodes.add(episode)

    if not processed_indices:
        raise RuntimeError("No valid non-terminal frames were collected")

    report = {
        "schema_version": "b1k_absolute_action_alignment_v1",
        "config": args.config,
        "robot": args.robot,
        "repo_id": data_config.repo_id,
        "dataset_root": str(data_config.dataset_root),
        "fps": fps,
        "action_horizon": horizon,
        "dataset_len": int(dataset_len),
        "sample_count": int(len(processed_indices)),
        "episode_count": int(len(sampled_episodes)),
        "seed": int(args.seed),
        "metrics": {
            metric: {group: _summary(values) for group, values in by_group.items()}
            for metric, by_group in metrics.items()
        },
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("===== B1K ABSOLUTE ACTION ALIGNMENT =====")
    print(f"config        : {args.config}")
    print(f"dataset       : {data_config.dataset_root}")
    print(f"fps           : {fps:g}")
    print(f"sample_count  : {len(processed_indices)}")
    print(f"episode_count : {len(sampled_episodes)}")
    print()

    important = [
        "action0_minus_state_t",
        "action0_minus_state_t+1",
        "action1_minus_action0",
        "state_t+1_minus_state_t",
        "action1_minus_state_t+1",
        "chunk_overlap_action1_vs_next_action0",
    ]
    print("===== JOINT SUMMARY =====")
    for metric in important:
        summary = report["metrics"][metric]["joints"]
        print(
            f"{metric:>38}: "
            f"median={summary['median']:.6f} "
            f"p90={summary['p90']:.6f} "
            f"p99={summary['p99']:.6f} "
            f"max={summary['max']:.6f}"
        )

    print("\n===== ACTION0 -> FUTURE STATE LAG SCAN (JOINTS) =====")
    for lag in range(args.max_state_lag + 1):
        metric = f"action0_minus_state_t+{lag}"
        summary = report["metrics"].get(metric, {}).get("joints")
        if not summary or not summary.get("count"):
            continue
        print(
            f"lag={lag:2d} ({lag / fps:.4f}s): "
            f"median={summary['median']:.6f} "
            f"p90={summary['p90']:.6f}"
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
