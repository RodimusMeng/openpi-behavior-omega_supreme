"""Episode-level adapters from the full B1K LeRobot dataset to WEAVER.

The adapter deliberately keeps two action representations separate:

* ``b1k_policy_mixed_v1`` is the 23-D command recorded in the source dataset.
* ``r1pro_compact_state_delta_v1`` is the observed 5 Hz compact-state change
  used as the first WEAVER dynamics-training representation.

They are not interchangeable.  In particular, the first three compact-state
coordinates are base velocity, whereas the first three policy-action
coordinates are base velocity commands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import math
from typing import Any

import numpy as np

from openpi.weaver import reward_contract as _reward_contract
from openpi.weaver.b1k_contract import B1KWeaverContract
from openpi.weaver.b1k_contract import build_r1pro_contract
from openpi.weaver.b1k_contract import world_model_frame_indices

SCHEMA_VERSION = "b1k_weaver_episode_v6"
POLICY_ACTION_REPRESENTATION = "b1k_policy_mixed_v1"
STATE_DELTA_ACTION_REPRESENTATION = "r1pro_compact_state_delta_v1"
TRAINING_ACTION_CONTRACT = {
    "representation": STATE_DELTA_ACTION_REPRESENTATION,
    "dim": 23,
    "causal": False,
    "online_steering_compatible": False,
    "provenance": "observed_next_state_difference",
}
SOURCE_POLICY_ACTION_CONTRACT = {
    "representation": POLICY_ACTION_REPRESENTATION,
    "dim": 23,
    "full_rate_preserved": True,
    "five_hz_aggregation": "undefined",
}
ONLINE_STEERING_CONTRACT = {
    "compatible": False,
    "reason": "offline_state_delta_depends_on_observed_next_state",
    "required_next_step": "define_and_validate_causal_30hz_policy_action_aggregation",
}
EPISODE_SEMANTIC_FIELDS = (
    "schema_version",
    "dataset",
    "episode_index",
    "split",
    "prompt",
    "target_hz",
    "sampled_count",
    "source_frame_indices_sha256",
    "timestamps_sha256",
    "max_timing_error_s",
    "views",
    "state",
    "training_action",
    "source_policy_action",
    "online_steering",
    "reward",
    "reward_available",
)


def _readonly_numeric(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


def array_sha256(value: np.ndarray) -> str:
    """Hash array semantics, including dtype and shape."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible semantic configuration deterministically."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_episode_manifest(manifest: Mapping[str, Any]) -> str:
    """Verify and return the episode's immutable semantic fingerprint."""

    try:
        semantic = {key: manifest[key] for key in EPISODE_SEMANTIC_FIELDS}
    except KeyError as exc:
        raise ValueError(f"Episode manifest is missing semantic field {exc.args[0]!r}") from exc
    expected = manifest.get("semantic_sha256")
    actual = semantic_sha256(semantic)
    if expected != actual:
        raise ValueError(f"Episode semantic SHA-256 mismatch: declared={expected!r}, actual={actual}")
    return actual


def selected_video_features(
    contract: B1KWeaverContract | None = None,
    *,
    include_depth: bool = False,
) -> tuple[str, ...]:
    """Return only the full-dataset video features required by WEAVER.

    The released transfer smoke keeps the checkpoint's two-view order:
    ``wrist_left`` followed by ``exterior_1_left``.  The right wrist and all
    unpaired modalities stay out of the reader request.
    """

    contract = contract or build_r1pro_contract()
    features: list[str] = []
    for view in contract.camera_views:
        features.append(view.dataset_rgb_key)
        if include_depth:
            observation = contract.robot.observations[view.robot_camera_key]
            if observation.depth_dataset_key is None:
                raise ValueError(f"{view.robot_camera_key} has no paired depth feature")
            features.append(observation.depth_dataset_key)
    return tuple(features)


def validate_selected_video_features(
    dataset_features: Mapping[str, Mapping[str, Any]],
    contract: B1KWeaverContract | None = None,
    *,
    include_depth: bool = False,
) -> tuple[str, ...]:
    """Fail before decoding when a selected RGB/depth feature is absent."""

    selected = selected_video_features(contract, include_depth=include_depth)
    for feature in selected:
        if feature not in dataset_features:
            raise KeyError(f"Missing selected B1K video feature: {feature}")
        feature_config = dataset_features[feature]
        if feature_config.get("dtype") not in {"video", "image"}:
            raise ValueError(f"Selected visual feature {feature!r} is not video/image metadata: {feature_config}")
        if "depth" in feature and not feature_config.get("info"):
            raise ValueError(f"Selected depth feature {feature!r} has no depth metadata marker")
    return selected


def _strict_world_model_frame_indices(
    timestamps_s: np.ndarray,
    *,
    target_hz: float,
    max_timing_error_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = _readonly_numeric(timestamps_s, name="timestamps_s", ndim=1).astype(np.float64, copy=False)
    if timestamps.size < 2:
        raise ValueError("An adapted episode must contain at least two source frames")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Episode timestamps must be strictly increasing")

    indices = world_model_frame_indices(timestamps, target_hz=target_hz)
    expected_count = math.floor((timestamps[-1] - timestamps[0]) * target_hz + 1e-9) + 1
    if len(indices) != expected_count:
        raise ValueError(
            "Timestamp gaps mapped multiple WEAVER timesteps to the same source frame: "
            f"expected {expected_count} frames at {target_hz:g} Hz, got {len(indices)}"
        )

    target_times = timestamps[0] + np.arange(expected_count, dtype=np.float64) / target_hz
    selected_times = timestamps[indices]
    timing_errors = np.abs(selected_times - target_times)
    if max_timing_error_s is None:
        median_source_period = float(np.median(np.diff(timestamps)))
        max_timing_error_s = max(2.0 * median_source_period, 1e-6)
    if not math.isfinite(max_timing_error_s) or max_timing_error_s < 0:
        raise ValueError(f"max_timing_error_s must be finite and non-negative, got {max_timing_error_s}")
    if timing_errors.max(initial=0.0) > max_timing_error_s:
        raise ValueError(
            "Source timestamps are too far from the requested WEAVER grid: "
            f"max error {timing_errors.max():.6f}s > {max_timing_error_s:.6f}s"
        )
    return indices, timing_errors


def angular_state_indices(contract: B1KWeaverContract | None = None) -> tuple[int, ...]:
    """Return compact-state coordinates that use shortest-angle deltas."""

    contract = contract or build_r1pro_contract()
    angular_components = {"trunk_qpos", "left_arm_qpos", "right_arm_qpos"}
    return tuple(
        index
        for component in contract.state_components
        if component.name in angular_components
        for index in component.compact_indices
    )


def observed_state_delta_actions(states: np.ndarray, contract: B1KWeaverContract | None = None) -> np.ndarray:
    """Compute observed 5 Hz state changes using the official plain-delta convention.

    The returned array has the same length as ``states`` to match WEAVER's
    trajectory contract.  The final row is zero and must be ignored according
    to :attr:`AdaptedEpisode.transition_valid`.
    """

    contract = contract or build_r1pro_contract()
    states = _readonly_numeric(states, name="states", ndim=2)
    if states.shape[-1] != contract.state_dim:
        raise ValueError(f"states must end in {contract.state_dim}, got {states.shape}")
    if states.shape[0] < 2:
        raise ValueError("At least two compact states are required to compute transitions")

    actions = np.zeros_like(states)
    actions[:-1] = states[1:] - states[:-1]
    return actions


@dataclasses.dataclass(frozen=True)
class AdaptedEpisode:
    episode_index: int
    prompt: str
    target_hz: float
    source_frame_indices: np.ndarray
    timestamps_s: np.ndarray
    timing_errors_s: np.ndarray
    states: np.ndarray
    state_delta_actions: np.ndarray
    policy_actions_at_frame: np.ndarray
    transition_valid: np.ndarray
    rewards: np.ndarray | None = None

    def __post_init__(self) -> None:
        if isinstance(self.episode_index, bool) or not isinstance(self.episode_index, int) or self.episode_index < 0:
            raise ValueError(f"episode_index must be a non-negative integer, got {self.episode_index!r}")
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not math.isfinite(self.target_hz) or self.target_hz <= 0:
            raise ValueError(f"target_hz must be positive and finite, got {self.target_hz}")

        fields = {
            "source_frame_indices": _readonly_numeric(self.source_frame_indices, name="source_frame_indices", ndim=1),
            "timestamps_s": _readonly_numeric(self.timestamps_s, name="timestamps_s", ndim=1),
            "timing_errors_s": _readonly_numeric(self.timing_errors_s, name="timing_errors_s", ndim=1),
            "states": _readonly_numeric(self.states, name="states", ndim=2),
            "state_delta_actions": _readonly_numeric(self.state_delta_actions, name="state_delta_actions", ndim=2),
            "policy_actions_at_frame": _readonly_numeric(
                self.policy_actions_at_frame, name="policy_actions_at_frame", ndim=2
            ),
        }
        if not np.issubdtype(fields["source_frame_indices"].dtype, np.integer):
            raise ValueError(f"source_frame_indices must contain integers, got {fields['source_frame_indices'].dtype}")
        transition_valid = np.asarray(self.transition_valid)
        if transition_valid.dtype != np.bool_ or transition_valid.ndim != 1:
            raise ValueError(
                f"transition_valid must be a boolean vector, got {transition_valid.shape}/{transition_valid.dtype}"
            )
        transition_valid = np.array(transition_valid, copy=True)
        transition_valid.setflags(write=False)
        fields["transition_valid"] = transition_valid

        length = fields["timestamps_s"].shape[0]
        if length < 2 or any(value.shape[0] != length for value in fields.values()):
            shapes = {name: value.shape for name, value in fields.items()}
            raise ValueError(f"Every adapted episode field must share a length >= 2, got {shapes}")
        if fields["states"].shape[-1] != 23:
            raise ValueError(f"states must be 23-D, got {fields['states'].shape}")
        if fields["state_delta_actions"].shape[-1] != 23:
            raise ValueError(f"state_delta_actions must be 23-D, got {fields['state_delta_actions'].shape}")
        if fields["policy_actions_at_frame"].shape[-1] != 23:
            raise ValueError(f"policy_actions_at_frame must be 23-D, got {fields['policy_actions_at_frame'].shape}")
        if fields["transition_valid"][-1] or not fields["transition_valid"][:-1].all():
            raise ValueError("transition_valid must be true except for the terminal frame")

        for name, value in fields.items():
            object.__setattr__(self, name, value)

        if self.rewards is not None:
            rewards = _readonly_numeric(self.rewards, name="rewards", ndim=1)
            if rewards.shape[0] != length:
                raise ValueError(f"rewards length {rewards.shape[0]} does not match episode length {length}")
            object.__setattr__(self, "rewards", rewards)


def adapt_episode(
    *,
    episode_index: int,
    prompt: str,
    timestamps_s: np.ndarray,
    raw_states: np.ndarray,
    policy_actions: np.ndarray,
    rewards: np.ndarray | None = None,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
    target_hz: float = 5.0,
    max_timing_error_s: float | None = None,
    min_sampled_frames: int = 2,
    contract: B1KWeaverContract | None = None,
) -> AdaptedEpisode:
    """Adapt one complete LeRobot episode without crossing boundaries."""

    contract = contract or build_r1pro_contract()
    raw_states = _readonly_numeric(raw_states, name="raw_states", ndim=2)
    policy_actions = _readonly_numeric(policy_actions, name="policy_actions", ndim=2)
    timestamps = _readonly_numeric(timestamps_s, name="timestamps_s", ndim=1)
    source_length = timestamps.shape[0]
    if raw_states.shape[0] != source_length or policy_actions.shape[0] != source_length:
        raise ValueError(
            "timestamps, raw_states, and policy_actions must describe one complete episode: "
            f"{timestamps.shape}, {raw_states.shape}, {policy_actions.shape}"
        )
    if policy_actions.shape[-1] != contract.action_dim:
        raise ValueError(f"policy_actions must be {contract.action_dim}-D, got {policy_actions.shape}")
    required_raw_state_dim = (
        max(index for component in contract.state_components for index in component.raw_indices) + 1
    )
    if raw_states.shape[-1] < required_raw_state_dim:
        raise ValueError(
            f"raw_states must contain at least {required_raw_state_dim} R1Pro coordinates, got {raw_states.shape}"
        )
    if isinstance(min_sampled_frames, bool) or not isinstance(min_sampled_frames, int) or min_sampled_frames < 2:
        raise ValueError(f"min_sampled_frames must be an integer >= 2, got {min_sampled_frames!r}")

    frame_indices, timing_errors = _strict_world_model_frame_indices(
        timestamps,
        target_hz=target_hz,
        max_timing_error_s=max_timing_error_s,
    )
    # The LeRobot episode metadata offsets are the authoritative episode
    # boundary. BEHAVIOR's recorded ``next.terminated`` marker can remain
    # true after a success while the exporter continues recording actions,
    # observations, and later progress rewards. It is retained as a source
    # label and audited by the reward contract, but must not crop this episode.
    if frame_indices.size < min_sampled_frames:
        raise ValueError(
            f"Episode {episode_index} has only {frame_indices.size} sampled frames through its valid boundary; "
            f"need {min_sampled_frames}"
        )

    compact_states = contract.extract_state(raw_states)[frame_indices]
    state_delta_actions = observed_state_delta_actions(compact_states, contract)
    transition_valid = np.ones(frame_indices.size, dtype=bool)
    transition_valid[-1] = False
    aggregated_rewards = None
    if rewards is not None:
        reward_result = _reward_contract.aggregate_progress_rewards(
            rewards,
            frame_indices,
            source_terminated=terminated,
            source_truncated=truncated,
        )
        if not np.array_equal(reward_result.transition_valid, transition_valid):
            raise AssertionError("Reward and observed-state transition masks diverged")
        aggregated_rewards = reward_result.rewards
    elif terminated is not None or truncated is not None:
        raise ValueError("terminated/truncated cannot be supplied without rewards")

    return AdaptedEpisode(
        episode_index=episode_index,
        prompt=prompt,
        target_hz=target_hz,
        source_frame_indices=frame_indices,
        timestamps_s=timestamps[frame_indices],
        timing_errors_s=timing_errors,
        states=compact_states,
        state_delta_actions=state_delta_actions,
        policy_actions_at_frame=policy_actions[frame_indices],
        transition_valid=transition_valid,
        rewards=aggregated_rewards,
    )


def deterministic_episode_split(
    episode_tasks: Mapping[int, Sequence[str]],
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
    held_out_tasks: Sequence[str] = (),
) -> dict[int, str]:
    """Create an order-independent episode split, stratified by one task.

    Episodes for ``held_out_tasks`` are assigned to ``test``.  Every remaining
    task with at least two episodes contributes at least one validation episode
    when ``val_fraction`` is non-zero, while retaining at least one train
    episode.
    """

    if not 0.0 <= val_fraction < 1.0 or not math.isfinite(val_fraction):
        raise ValueError(f"val_fraction must be finite and in [0,1), got {val_fraction}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")

    held_out = set(held_out_tasks)
    by_task: dict[str, list[int]] = {}
    result: dict[int, str] = {}
    normalized_tasks: dict[int, str] = {}
    for episode_index, tasks in episode_tasks.items():
        unique_tasks = tuple(dict.fromkeys(tasks))
        if len(unique_tasks) != 1:
            raise ValueError(
                f"Episode {episode_index} must have exactly one task for deterministic splitting, got {unique_tasks}"
            )
        task = unique_tasks[0]
        if not task:
            raise ValueError(f"Episode {episode_index} has an empty task")
        normalized_tasks[episode_index] = task

    unknown_held_out = held_out - set(normalized_tasks.values())
    if unknown_held_out:
        raise ValueError(f"Unknown held_out_tasks: {sorted(unknown_held_out)}")

    for episode_index, task in normalized_tasks.items():
        if task in held_out:
            result[episode_index] = "test"
        else:
            by_task.setdefault(task, []).append(episode_index)

    for task, episode_indices in by_task.items():
        ordered = sorted(
            episode_indices,
            key=lambda episode_index: hashlib.sha256(f"{seed}:{task}:{episode_index}".encode()).digest(),
        )
        if val_fraction == 0.0 or len(ordered) < 2:
            num_val = 0
        else:
            num_val = min(len(ordered) - 1, max(1, round(len(ordered) * val_fraction)))
        validation = set(ordered[:num_val])
        for episode_index in ordered:
            result[episode_index] = "val" if episode_index in validation else "train"

    return result


def deterministic_task_episode_sample(
    episode_tasks: Mapping[int, Sequence[str]],
    *,
    num_tasks: int,
    episodes_per_task: int,
    seed: int = 0,
) -> tuple[int, ...]:
    """Select a reproducible, task-diverse full-dataset smoke subset."""

    for name, value in (("num_tasks", num_tasks), ("episodes_per_task", episodes_per_task)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    by_task: dict[str, list[int]] = {}
    for episode_index, tasks in episode_tasks.items():
        unique_tasks = tuple(dict.fromkeys(tasks))
        if len(unique_tasks) != 1:
            raise ValueError(f"Episode {episode_index} must have exactly one task, got {unique_tasks}")
        by_task.setdefault(unique_tasks[0], []).append(episode_index)
    if num_tasks > len(by_task):
        raise ValueError(f"Requested {num_tasks} tasks but metadata contains only {len(by_task)}")

    selected_tasks = sorted(
        by_task,
        key=lambda task: hashlib.sha256(f"{seed}:task:{task}".encode()).digest(),
    )[:num_tasks]
    selected: list[int] = []
    for task in selected_tasks:
        ordered = sorted(
            by_task[task],
            key=lambda episode_index: hashlib.sha256(f"{seed}:episode:{task}:{episode_index}".encode()).digest(),
        )
        if len(ordered) < episodes_per_task:
            raise ValueError(f"Task {task!r} has only {len(ordered)} episodes; requested {episodes_per_task}")
        selected.extend(ordered[:episodes_per_task])
    return tuple(sorted(selected))


def episode_manifest(
    episode: AdaptedEpisode,
    *,
    split: str,
    dataset_repo_id: str,
    dataset_metadata_sha256: str,
    include_depth: bool = False,
    contract: B1KWeaverContract | None = None,
) -> dict[str, Any]:
    """Build the semantic portion of a latent-cache manifest."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test, got {split!r}")
    if len(dataset_metadata_sha256) != 64:
        raise ValueError("dataset_metadata_sha256 must be a SHA-256 digest")
    bytes.fromhex(dataset_metadata_sha256)
    contract = contract or build_r1pro_contract()
    view_mapping = {
        view.weaver_key: {
            "rgb": view.dataset_rgb_key,
            "depth": (contract.robot.observations[view.robot_camera_key].depth_dataset_key if include_depth else None),
        }
        for view in contract.camera_views
    }
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "repo_id": dataset_repo_id,
            "metadata_sha256": dataset_metadata_sha256,
        },
        "episode_index": episode.episode_index,
        "split": split,
        "prompt": episode.prompt,
        "target_hz": episode.target_hz,
        "sampled_count": int(episode.timestamps_s.shape[0]),
        "source_frame_indices_sha256": array_sha256(episode.source_frame_indices),
        "timestamps_sha256": array_sha256(episode.timestamps_s),
        "max_timing_error_s": float(episode.timing_errors_s.max(initial=0.0)),
        "views": view_mapping,
        "state": {"representation": "r1pro_compact_state_v1", "dim": 23},
        "training_action": dict(TRAINING_ACTION_CONTRACT),
        "source_policy_action": dict(SOURCE_POLICY_ACTION_CONTRACT),
        "online_steering": dict(ONLINE_STEERING_CONTRACT),
        "reward": (
            dict(_reward_contract.REWARD_CONTRACT)
            if episode.rewards is not None
            else {"representation": "none", "reason": "source_reward_not_exported"}
        ),
        "reward_available": episode.rewards is not None,
    }
    semantic["semantic_sha256"] = semantic_sha256(semantic)
    return semantic
