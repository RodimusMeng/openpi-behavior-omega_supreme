"""B1K episode artifacts exposed with WEAVER's precomputed batch contract.

This module intentionally has no PyTorch dependency.  ``torch.utils.data`` can
consume it as a map-style dataset and its default collate function converts the
returned NumPy arrays to tensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

from openpi.weaver import b1k_dataset_adapter as _episode_adapter
from openpi.weaver import latent_cache as _latent_cache
from openpi.weaver import reward_contract as _reward_contract

NORM_SCHEMA_VERSION = "b1k_weaver_norm_v1"
DEFAULT_IMAGE_KEYS = ("wrist_left", "exterior_1_left")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to {root}, got {value!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{field} escapes dataset root: {value!r}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact {path} must contain an object")
    return value


def _readonly_float32(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a rank-{ndim} floating array, got {array.shape}/{array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    result = np.ascontiguousarray(array, dtype=np.float32)
    result.setflags(write=False)
    return result


def _readonly_integer(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an integer vector, got {array.shape}/{array.dtype}")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NumericEpisode:
    states: np.ndarray
    actions: np.ndarray
    policy_actions: np.ndarray
    source_timestamps_s: np.ndarray
    source_policy_actions: np.ndarray
    rewards: np.ndarray | None
    transition_valid: np.ndarray
    source_frame_indices: np.ndarray
    source_absolute_indices: np.ndarray
    timestamps_s: np.ndarray
    timing_errors_s: np.ndarray

    @property
    def length(self) -> int:
        return int(self.states.shape[0])


def load_numeric_episode(root: str | Path, manifest: Mapping[str, Any]) -> NumericEpisode:
    """Load and byte/shape/semantic-check one numeric episode artifact."""

    root = Path(root)
    _episode_adapter.verify_episode_manifest(manifest)
    if manifest.get("state") != {"representation": "r1pro_compact_state_v1", "dim": 23}:
        raise ValueError(f"Unsupported B1K state contract: {manifest.get('state')!r}")
    if manifest.get("training_action") != _episode_adapter.TRAINING_ACTION_CONTRACT:
        raise ValueError(f"Unsupported WEAVER action contract: {manifest.get('training_action')!r}")
    if manifest.get("source_policy_action") != _episode_adapter.SOURCE_POLICY_ACTION_CONTRACT:
        raise ValueError(f"Unsupported source policy-action contract: {manifest.get('source_policy_action')!r}")
    if manifest.get("online_steering") != _episode_adapter.ONLINE_STEERING_CONTRACT:
        raise ValueError(f"Unsupported online-steering contract: {manifest.get('online_steering')!r}")
    if manifest.get("reward_available") is True:
        if manifest.get("reward") != _reward_contract.REWARD_CONTRACT:
            raise ValueError(f"Unsupported B1K reward contract: {manifest.get('reward')!r}")
    elif manifest.get("reward") != {"representation": "none", "reason": "source_reward_not_exported"}:
        raise ValueError(f"Invalid no-reward contract: {manifest.get('reward')!r}")

    artifact = manifest.get("numeric_artifact")
    if not isinstance(artifact, Mapping) or artifact.get("complete") is not True:
        raise ValueError("Numeric episode artifact is incomplete")
    path = _safe_relative_path(root, artifact.get("path"), field="numeric_artifact.path")
    if not path.is_file():
        raise FileNotFoundError(f"Numeric episode artifact is missing: {path}")
    if artifact.get("size_bytes") != path.stat().st_size or artifact.get("sha256") != _file_sha256(path):
        raise ValueError(f"Numeric episode bytes do not match the manifest: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Numeric episode NPZ is corrupt or unsafe: {path}: {exc}") from exc

    required = {
        "index",
        "frame_index",
        "source_timestamps_s",
        "source_policy_actions",
        "source_frame_indices",
        "source_absolute_indices",
        "timestamps_s",
        "timing_errors_s",
        "states",
        "state_delta_actions",
        "policy_actions_at_frame",
        "transition_valid",
    }
    if manifest.get("reward_available") is True:
        required.update({"source_rewards", "source_terminated", "source_truncated", "rewards"})
    if set(arrays) != required or set(artifact.get("keys", ())) != required:
        raise ValueError(
            f"Numeric episode keys mismatch: expected {sorted(required)}, "
            f"manifest={sorted(artifact.get('keys', ()))}, npz={sorted(arrays)}"
        )

    source_indices = _readonly_integer(arrays["source_frame_indices"], name="source_frame_indices")
    absolute_indices = _readonly_integer(arrays["source_absolute_indices"], name="source_absolute_indices")
    full_indices = _readonly_integer(arrays["index"], name="index")
    full_frame_indices = _readonly_integer(arrays["frame_index"], name="frame_index")
    timestamps = np.asarray(arrays["timestamps_s"])
    timing_errors = np.asarray(arrays["timing_errors_s"])
    for name, array in (("timestamps_s", timestamps), ("timing_errors_s", timing_errors)):
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite floating vector, got {array.shape}/{array.dtype}")
    if np.any(np.diff(timestamps) <= 0) or np.any(timing_errors < 0):
        raise ValueError("Episode timestamps must increase and timing errors must be non-negative")
    timestamps.setflags(write=False)
    timing_errors.setflags(write=False)
    full_timestamps = np.asarray(arrays["source_timestamps_s"])
    if (
        full_timestamps.ndim != 1
        or not np.issubdtype(full_timestamps.dtype, np.floating)
        or not np.isfinite(full_timestamps).all()
        or np.any(np.diff(full_timestamps) <= 0)
    ):
        raise ValueError(
            "source_timestamps_s must be a finite, strictly increasing floating vector, "
            f"got {full_timestamps.shape}/{full_timestamps.dtype}"
        )
    full_timestamps.setflags(write=False)

    states = _readonly_float32(arrays["states"], name="states", ndim=2)
    actions = _readonly_float32(arrays["state_delta_actions"], name="state_delta_actions", ndim=2)
    policy_actions = _readonly_float32(arrays["policy_actions_at_frame"], name="policy_actions_at_frame", ndim=2)
    full_policy_actions = _readonly_float32(arrays["source_policy_actions"], name="source_policy_actions", ndim=2)
    transition_valid = np.asarray(arrays["transition_valid"])
    if transition_valid.ndim != 1 or transition_valid.dtype != np.bool_:
        raise ValueError(
            f"transition_valid must be a boolean vector, got {transition_valid.shape}/{transition_valid.dtype}"
        )
    transition_valid.setflags(write=False)

    length = int(manifest.get("sampled_count", -1))
    fields = (
        source_indices,
        absolute_indices,
        timestamps,
        timing_errors,
        states,
        actions,
        policy_actions,
        transition_valid,
    )
    if length < 2 or any(value.shape[0] != length for value in fields):
        raise ValueError(f"Numeric episode fields do not share sampled_count={length}")
    if states.shape[1] != 23 or actions.shape[1] != 23 or policy_actions.shape[1] != 23:
        raise ValueError("B1K state, WEAVER action, and source policy action must all be 23-D")
    if transition_valid[-1] or not transition_valid[:-1].all():
        raise ValueError("Only the final state-delta transition may be invalid")
    if np.any(np.diff(source_indices) <= 0) or np.any(np.diff(absolute_indices) <= 0):
        raise ValueError("Source frame indices must be strictly increasing within an episode")
    source = manifest.get("source", {})
    if not isinstance(source, Mapping):
        raise ValueError("Episode manifest source must be an object")
    source_start = int(source.get("dataset_from_index", -1))
    source_end = int(source.get("dataset_to_index", -1))
    full_length = source_end - source_start
    full_fields = (full_indices, full_frame_indices, full_timestamps, full_policy_actions)
    if full_length <= 0 or any(value.shape[0] != full_length for value in full_fields):
        raise ValueError(f"Full-rate source fields do not match source bounds [{source_start}, {source_end})")
    if source.get("full_rate_count") != full_length:
        raise ValueError("source.full_rate_count does not match the authoritative source index range")
    if source.get("policy_action_aggregation_5hz") != "undefined":
        raise ValueError("A causal 30 Hz to 5 Hz policy-action aggregation has not been defined")
    if not np.array_equal(full_indices, np.arange(source_start, source_end, dtype=full_indices.dtype)):
        raise ValueError("Authoritative index does not match dataset_from_index/dataset_to_index")
    if not np.array_equal(full_frame_indices, np.arange(full_length, dtype=full_frame_indices.dtype)):
        raise ValueError("Authoritative frame_index is not episode-local and contiguous")
    if source_indices[0] < 0 or source_indices[-1] >= full_length:
        raise ValueError("Sampled source_frame_indices fall outside the full-rate episode")
    if not np.array_equal(source_indices, full_frame_indices[source_indices]):
        raise ValueError("Sampled source_frame_indices do not match authoritative frame_index")
    if not np.array_equal(absolute_indices, full_indices[source_indices]):
        raise ValueError("source_absolute_indices do not match authoritative index at sampled frames")
    if not np.array_equal(timestamps, full_timestamps[source_indices]):
        raise ValueError("Sampled timestamps do not match full-rate source timestamps")
    if not np.array_equal(policy_actions, full_policy_actions[source_indices]):
        raise ValueError("Sampled policy actions do not match full-rate source policy actions")
    expected_actions = _episode_adapter.observed_state_delta_actions(states)
    if not np.allclose(actions, expected_actions, rtol=1e-6, atol=1e-6):
        raise ValueError("state_delta_actions do not match adjacent compact states")
    if _episode_adapter.array_sha256(source_indices) != manifest.get("source_frame_indices_sha256"):
        raise ValueError("source_frame_indices SHA-256 does not match the episode semantics")
    if _episode_adapter.array_sha256(timestamps) != manifest.get("timestamps_sha256"):
        raise ValueError("timestamps SHA-256 does not match the episode semantics")

    rewards = None
    if "rewards" in arrays:
        rewards = _readonly_float32(arrays["rewards"], name="rewards", ndim=1)
        if rewards.shape[0] != length:
            raise ValueError(f"Reward length {rewards.shape[0]} does not match sampled_count={length}")
        source_rewards = np.asarray(arrays["source_rewards"])
        source_terminated = np.asarray(arrays["source_terminated"])
        source_truncated = np.asarray(arrays["source_truncated"])
        for name, value in (
            ("source_rewards", source_rewards),
            ("source_terminated", source_terminated),
            ("source_truncated", source_truncated),
        ):
            if value.ndim == 1 or (value.ndim == 2 and value.shape[1] == 1):
                value_length = value.shape[0]
            else:
                raise ValueError(f"{name} must have shape (T,) or (T,1), got {value.shape}")
            if value_length != full_length:
                raise ValueError(f"{name} length {value_length} does not match full-rate length {full_length}")
        aggregated = _reward_contract.aggregate_progress_rewards(
            source_rewards,
            source_indices,
            source_terminated=source_terminated,
            source_truncated=source_truncated,
        )
        if not np.array_equal(transition_valid, aggregated.transition_valid):
            raise ValueError("Reward and numeric transition_valid masks differ")
        if not np.allclose(rewards, aggregated.rewards, rtol=1e-6, atol=1e-6):
            raise ValueError("5 Hz rewards do not match the audited full-rate aggregation")
    return NumericEpisode(
        states=states,
        actions=actions,
        policy_actions=policy_actions,
        source_timestamps_s=full_timestamps,
        source_policy_actions=full_policy_actions,
        rewards=rewards,
        transition_valid=transition_valid,
        source_frame_indices=source_indices,
        source_absolute_indices=absolute_indices,
        timestamps_s=timestamps,
        timing_errors_s=timing_errors,
    )


def load_episode_latents(
    latent_root: str | Path,
    numeric_manifest: Mapping[str, Any],
    *,
    image_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Load one named-view latent artifact and prove linkage to numeric data."""

    latent_root = Path(latent_root).resolve()
    split = str(numeric_manifest["split"])
    episode_index = int(numeric_manifest["episode_index"])
    manifest_path = latent_root / split / f"{episode_index:06d}.json"
    latent_manifest = _read_json(manifest_path)
    if latent_manifest.get("numeric_episode_semantic_sha256") != numeric_manifest.get("semantic_sha256"):
        raise ValueError(f"Latent artifact is linked to different numeric semantics: {manifest_path}")
    if latent_manifest.get("episode_index") != episode_index or latent_manifest.get("split") != split:
        raise ValueError(f"Latent artifact episode/split mismatch: {manifest_path}")
    if tuple(latent_manifest.get("view_order", ())) != image_keys:
        raise ValueError(
            f"Latent view order {latent_manifest.get('view_order')!r} does not match requested {image_keys!r}"
        )
    artifact = latent_manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError(f"Latent manifest has no artifact object: {manifest_path}")
    npz_path = _safe_relative_path(manifest_path.parent, artifact.get("path"), field="latent artifact.path")
    arrays = _latent_cache.load_latent_artifact(
        npz_path,
        manifest_path,
        expected_semantic_sha256=str(latent_manifest.get("semantic_sha256")),
    )
    return {key: _readonly_float32(value, name=key, ndim=3) for key, value in arrays.items()}


def _load_norm_stats(
    path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_dataset_metadata_sha256: str | None = None,
    expected_target_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = _read_json(path)
    if value.get("schema_version") != NORM_SCHEMA_VERSION:
        raise ValueError(f"Unsupported WEAVER norm schema in {path}: {value.get('schema_version')!r}")
    stats = value.get("norm_stats")
    if not isinstance(stats, Mapping):
        raise ValueError(f"Missing norm_stats object in {path}")
    state = stats.get("state")
    actions = stats.get("actions")
    if value.get("source_split") != "train":
        raise ValueError(f"Normalization statistics must declare source_split='train' in {path}")
    if expected_manifest_sha256 is not None and value.get("train_manifest_sha256") != expected_manifest_sha256:
        raise ValueError(f"Normalization statistics are from a different manifest index: {path}")
    dataset = value.get("dataset")
    if expected_dataset_metadata_sha256 is not None and (
        not isinstance(dataset, Mapping) or dataset.get("metadata_sha256") != expected_dataset_metadata_sha256
    ):
        raise ValueError(f"Normalization statistics are from a different dataset metadata hash: {path}")
    if expected_target_hz is not None and float(value.get("target_hz", -1)) != float(expected_target_hz):
        raise ValueError(f"Normalization statistics target_hz does not match dataset: {path}")
    if not isinstance(state, Mapping) or state.get("representation") != "r1pro_compact_state_v1":
        raise ValueError(f"Wrong state normalization representation in {path}")
    if not isinstance(actions, Mapping) or actions.get("representation") != (
        _episode_adapter.STATE_DELTA_ACTION_REPRESENTATION
    ):
        raise ValueError(f"Wrong action normalization representation in {path}")

    def pair(entry: Mapping[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
        mean = _readonly_float32(entry.get("mean"), name=f"{name}.mean", ndim=1)
        std = _readonly_float32(entry.get("std"), name=f"{name}.std", ndim=1)
        if mean.shape != (23,) or std.shape != (23,) or np.any(std <= 0):
            raise ValueError(f"{name} norm mean/std must be 23-D with positive std")
        return mean, std

    state_mean, state_std = pair(state, "state")
    action_mean, action_std = pair(actions, "actions")
    return state_mean, state_std, action_mean, action_std


def norm_stats_provenance(path: str | Path) -> dict[str, Any]:
    """Return the immutable identity fields for one normalization artifact."""

    resolved = Path(path).expanduser().resolve()
    value = _read_json(resolved)
    required = ("schema_version", "source_split", "target_hz", "train_manifest_sha256", "episode_count")
    missing = tuple(name for name in required if name not in value)
    dataset = value.get("dataset")
    if missing or not isinstance(dataset, Mapping):
        raise ValueError(f"Normalization provenance is incomplete in {resolved}: missing={missing}")
    return {
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
        "schema_version": value["schema_version"],
        "source_split": value["source_split"],
        "target_hz": value["target_hz"],
        "train_manifest_sha256": value["train_manifest_sha256"],
        "episode_count": value["episode_count"],
        "dataset": dict(dataset),
    }


class BehaviorPrecomputed:
    """Map-style dataset matching ``PrecomputedDroid``'s batch structure.

    Each dataset item is one random window from one episode, matching upstream
    sampling.  ``window_at`` is the deterministic inspection/evaluation API.
    Rewards are either required for every episode or disabled for dynamics-only
    training; they are never synthesized.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        horizon: int = 8,
        image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS,
        n_history: int = 2,
        n_memory_frames: int = 0,
        t_memory: int = 1,
        normalize: bool = True,
        norm_stats_path: str | Path | None = None,
        reward_mode: str = "required",
        collapse_prob: float = 0.0,
        max_trajectories: int | None = None,
        latent_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if latent_root is None:
            self.latent_root = self.root / "latents"
        else:
            candidate_latent_root = Path(latent_root)
            self.latent_root = (
                candidate_latent_root.resolve()
                if candidate_latent_root.is_absolute()
                else (self.root / candidate_latent_root).resolve()
            )
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (horizon, n_history)):
            raise ValueError("horizon and n_history must be positive integers")
        if isinstance(n_memory_frames, bool) or not isinstance(n_memory_frames, int) or n_memory_frames < 0:
            raise ValueError("n_memory_frames must be a non-negative integer")
        if isinstance(t_memory, bool) or not isinstance(t_memory, int) or t_memory <= 0:
            raise ValueError("t_memory must be a positive integer")
        if not image_keys or len(set(image_keys)) != len(image_keys):
            raise ValueError(f"image_keys must be non-empty and unique, got {image_keys!r}")
        if reward_mode not in {"required", "disabled"}:
            raise ValueError("reward_mode must be 'required' or 'disabled'")
        if not math.isfinite(collapse_prob) or not 0.0 <= collapse_prob <= 1.0:
            raise ValueError("collapse_prob must be finite and in [0,1]")

        index_path = self.root / "manifest.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing B1K WEAVER manifest index: {index_path}")
        manifests: list[dict[str, Any]] = []
        seen: set[int] = set()
        with index_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    manifest = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {index_path}:{line_number}: {exc}") from exc
                _episode_adapter.verify_episode_manifest(manifest)
                if manifest["split"] != split:
                    continue
                episode_index = int(manifest["episode_index"])
                if episode_index in seen:
                    raise ValueError(f"Duplicate episode_index {episode_index} in {index_path}")
                seen.add(episode_index)
                minimum_frames = n_history + horizon + (1 if reward_mode == "required" else 0)
                if int(manifest["sampled_count"]) < minimum_frames:
                    continue
                if reward_mode == "required" and manifest.get("reward_available") is not True:
                    raise ValueError(
                        f"Episode {episode_index} has no real reward labels; use reward_mode='disabled' "
                        "for dynamics-only training"
                    )
                manifests.append(manifest)
        manifests.sort(key=lambda manifest: int(manifest["episode_index"]))
        if max_trajectories is not None:
            if isinstance(max_trajectories, bool) or not isinstance(max_trajectories, int) or max_trajectories <= 0:
                raise ValueError("max_trajectories must be a positive integer")
            manifests = manifests[:max_trajectories]
        if not manifests:
            raise ValueError(
                f"No {split} episodes contain the required {n_history}+{horizon} sampled frames in {index_path}"
            )

        self.split = split
        self.horizon = horizon
        self.image_keys = tuple(image_keys)
        self.n_history = n_history
        self.n_memory_frames = n_memory_frames
        self.t_memory = t_memory
        self.reward_mode = reward_mode
        self.collapse_prob = collapse_prob
        self.manifests = tuple(manifests)
        self.normalize = normalize
        self.norm_stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.norm_stats_provenance: dict[str, Any] | None = None
        if normalize:
            norm_path = Path(norm_stats_path) if norm_stats_path is not None else self.root / "norm_stats.json"
            if not norm_path.is_absolute():
                norm_path = self.root / norm_path
            manifest_sha = _file_sha256(self.root / "manifest.jsonl")
            metadata_hashes = {str(manifest["dataset"]["metadata_sha256"]) for manifest in self.manifests}
            target_rates = {float(manifest["target_hz"]) for manifest in self.manifests}
            if len(metadata_hashes) != 1 or len(target_rates) != 1:
                raise ValueError("Dataset manifests mix metadata hashes or target rates")
            self.norm_stats = _load_norm_stats(
                norm_path.resolve(),
                expected_manifest_sha256=manifest_sha,
                expected_dataset_metadata_sha256=metadata_hashes.pop(),
                expected_target_hz=target_rates.pop(),
            )
            self.norm_stats_provenance = norm_stats_provenance(norm_path)

    def __len__(self) -> int:
        return len(self.manifests)

    def _load_episode(self, index: int) -> tuple[Mapping[str, Any], NumericEpisode, dict[str, np.ndarray]]:
        manifest = self.manifests[index]
        numeric = load_numeric_episode(self.root, manifest)
        latents = load_episode_latents(self.latent_root, manifest, image_keys=self.image_keys)
        for key, value in latents.items():
            if value.shape[0] != numeric.length:
                raise ValueError(f"{key} length {value.shape[0]} does not match numeric length {numeric.length}")
        return manifest, numeric, latents

    def window_at(self, index: int, t: int, *, collapse: bool = False) -> dict[str, Any]:
        manifest, numeric, latents = self._load_episode(index)
        min_t = self.n_history - 1
        terminal_margin = 1 if self.reward_mode == "required" else 0
        max_t = numeric.length - self.horizon - 1 - terminal_margin
        if isinstance(t, bool) or not isinstance(t, int) or not min_t <= t <= max_t:
            raise ValueError(f"t must be in [{min_t}, {max_t}] for episode {manifest['episode_index']}, got {t!r}")
        start = t - self.n_history + 1
        end = t + self.horizon + 1

        states = np.array(numeric.states[start:end], dtype=np.float32, copy=True)
        actions = np.array(numeric.actions[start:end], dtype=np.float32, copy=True)
        obs = {
            f"{image_key}_features": np.array(latents[f"{image_key}_features"][start:end], dtype=np.float32, copy=True)
            for image_key in self.image_keys
        }
        if self.norm_stats is not None:
            state_mean, state_std, action_mean, action_std = self.norm_stats
            states = (states - state_mean) / state_std
            actions = (actions - action_mean) / action_std

        if collapse and self.n_history > 1:
            states[: self.n_history] = states[self.n_history - 1]
            actions[: self.n_history - 1] = 0.0
            for value in obs.values():
                value[: self.n_history] = value[self.n_history - 1]
        obs["states"] = np.ascontiguousarray(states)

        result: dict[str, Any] = {
            "obs": obs,
            "actions": np.ascontiguousarray(actions),
            "transition_valid": np.array(numeric.transition_valid[start:end], copy=True),
            "task": {"text": str(manifest["prompt"])},
        }
        if self.reward_mode == "required":
            if numeric.rewards is None:
                raise ValueError(f"Episode {manifest['episode_index']} lost required reward labels")
            if not numeric.transition_valid[start:end].all():
                raise ValueError(
                    f"Full WEAVER window for episode {manifest['episode_index']} contains an invalid transition"
                )
            result["rewards"] = np.array(numeric.rewards[start:end], dtype=np.float32, copy=True)

        if self.n_memory_frames > 0:
            effective_t_memory = 0 if collapse else self.t_memory
            memory_indices = [
                max(0, t - (self.n_memory_frames - offset) * effective_t_memory)
                for offset in range(self.n_memory_frames)
            ]
            memory_states = np.array(numeric.states[memory_indices], dtype=np.float32, copy=True)
            if self.norm_stats is not None:
                state_mean, state_std, _, _ = self.norm_stats
                memory_states = (memory_states - state_mean) / state_std
            memory = {
                f"{image_key}_features": np.ascontiguousarray(
                    latents[f"{image_key}_features"][memory_indices], dtype=np.float32
                )
                for image_key in self.image_keys
            }
            memory["states"] = np.ascontiguousarray(memory_states)
            result["memory"] = memory
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        sampled_count = int(self.manifests[index]["sampled_count"])
        min_t = self.n_history - 1
        terminal_margin = 1 if self.reward_mode == "required" else 0
        max_t = sampled_count - self.horizon - 1 - terminal_margin
        t = random.randint(min_t, max_t) if max_t > min_t else min_t
        return self.window_at(index, t, collapse=random.random() < self.collapse_prob)
