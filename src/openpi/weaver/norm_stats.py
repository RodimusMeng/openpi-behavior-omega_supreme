"""Streaming, train-only normalization statistics for B1K WEAVER artifacts."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import math
from typing import Any

import numpy as np

from openpi.weaver import b1k_dataset_adapter as _episode_adapter
from openpi.weaver.behavior_precomputed import NORM_SCHEMA_VERSION
from openpi.weaver.behavior_precomputed import NumericEpisode


@dataclasses.dataclass
class RunningMoments:
    """Numerically stable population moments for vector observations."""

    dimension: int
    count: int = 0
    mean: np.ndarray = dataclasses.field(init=False)
    m2: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {self.dimension!r}")
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.m2 = np.zeros(self.dimension, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"values must have shape (N,{self.dimension}), got {values.shape}")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("values must be finite numeric observations")
        batch_count = int(values.shape[0])
        if batch_count == 0:
            return
        values64 = values.astype(np.float64, copy=False)
        batch_mean = values64.mean(axis=0)
        centered = values64 - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean[:] = batch_mean
            self.m2[:] = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * (batch_count / total)
        self.m2 += batch_m2 + delta * delta * (self.count * batch_count / total)
        self.count = total

    def population_std(self) -> np.ndarray:
        if self.count == 0:
            raise ValueError("Cannot finalize empty running moments")
        variance = np.maximum(self.m2 / self.count, 0.0)
        return np.sqrt(variance)


def build_norm_stats(
    episodes: Iterable[NumericEpisode],
    *,
    dataset_repo_id: str,
    dataset_metadata_sha256: str,
    target_hz: float,
    train_manifest_sha256: str,
    min_std: float = 1e-6,
    clamp_small_std: bool = False,
) -> dict[str, Any]:
    """Compute upstream-compatible stats without loading all episodes at once."""

    if not dataset_repo_id:
        raise ValueError("dataset_repo_id must be non-empty")
    for name, digest in (
        ("dataset_metadata_sha256", dataset_metadata_sha256),
        ("train_manifest_sha256", train_manifest_sha256),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} must be a SHA-256 digest")
        bytes.fromhex(digest)
    if not math.isfinite(target_hz) or target_hz <= 0:
        raise ValueError("target_hz must be positive and finite")
    if not math.isfinite(min_std) or min_std <= 0:
        raise ValueError("min_std must be positive and finite")

    state_moments = RunningMoments(23)
    action_moments = RunningMoments(23)
    episode_count = 0
    for episode in episodes:
        state_moments.update(episode.states)
        action_moments.update(episode.actions[episode.transition_valid])
        episode_count += 1
    if episode_count == 0:
        raise ValueError("No train episodes were provided for normalization")

    state_std = state_moments.population_std()
    action_std = action_moments.population_std()
    state_small = state_std < min_std
    action_small = action_std < min_std
    if (state_small.any() or action_small.any()) and not clamp_small_std:
        raise ValueError(
            "Near-zero normalization std; inspect the data or explicitly enable clamping. "
            f"state dims={np.flatnonzero(state_small).tolist()}, "
            f"action dims={np.flatnonzero(action_small).tolist()}"
        )
    state_std = np.maximum(state_std, min_std)
    action_std = np.maximum(action_std, min_std)

    return {
        "schema_version": NORM_SCHEMA_VERSION,
        "dataset": {"repo_id": dataset_repo_id, "metadata_sha256": dataset_metadata_sha256},
        "source_split": "train",
        "target_hz": target_hz,
        "train_manifest_sha256": train_manifest_sha256,
        "episode_count": episode_count,
        "norm_stats": {
            "state": {
                "representation": "r1pro_compact_state_v1",
                "count": state_moments.count,
                "mean": state_moments.mean.tolist(),
                "std": state_std.tolist(),
                "zero_std_mask": state_small.tolist(),
            },
            "actions": {
                "representation": _episode_adapter.STATE_DELTA_ACTION_REPRESENTATION,
                "count": action_moments.count,
                "mean": action_moments.mean.tolist(),
                "std": action_std.tolist(),
                "zero_std_mask": action_small.tolist(),
            },
        },
    }
