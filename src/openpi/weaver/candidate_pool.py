"""Candidate-pool primitives for WEAVER-guided B1K action selection.

The current OpenPI policy exposes one sampled action chunk per ``infer`` call.
This module deliberately reuses that interface and freezes the resulting pool
before any selector sees it. Batched sampling can replace the loop later without
changing the comparison contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import math
from typing import Any, Protocol

import numpy as np


class InferencePolicy(Protocol):
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        """Sample one post-transform action chunk."""


def _readonly_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
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


def observation_sha256(observation: Mapping[str, Any]) -> str:
    """Hash an observation without retaining simulator or image contents."""

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, Mapping):
            digest.update(b"mapping{")
            for key in sorted(value):
                digest.update(str(key).encode())
                digest.update(b"=")
                update(value[key])
            digest.update(b"}")
            return
        if isinstance(value, np.ndarray):
            contiguous = np.ascontiguousarray(value)
            digest.update(b"ndarray:")
            digest.update(contiguous.dtype.str.encode())
            digest.update(repr(contiguous.shape).encode())
            digest.update(contiguous.tobytes())
            return
        if isinstance(value, str | bytes | int | float | bool | type(None)):
            digest.update(type(value).__name__.encode())
            digest.update(repr(value).encode())
            return
        update(np.asarray(value))

    update(observation)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class CandidatePool:
    """One immutable pool shared by every candidate-selection baseline."""

    actions: np.ndarray
    observation_digest: str
    noises: np.ndarray | None = None

    def __post_init__(self) -> None:
        actions = _readonly_array(self.actions, name="actions", ndim=3)
        if 0 in actions.shape:
            raise ValueError(f"actions must have non-empty (N,T,D) axes, got {actions.shape}")
        object.__setattr__(self, "actions", actions)

        if len(self.observation_digest) != 64:
            raise ValueError("observation_digest must be a SHA-256 hex digest")
        try:
            bytes.fromhex(self.observation_digest)
        except ValueError as exc:
            raise ValueError("observation_digest must be a SHA-256 hex digest") from exc

        if self.noises is not None:
            noises = _readonly_array(self.noises, name="noises")
            if noises.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"noises and actions must contain the same number of candidates, got {noises.shape[0]} and "
                    f"{actions.shape[0]}"
                )
            object.__setattr__(self, "noises", noises)

    @property
    def num_candidates(self) -> int:
        return self.actions.shape[0]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(f"candidate-{index:04d}" for index in range(self.num_candidates))

    @property
    def actions_sha256(self) -> str:
        """Digest the immutable candidate tensor for downstream linkage."""

        digest = hashlib.sha256()
        digest.update(self.actions.dtype.str.encode())
        digest.update(repr(self.actions.shape).encode())
        digest.update(np.ascontiguousarray(self.actions).tobytes())
        return digest.hexdigest()


def sample_policy_candidates(
    policy: InferencePolicy,
    observation: dict,
    *,
    num_candidates: int,
    action_dim: int = 23,
    noises: Sequence[np.ndarray] | np.ndarray | None = None,
) -> CandidatePool:
    """Generate an immutable ``(N,T,D)`` pool through the existing policy API.

    Returned actions are the post-output-transform chunks produced by
    ``Policy.infer``. For R1Pro this means ``D=23``; the internal padded 32-D
    model representation never crosses this boundary.
    """
    if isinstance(num_candidates, bool) or not isinstance(num_candidates, int) or num_candidates <= 0:
        raise ValueError(f"num_candidates must be a positive integer, got {num_candidates!r}")
    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
        raise ValueError(f"action_dim must be a positive integer, got {action_dim!r}")

    noise_items: list[np.ndarray] | None
    if noises is None:
        noise_items = None
    else:
        noise_array = np.asarray(noises)
        if noise_array.shape[0] != num_candidates:
            raise ValueError(
                f"Expected one noise tensor per candidate ({num_candidates}), got shape {noise_array.shape}"
            )
        noise_items = [np.array(noise_array[index], copy=True) for index in range(num_candidates)]

    chunks: list[np.ndarray] = []
    for index in range(num_candidates):
        if noise_items is None:
            output = policy.infer(observation)
        else:
            output = policy.infer(observation, noise=noise_items[index])
        if "actions" not in output:
            raise KeyError(f"Policy output for candidate {index} has no 'actions' key")
        chunk = np.asarray(output["actions"])
        if chunk.ndim != 2:
            raise ValueError(f"Candidate {index} actions must have shape (T,D), got {chunk.shape}")
        if chunk.shape[-1] != action_dim:
            raise ValueError(
                f"Candidate {index} must contain post-transform {action_dim}-D actions, got shape {chunk.shape}"
            )
        chunks.append(np.array(chunk, copy=True))

    return CandidatePool(
        actions=np.stack(chunks),
        observation_digest=observation_sha256(observation),
        noises=None if noise_items is None else np.stack(noise_items),
    )


def weaver_selection_scores(
    rewards: np.ndarray,
    *,
    gamma: float = 0.995,
    initial_values: np.ndarray | None = None,
    bootstrap_values: np.ndarray | None = None,
) -> np.ndarray:
    """Match the released WEAVER reward or discounted-advantage selector.

    Reward-only mode is selected when both value arrays are omitted and returns
    an undiscounted reward sum. Supplying both value arrays returns
    ``sum(gamma**t * reward[t]) + gamma**H * V_H - V_0``.
    """
    rewards = _readonly_array(rewards, name="rewards", ndim=2)
    if rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError(f"rewards must have non-empty (N,H) axes, got {rewards.shape}")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in [0, 1], got {gamma}")
    if (initial_values is None) != (bootstrap_values is None):
        raise ValueError("initial_values and bootstrap_values must be supplied together")

    if initial_values is None:
        scores = rewards.sum(axis=1)
    else:
        initial = _readonly_array(initial_values, name="initial_values", ndim=1)
        bootstrap = _readonly_array(bootstrap_values, name="bootstrap_values", ndim=1)
        expected_shape = (rewards.shape[0],)
        if initial.shape != expected_shape or bootstrap.shape != expected_shape:
            raise ValueError(
                "Value predictions must have one value per candidate: "
                f"expected {expected_shape}, got {initial.shape} and {bootstrap.shape}"
            )
        powers = gamma ** np.arange(rewards.shape[1], dtype=np.float64)
        scores = (rewards.astype(np.float64) * powers).sum(axis=1)
        scores += gamma ** rewards.shape[1] * bootstrap
        scores -= initial

    return _readonly_array(scores, name="scores", ndim=1)


def _validate_scores(pool: CandidatePool, scores: np.ndarray, *, name: str) -> np.ndarray:
    scores = _readonly_array(scores, name=name, ndim=1)
    if scores.shape != (pool.num_candidates,):
        raise ValueError(f"{name} must have shape ({pool.num_candidates},), got {scores.shape}")
    return scores


@dataclasses.dataclass(frozen=True)
class SamePoolSelections:
    """Indices chosen by the four required baselines from one frozen pool."""

    top1: int
    random: int
    weaver: int
    oracle: int


def select_same_pool_baselines(
    pool: CandidatePool,
    *,
    weaver_scores: np.ndarray,
    oracle_scores: np.ndarray,
    rng: np.random.Generator,
) -> SamePoolSelections:
    """Select Top-1/random/WEAVER/oracle indices without regenerating actions."""
    predicted = _validate_scores(pool, weaver_scores, name="weaver_scores")
    actual = _validate_scores(pool, oracle_scores, name="oracle_scores")
    return SamePoolSelections(
        top1=0,
        random=int(rng.integers(pool.num_candidates)),
        weaver=int(np.argmax(predicted)),
        oracle=int(np.argmax(actual)),
    )
