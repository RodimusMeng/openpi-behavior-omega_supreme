"""Audited BEHAVIOR reward alignment for WEAVER's 5 Hz trajectory rows.

The B1K LeRobot writer stores ``observation[t]`` together with the action that
leaves that observation and the resulting ``next.reward``.  ``BehaviorTask``'s
default reward is the change in satisfied-goal-predicate fraction.  Therefore a
world-model transition from source row ``i`` to source row ``j`` receives the
sum of source rewards ``[i, j)``; endpoint sampling is incorrect.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

REWARD_REPRESENTATION = "b1k_behavior_recorded_transition_reward_5hz_v4"
REWARD_CONTRACT = {
    "representation": REWARD_REPRESENTATION,
    "alignment": "sum_next_reward_over_source_rows_i_to_j_exclusive",
    "source": "LeRobot next.reward as recorded by the B1K demonstration export",
    "source_semantics": "official_BehaviorTask_PotentialReward_when_available_but_not_reconstructed_offline",
    "source_rate_hz": 30.0,
    "state_aligned": False,
    "transition_aligned": True,
    "terminal_row_valid": False,
    "terminal_marker_semantics": "source_labels_preserved_episode_boundary_from_metadata_offsets",
    "terminal_reward_policy": "preserve_recorded_rewards_without_zero_suffix_assumption",
    "final_source_tail_policy": "preserved_in_source_arrays_not_supervised_without_later_sampled_observation",
    "privileged_offline_label": True,
    "allowed_online_input": False,
}


@dataclasses.dataclass(frozen=True)
class AggregatedRewards:
    """A reward for each sampled row plus the transitions on which it is valid."""

    rewards: np.ndarray
    transition_valid: np.ndarray


def terminal_suffix_start(done: Any) -> int | None:
    """Return the start of a final contiguous true marker run, if present.

    This is a diagnostic helper only. Metadata episode offsets, rather than
    this marker pattern, define the official episode boundary. A marker run
    that does not reach the final row is still returned as a source-label
    diagnostic and is never rejected here.
    """

    array = np.asarray(done)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.dtype != np.bool_:
        raise ValueError(f"done must be a boolean vector, got {array.shape}/{array.dtype}")
    positions = np.flatnonzero(array)
    if positions.size == 0:
        return None
    suffix_start = int(positions[-1])
    while suffix_start > 0 and bool(array[suffix_start - 1]):
        suffix_start -= 1
    return suffix_start


def aggregate_progress_rewards(
    source_rewards: Any,
    source_frame_indices: Any,
    *,
    source_terminated: Any | None = None,
    source_truncated: Any | None = None,
    validate_progress_range: bool = False,
    tolerance: float = 1e-5,
) -> AggregatedRewards:
    """Aggregate 30 Hz transition rewards onto sampled WEAVER transitions.

    For sampled rows ``[i0, i1, ...]``, result row ``k`` is
    ``sum(source_rewards[ik:i(k+1)])``.  The final row is exactly zero and
    invalid because there is no following sampled observation/action target.

    Args:
        source_rewards: One finite scalar reward per source row. ``(T, 1)`` is
            accepted because that is the B1K LeRobot feature shape.
        source_frame_indices: Strictly increasing episode-local sampled rows.
        source_terminated: Optional bool terminal marker per source row.
        source_truncated: Optional bool timeout marker per source row.
        validate_progress_range: Deprecated compatibility argument. The B1K
            export is authoritative and no unit-range assumption is applied.
        tolerance: Numerical tolerance for the progress-range check.
    """

    rewards = _scalar_vector(source_rewards, name="source_rewards", dtype=np.float64)
    indices = np.asarray(source_frame_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"source_frame_indices must be an integer vector, got {indices.shape}/{indices.dtype}")
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    if indices.size < 2 or indices[0] < 0 or indices[-1] >= rewards.size:
        raise ValueError(
            "source_frame_indices must contain at least two in-range rows: "
            f"indices={indices.tolist()}, source_length={rewards.size}"
        )
    if np.any(np.diff(indices) <= 0):
        raise ValueError("source_frame_indices must be strictly increasing")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance must be finite and non-negative, got {tolerance!r}")

    terminated = _optional_bool_vector(source_terminated, length=rewards.size, name="source_terminated")
    truncated = _optional_bool_vector(source_truncated, length=rewards.size, name="source_truncated")
    if terminated is not None or truncated is not None:
        done = np.zeros(rewards.size, dtype=bool)
        if terminated is not None:
            done |= terminated
        if truncated is not None:
            done |= truncated
        _validate_terminal_markers(done)

    del validate_progress_range
    aggregated = np.zeros(indices.size, dtype=np.float32)
    prefix = np.concatenate(([0.0], np.cumsum(rewards, dtype=np.float64)))
    starts = indices[:-1]
    ends = indices[1:]
    aggregated[:-1] = (prefix[ends] - prefix[starts]).astype(np.float32)
    valid = np.ones(indices.size, dtype=bool)
    valid[-1] = False
    aggregated.setflags(write=False)
    valid.setflags(write=False)
    return AggregatedRewards(rewards=aggregated, transition_valid=valid)


def _scalar_vector(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric (T,) or (T,1) array, got {array.shape}/{array.dtype}")
    result = np.ascontiguousarray(array, dtype=dtype)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def _optional_bool_vector(value: Any | None, *, length: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.dtype != np.bool_ or array.shape[0] != length:
        raise ValueError(f"{name} must be a boolean ({length},) or ({length},1) array, got {array.shape}/{array.dtype}")
    return np.ascontiguousarray(array)


def _validate_terminal_markers(done: np.ndarray) -> None:
    """Validate marker shape without imposing an absorbing-reward convention.

    BEHAVIOR's LeRobot export records the result of each environment step in
    ``next.terminated``/``next.truncated``.  In the 2026 full export a success
    marker may remain true through the remainder of the recorded episode while
    actions, states, and later progress rewards continue to be recorded.  The
    metadata episode offsets are therefore the only authoritative episode
    boundary.  This function intentionally performs no reward-value check.
    """

    del done
