"""Online same-candidate-pool WEAVER steering policy boundary.

The class is deliberately agnostic to the official WebSocket transport.  It
implements the policy-facing ``reset``/``act`` methods used by the existing
OpenPI server and returns one legal post-transform controller action per call.
Every replan samples one immutable pool, adapts and scores that exact pool,
then executes the selected original policy chunk; no candidate is synthesized
after scoring and no simulator oracle is accepted on the online path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any

import numpy as np

from openpi.weaver import candidate_pool
from openpi.weaver import causal_action_adapter
from openpi.weaver import steering_runtime


@dataclasses.dataclass(frozen=True)
class SteeringTelemetry:
    replans: int
    selected_candidate_ids: tuple[str, ...]
    selected_indices: tuple[int, ...]
    pool_action_digests: tuple[str, ...]


class WeaverSteeringPolicy:
    """Wrap an existing OpenPI policy with WEAVER candidate selection."""

    def __init__(
        self,
        *,
        policy: Any,
        adapter_model: Any,
        adapter_normalization: causal_action_adapter.AdapterNormalization,
        weaver_action_mean: np.ndarray,
        weaver_action_std: np.ndarray,
        weaver_model: Any,
        context_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        history_actions_fn: Callable[[Mapping[str, Any]], Any],
        task_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        current_state_fn: Callable[[Mapping[str, Any]], np.ndarray],
        action_time_edges_fn: Callable[[Mapping[str, Any], int], np.ndarray] | None = None,
        memory_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        imagination_horizon: int,
        num_candidates: int = 4,
        replan_interval: int = 1,
        device: str = "cuda",
        gamma: float = 0.995,
        criterion: steering_runtime.SelectionCriterion = "advantage",
        score_fn: Callable[..., steering_runtime.ScoredCandidatePool] | None = None,
    ) -> None:
        if num_candidates <= 0 or replan_interval <= 0 or imagination_horizon <= 0:
            raise ValueError("num_candidates, replan_interval, and imagination_horizon must be positive")
        self.policy = policy
        self.adapter_model = adapter_model
        self.adapter_normalization = adapter_normalization
        self.weaver_action_mean = np.asarray(weaver_action_mean, dtype=np.float32)
        self.weaver_action_std = np.asarray(weaver_action_std, dtype=np.float32)
        self.weaver_model = weaver_model
        self.context_fn = context_fn
        self.history_actions_fn = history_actions_fn
        self.task_fn = task_fn
        self.current_state_fn = current_state_fn
        self.action_time_edges_fn = action_time_edges_fn or self._default_edges
        self.memory_fn = memory_fn
        self.imagination_horizon = int(imagination_horizon)
        self.num_candidates = int(num_candidates)
        self.replan_interval = int(replan_interval)
        self.device = device
        self.gamma = float(gamma)
        self.criterion = criterion
        self.score_fn = score_fn or steering_runtime.imagine_and_score
        self._remaining: np.ndarray | None = None
        self._telemetry: list[tuple[str, int, str]] = []

    def reset(self) -> None:
        self._remaining = None
        self._telemetry.clear()
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            reset()

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self._remaining is not None and len(self._remaining) > 0:
            action = np.array(self._remaining[0], copy=True)
            self._remaining = self._remaining[1:]
            return action

        pool = candidate_pool.sample_policy_candidates(
            self.policy,
            dict(observation),
            num_candidates=self.num_candidates,
            action_dim=23,
        )
        expected_digest = candidate_pool.observation_sha256(dict(observation))
        if pool.observation_digest != expected_digest:
            raise ValueError("Candidate pool observation digest changed before WEAVER scoring")
        current_state = self.current_state_fn(observation)
        edges = self.action_time_edges_fn(observation, pool.actions.shape[1])
        adapted = causal_action_adapter.adapt_candidate_pool(
            pool,
            self.adapter_model,
            self.adapter_normalization,
            current_state,
            edges,
            weaver_action_mean=self.weaver_action_mean,
            weaver_action_std=self.weaver_action_std,
            device=self.device,
            max_horizon=self.imagination_horizon,
        )
        if adapted.normalized_weaver_actions.shape[1] != self.imagination_horizon:
            raise ValueError(
                "Policy candidate chunk is shorter than the configured WEAVER imagination horizon: "
                f"available={adapted.normalized_weaver_actions.shape[1]}, required={self.imagination_horizon}"
            )
        if adapted.source_pool_observation_digest != pool.observation_digest:
            raise ValueError("Causal adapter observation provenance does not match candidate pool")
        if adapted.source_action_time_edges_sha256 != causal_action_adapter.action_time_edges_sha256(edges):
            raise ValueError("Causal adapter timing provenance does not match candidate pool")
        context = self.context_fn(observation)
        scored = self.score_fn(
            self.weaver_model,
            context=context,
            history_actions=self.history_actions_fn(observation),
            adapted_actions=adapted,
            task=self.task_fn(observation),
            n_history=int(getattr(self.weaver_model, "_n_history", 1)),
            candidate_pool=pool,
            device=self.device,
            gamma=self.gamma,
            criterion=self.criterion,
            memory_tokens=self.memory_fn(observation) if self.memory_fn is not None else None,
        )
        selected = int(scored.best_index)
        selected_chunk = np.array(pool.actions[selected], copy=True)
        execute_count = min(self.replan_interval, len(selected_chunk))
        self._remaining = selected_chunk[1:execute_count]
        self._telemetry.append((pool.candidate_ids[selected], selected, pool.actions_sha256))
        return np.array(selected_chunk[0], copy=True)

    @property
    def telemetry(self) -> SteeringTelemetry:
        return SteeringTelemetry(
            replans=len(self._telemetry),
            selected_candidate_ids=tuple(item[0] for item in self._telemetry),
            selected_indices=tuple(item[1] for item in self._telemetry),
            pool_action_digests=tuple(item[2] for item in self._telemetry),
        )

    @staticmethod
    def _default_edges(observation: Mapping[str, Any], length: int) -> np.ndarray:
        del observation
        return np.arange(length + 1, dtype=np.float64) / 30.0
