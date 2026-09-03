"""Same-candidate-pool WEAVER reward/advantage scoring for B1K.

This module is deliberately a model-facing primitive, not a radio/server
launcher. It accepts immutable post-transform policy candidates after the
causal adapter has produced normalized WEAVER actions, runs one batched frozen
upstream rollout, and returns reward-only or paper-style advantage scores. Any
oracle score is an explicitly privileged diagnostic label and is never passed
to WEAVER.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import dataclasses
import math
from typing import Any, Literal

import numpy as np

from openpi.weaver.candidate_pool import CandidatePool
from openpi.weaver.causal_action_adapter import AdaptedCandidateActions

SelectionCriterion = Literal["reward", "advantage"]


@dataclasses.dataclass(frozen=True)
class ScoredCandidatePool:
    candidate_ids: tuple[str, ...]
    rewards: np.ndarray
    values: np.ndarray | None
    scores: np.ndarray
    criterion: SelectionCriterion
    gamma: float
    source_pool_actions_sha256: str
    imagined_horizon: int
    oracle_scores: np.ndarray | None = None
    rollout_diagnostics: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        rewards = _finite(self.rewards, name="rewards", ndim=2)
        scores = _finite(self.scores, name="scores", ndim=1)
        if rewards.shape[0] != len(self.candidate_ids) or scores.shape != (rewards.shape[0],):
            raise ValueError("candidate IDs/rewards/scores have inconsistent candidate axis")
        if self.values is not None:
            values = _finite(self.values, name="values", ndim=2)
            if values.shape != (rewards.shape[0], rewards.shape[1] + 1):
                raise ValueError("critic values must have one bootstrap value after the reward sequence")
            object.__setattr__(self, "values", values)
        if self.oracle_scores is not None:
            oracle = _finite(self.oracle_scores, name="oracle_scores", ndim=1)
            if oracle.shape != scores.shape:
                raise ValueError("oracle_scores must have one privileged score per candidate")
            object.__setattr__(self, "oracle_scores", oracle)
        if self.criterion not in {"reward", "advantage"}:
            raise ValueError(f"Unsupported selection criterion {self.criterion!r}")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be finite and in [0,1]")
        if self.imagined_horizon != rewards.shape[1] or self.imagined_horizon <= 0:
            raise ValueError("imagined_horizon must match a positive reward horizon")
        if len(self.source_pool_actions_sha256) != 64:
            raise ValueError("source_pool_actions_sha256 must be a SHA-256 digest")
        if self.rollout_diagnostics is not None:
            diagnostics = {str(key): dict(value) for key, value in self.rollout_diagnostics.items()}
            if not diagnostics:
                raise ValueError("rollout_diagnostics must be non-empty when supplied")
            for key, value in diagnostics.items():
                shape = value.get("shape")
                future_shape = value.get("future_shape")
                if not isinstance(shape, (list, tuple)) or not isinstance(future_shape, (list, tuple)):
                    raise ValueError(f"rollout diagnostic {key!r} must contain shape and future_shape")
                if not shape or not future_shape or int(shape[0]) != rewards.shape[0]:
                    raise ValueError(f"rollout diagnostic {key!r} has the wrong candidate axis")
                if int(future_shape[0]) != rewards.shape[0] or int(future_shape[1]) != self.imagined_horizon:
                    raise ValueError(f"rollout diagnostic {key!r} has the wrong future horizon")
                if value.get("finite") is not True:
                    raise ValueError(f"rollout diagnostic {key!r} contains non-finite predictions")
            object.__setattr__(self, "rollout_diagnostics", diagnostics)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "scores", scores)

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.scores))

    @property
    def best_candidate_id(self) -> str:
        return self.candidate_ids[self.best_index]


def score_rollout_tensors(
    rewards_all: Any,
    *,
    n_history: int,
    gamma: float,
    criterion: SelectionCriterion,
    values_all: Any | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Score tensors with the exact upstream steering slice/equations."""

    rewards = _to_numpy(rewards_all).astype(np.float64, copy=False)
    if rewards.ndim != 2 or rewards.shape[1] <= n_history:
        raise ValueError(f"rewards_all must have shape (N,T) with T>{n_history}, got {rewards.shape}")
    if isinstance(n_history, bool) or not isinstance(n_history, int) or n_history <= 0:
        raise ValueError("n_history must be a positive integer")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0,1]")
    # Upstream steer_pi_policy.py: rewards[:, n_history-1:-1].
    chunk_rewards = rewards[:, n_history - 1 : -1]
    if chunk_rewards.shape[1] == 0:
        raise ValueError("Rollout contains no scoreable reward transitions")
    if criterion == "reward":
        return chunk_rewards, chunk_rewards.sum(axis=1)
    if criterion != "advantage":
        raise ValueError(f"Unsupported selection criterion {criterion!r}")
    if values_all is None:
        raise ValueError("Advantage scoring requires critic values")
    values = _to_numpy(values_all).astype(np.float64, copy=False)
    if values.shape != rewards.shape:
        raise ValueError(f"values_all must match rewards_all shape, got {values.shape} vs {rewards.shape}")
    horizon = chunk_rewards.shape[1]
    powers = gamma ** np.arange(horizon, dtype=np.float64)
    initial = values[:, n_history - 1]
    bootstrap = values[:, -1]
    scores = (chunk_rewards * powers).sum(axis=1) + gamma**horizon * bootstrap - initial
    return chunk_rewards, scores


def imagine_and_score(
    model: Any,
    *,
    context: Mapping[str, Any],
    history_actions: Any,
    adapted_actions: AdaptedCandidateActions,
    task: Mapping[str, Any],
    n_history: int,
    candidate_pool: CandidatePool,
    device: str,
    gamma: float,
    criterion: SelectionCriterion = "reward",
    memory_tokens: Any | None = None,
    future_state_placeholder: Any | None = None,
    oracle_scores: Sequence[float] | None = None,
    use_autocast: bool = True,
) -> ScoredCandidatePool:
    """Run one batched upstream rollout for an immutable candidate pool."""

    torch = _require_torch()
    if adapted_actions.source_pool_actions_sha256 != candidate_pool.actions_sha256:
        raise ValueError("Causal WEAVER actions do not belong to this immutable candidate pool")
    actions = torch.as_tensor(
        np.array(adapted_actions.normalized_weaver_actions, copy=True),
        dtype=torch.float32,
        device=device,
    )
    if actions.ndim != 3 or actions.shape[0] != candidate_pool.num_candidates:
        raise ValueError(
            f"adapted WEAVER actions must have shape (N,H,A) with N={candidate_pool.num_candidates}, got {tuple(actions.shape)}"
        )
    if actions.shape[2] <= 0:
        raise ValueError("normalized_weaver_actions action dimension must be positive")
    batch_size, horizon, action_dim = actions.shape
    if int(getattr(model, "_n_history", n_history)) != n_history:
        raise ValueError("model n_history and scoring n_history differ")
    if not isinstance(context, Mapping) or "states" not in context:
        raise ValueError("context must contain encoded states")
    context_batch = {
        key: _expand_candidate(value, batch_size, torch=torch, device=device) for key, value in context.items()
    }
    hist = _to_device(history_actions, torch=torch, device=device)
    if hist.ndim != 3 or hist.shape[0] not in {1, batch_size} or hist.shape[1] != n_history:
        raise ValueError("history_actions must have shape (1|N,n_history,A)")
    if hist.shape[0] == 1:
        hist = hist.expand(batch_size, -1, -1)
    if hist.shape[2] != action_dim:
        raise ValueError("history action dimension differs from candidate action dimension")
    if future_state_placeholder is None:
        future_states = torch.zeros(
            batch_size,
            horizon,
            int(context_batch["states"].shape[-1]),
            device=device,
            dtype=context_batch["states"].dtype,
        )
    else:
        future_states = _to_device(future_state_placeholder, torch=torch, device=device)
        if future_states.ndim != 3 or future_states.shape[:2] != (batch_size, horizon):
            raise ValueError("future_state_placeholder must have shape (N,H,n_states)")
    task_batch = _expand_task(task, batch_size)
    task_embed = model.encode_task(task_batch)
    if memory_tokens is not None:
        memory_tokens = _to_device(memory_tokens, torch=torch, device=device)
        if memory_tokens.shape[0] not in {1, batch_size}:
            raise ValueError("memory_tokens candidate axis must be 1 or N")
        if memory_tokens.shape[0] == 1:
            memory_tokens = memory_tokens.expand(batch_size, *memory_tokens.shape[1:])
    actions_full = torch.cat([hist, actions], dim=1)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_autocast and str(device).startswith("cuda")
        else nullcontext()
    )
    with torch.no_grad(), autocast:
        imagined = model.generate_latent_rollouts_variable_horizon(
            context=context_batch,
            future_actions=actions,
            future_states=future_states,
            history_actions=hist,
            memory_tokens=memory_tokens,
        )
        rewards_all = model.rm(imagined, actions_full, task_embed)
        values_all = model.critic(imagined, actions_full, task_embed) if criterion == "advantage" else None
    chunk_rewards, scores = score_rollout_tensors(
        rewards_all,
        n_history=n_history,
        gamma=gamma,
        criterion=criterion,
        values_all=values_all,
    )
    rollout_diagnostics = _summarize_rollout(imagined, n_history=n_history, torch=torch)
    oracle = None if oracle_scores is None else np.asarray(oracle_scores, dtype=np.float64)
    return ScoredCandidatePool(
        candidate_ids=candidate_pool.candidate_ids,
        rewards=chunk_rewards,
        values=None if values_all is None else _to_numpy(values_all)[:, n_history - 1 :],
        scores=scores,
        criterion=criterion,
        gamma=gamma,
        source_pool_actions_sha256=candidate_pool.actions_sha256,
        imagined_horizon=horizon,
        oracle_scores=oracle,
        rollout_diagnostics=rollout_diagnostics,
    )


def _summarize_rollout(
    imagined: Mapping[str, Any],
    *,
    n_history: int,
    torch: Any,
) -> dict[str, dict[str, Any]]:
    """Return small, JSON-safe diagnostics without retaining rollout tensors."""

    result: dict[str, dict[str, Any]] = {}
    for key, raw_value in imagined.items():
        value = raw_value if hasattr(raw_value, "detach") else torch.as_tensor(raw_value)
        if value.ndim < 3 or value.shape[1] <= n_history:
            raise ValueError(f"Imagined rollout {key!r} must have shape (N,n_history+H,...), got {tuple(value.shape)}")
        future = value[:, n_history:].detach().float()
        flat = future.reshape(future.shape[0], -1)
        if flat.shape[0] > 1:
            pairwise = torch.pdist(flat, p=2) / math.sqrt(flat.shape[1])
            pairwise_min = float(pairwise.min().cpu())
            pairwise_max = float(pairwise.max().cpu())
        else:
            pairwise_min = 0.0
            pairwise_max = 0.0
        result[str(key)] = {
            "shape": [int(item) for item in value.shape],
            "future_shape": [int(item) for item in future.shape],
            "finite": bool(torch.isfinite(future).all().item()),
            "future_min": float(future.min().cpu()),
            "future_max": float(future.max().cpu()),
            "candidate_pairwise_rms_min": pairwise_min,
            "candidate_pairwise_rms_max": pairwise_max,
        }
    if not result:
        raise ValueError("WEAVER returned an empty imagined rollout")
    return result


def _expand_candidate(value: Any, batch_size: int, *, torch: Any, device: str) -> Any:
    tensor = _to_device(value, torch=torch, device=device)
    if tensor.ndim < 2 or tensor.shape[0] not in {1, batch_size}:
        raise ValueError("context tensors must have candidate axis 1 or N")
    return tensor if tensor.shape[0] == batch_size else tensor.expand(batch_size, *tensor.shape[1:])


def _expand_task(task: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    if not isinstance(task, Mapping) or not ({"text", "features"} & set(task)):
        raise ValueError("task must contain text or precomputed features")
    result = dict(task)
    if "text" in result:
        text = result["text"]
        if isinstance(text, str):
            result["text"] = [text] * batch_size
        elif isinstance(text, Sequence) and len(text) == 1:
            result["text"] = list(text) * batch_size
        elif not isinstance(text, Sequence) or len(text) != batch_size:
            raise ValueError("task text must contain one prompt or one prompt per candidate")
    if "features" in result:
        features = result["features"]
        if not hasattr(features, "shape") or features.shape[0] not in {1, batch_size}:
            raise ValueError("task features must have candidate axis 1 or N")
        if features.shape[0] == 1:
            result["features"] = features.expand(batch_size, *features.shape[1:])
    return result


def _to_device(value: Any, *, torch: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if str(getattr(value, "dtype", "")) == "torch.bfloat16":
        value = value.float()
    return np.asarray(value)


def _finite(value: Any, *, name: str, ndim: int) -> np.ndarray:
    result = _to_numpy(value)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite rank-{ndim}, got {result.shape}")
    return result.astype(np.float64, copy=True)


def _require_torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("WEAVER steering requires PyTorch") from exc
    return torch
