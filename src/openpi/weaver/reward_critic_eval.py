"""Held-out reward/critic and candidate-ranking metrics.

This module deliberately accepts model predictions and offline labels as
separate arrays.  The labels may be privileged BDDL/PotentialReward targets,
but they are never an input to the selector.  It is therefore usable for a
held-out audit without silently turning privileged supervision into a
challenge-track observation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class RewardCriticMetrics:
    reward_mae: float
    reward_rmse: float
    reward_pearson: float | None
    value_mae: float | None
    value_rmse: float | None
    pairwise_ranking_accuracy: float | None
    harmful_intervention_rate: float | None
    oracle_headroom: float | None
    selector_gap: float | None
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_predictions(
    reward_pred: Any,
    reward_target: Any,
    *,
    value_pred: Any | None = None,
    value_target: Any | None = None,
    candidate_pred_scores: Any | None = None,
    candidate_target_scores: Any | None = None,
    group_ids: Sequence[Any] | None = None,
) -> RewardCriticMetrics:
    """Compute held-out reward, critic, and same-pool ranking evidence."""

    pred = _finite_array(reward_pred, name="reward_pred", ndim=2)
    target = _finite_array(reward_target, name="reward_target", ndim=2)
    if pred.shape != target.shape:
        raise ValueError(f"reward_pred and reward_target must match, got {pred.shape}/{target.shape}")
    error = pred - target
    value_mae = value_rmse = None
    if value_pred is not None or value_target is not None:
        if value_pred is None or value_target is None:
            raise ValueError("value_pred and value_target must be supplied together")
        vp = _finite_array(value_pred, name="value_pred", ndim=2)
        vt = _finite_array(value_target, name="value_target", ndim=2)
        if vp.shape != vt.shape:
            raise ValueError(f"value_pred and value_target must match, got {vp.shape}/{vt.shape}")
        value_error = vp - vt
        value_mae = float(np.abs(value_error).mean())
        value_rmse = float(np.sqrt(np.square(value_error).mean()))

    ranking = _ranking_metrics(candidate_pred_scores, candidate_target_scores, group_ids)
    return RewardCriticMetrics(
        reward_mae=float(np.abs(error).mean()),
        reward_rmse=float(np.sqrt(np.square(error).mean())),
        reward_pearson=_pearson(pred.reshape(-1), target.reshape(-1)),
        value_mae=value_mae,
        value_rmse=value_rmse,
        pairwise_ranking_accuracy=ranking[0],
        harmful_intervention_rate=ranking[1],
        oracle_headroom=ranking[2],
        selector_gap=ranking[3],
        sample_count=int(pred.shape[0]),
    )


def score_frozen_heads(
    model: Any,
    *,
    encoded_obs: Any,
    actions: Any,
    task_embedding: Any,
    rewards: Any,
    torch: Any,
) -> dict[str, np.ndarray]:
    """Run the frozen upstream RM and critic heads without updating them.

    ``encoded_obs`` may be either ground-truth encoded observations or detached
    imagined latents.  The reward labels are used only to construct the
    critic's offline V-lambda target; they are never passed to either head.
    """

    if not hasattr(model, "rm") or not hasattr(model, "critic"):
        raise ValueError("model must expose frozen upstream rm and critic heads")
    with torch.no_grad():
        reward = model.rm(encoded_obs, actions, task_embedding)
        value = model.critic(encoded_obs, actions, task_embedding)
    reward_np = _to_numpy_tensor(reward)
    value_np = _to_numpy_tensor(value)
    reward_labels = _finite_array(rewards, name="rewards", ndim=2)
    if reward_np.shape != reward_labels.shape:
        raise ValueError(f"reward prediction/label shape mismatch: {reward_np.shape}/{reward_labels.shape}")
    if value_np.ndim != 2 or value_np.shape[0] != reward_labels.shape[0] or value_np.shape[1] != reward_labels.shape[1]:
        raise ValueError(f"critic values must have shape {reward_labels.shape}, got {value_np.shape}")
    return {"reward_pred": reward_np, "reward_target": reward_labels, "value_pred": value_np}


def _ranking_metrics(
    predicted: Any | None,
    target: Any | None,
    group_ids: Sequence[Any] | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    if predicted is None and target is None:
        return None, None, None, None
    if predicted is None or target is None or group_ids is None:
        raise ValueError("candidate scores and group_ids must be supplied together")
    pred = _finite_array(predicted, name="candidate_pred_scores", ndim=1)
    truth = _finite_array(target, name="candidate_target_scores", ndim=1)
    if pred.shape != truth.shape or len(group_ids) != pred.size:
        raise ValueError("candidate score arrays and group_ids must have matching length")
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        groups[group].append(index)
    pair_total = pair_correct = 0
    harmful: list[float] = []
    headroom: list[float] = []
    gap: list[float] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                i, j = indices[left], indices[right]
                true_delta = truth[i] - truth[j]
                if true_delta == 0:
                    continue
                pair_total += 1
                if np.sign(pred[i] - pred[j]) == np.sign(true_delta):
                    pair_correct += 1
        chosen = indices[int(np.argmax(pred[indices]))]
        baseline = indices[0]
        best = max(float(truth[index]) for index in indices)
        harmful.append(float(truth[chosen] < truth[baseline]))
        headroom.append(best - float(truth[baseline]))
        gap.append(best - float(truth[chosen]))
    if not harmful:
        return None, None, None, None
    return (
        float(pair_correct / pair_total) if pair_total else None,
        float(np.mean(harmful)),
        float(np.mean(headroom)),
        float(np.mean(gap)),
    )


def _finite_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite array with ndim={ndim}, got {array.shape}")
    return array


def _to_numpy_tensor(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])
