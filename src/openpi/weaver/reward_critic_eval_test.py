from __future__ import annotations

import pytest

from openpi.weaver.reward_critic_eval import evaluate_predictions
from openpi.weaver.reward_critic_eval import score_frozen_heads


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class _Torch:
    @staticmethod
    def no_grad():
        return _NoGrad()


class _Head:
    def __init__(self, value):
        self.value = value

    def __call__(self, obs, actions, task):
        del obs, actions, task
        return self.value


class _Model:
    rm = _Head([[0.1, 0.2]])
    critic = _Head([[0.4, 0.5]])


def test_reward_critic_and_same_pool_metrics_are_separate() -> None:
    metrics = evaluate_predictions(
        [[0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.2]],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
        value_pred=[[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        value_target=[[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        candidate_pred_scores=[0.1, 0.9, 0.2, 0.8],
        candidate_target_scores=[0.5, 0.1, 0.2, 0.9],
        group_ids=["a", "a", "b", "b"],
    )
    assert metrics.reward_mae == pytest.approx(0.375)
    assert metrics.value_mae == pytest.approx(1.0 / 3.0)
    assert metrics.pairwise_ranking_accuracy == pytest.approx(0.5)
    assert metrics.harmful_intervention_rate == pytest.approx(0.5)
    assert metrics.oracle_headroom == pytest.approx(0.35)
    assert metrics.selector_gap == pytest.approx(0.2)


def test_ranking_requires_complete_candidate_inputs() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        evaluate_predictions([[0.0]], [[0.0]], candidate_pred_scores=[1.0])


def test_frozen_heads_receive_no_reward_labels() -> None:
    result = score_frozen_heads(
        _Model(),
        encoded_obs={"states": "latent"},
        actions="actions",
        task_embedding="task",
        rewards=[[0.0, 1.0]],
        torch=_Torch(),
    )
    assert result["reward_pred"].shape == (1, 2)
    assert result["value_pred"].shape == (1, 2)
