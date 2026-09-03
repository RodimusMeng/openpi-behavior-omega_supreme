from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openpi.weaver.candidate_pool import CandidatePool  # noqa: E402
from openpi.weaver.causal_action_adapter import AdaptedCandidateActions  # noqa: E402
from openpi.weaver.steering_runtime import imagine_and_score  # noqa: E402
from openpi.weaver.steering_runtime import score_rollout_tensors  # noqa: E402


def test_score_equations_match_upstream_reward_slice_and_advantage():
    rewards = np.arange(2 * 6, dtype=np.float64).reshape(2, 6)
    values = np.arange(2 * 6, dtype=np.float64).reshape(2, 6) / 10

    reward_slice, reward_scores = score_rollout_tensors(
        rewards,
        n_history=2,
        gamma=0.9,
        criterion="reward",
    )
    _, advantage_scores = score_rollout_tensors(
        rewards,
        n_history=2,
        gamma=0.9,
        criterion="advantage",
        values_all=values,
    )

    np.testing.assert_array_equal(reward_slice, rewards[:, 1:-1])
    np.testing.assert_allclose(reward_scores, reward_slice.sum(axis=1))
    expected = (reward_slice * np.array([1.0, 0.9, 0.81, 0.729])).sum(axis=1) + 0.9**4 * values[:, -1] - values[:, 1]
    np.testing.assert_allclose(advantage_scores, expected)


class _FakeModel:
    _n_history = 2

    def encode_task(self, task):
        return torch.zeros(len(task["text"]), 4)

    def generate_latent_rollouts_variable_horizon(
        self, *, context, future_actions, future_states, history_actions, memory_tokens
    ):
        del future_states, history_actions, memory_tokens
        batch_size = future_actions.shape[0]
        camera_history = context["camera"].expand(batch_size, -1, -1, -1)
        state_history = context["states"].expand(batch_size, -1, -1)
        return {
            "camera": torch.cat([camera_history, future_actions[..., :1, None]], dim=1),
            "states": torch.cat([state_history, future_actions], dim=1),
        }

    def rm(self, imagined, actions, task_embed):
        del imagined, task_embed
        return actions[..., 0]

    def critic(self, imagined, actions, task_embed):
        del imagined, task_embed
        return actions[..., 0] * 0.5


def _adapted(pool: CandidatePool, horizon: int = 3) -> AdaptedCandidateActions:
    return AdaptedCandidateActions(
        canonical_deltas=np.ones((pool.num_candidates, horizon, 23), dtype=np.float32),
        normalized_weaver_actions=np.ones((pool.num_candidates, horizon, 23), dtype=np.float32),
        source_pool_actions_sha256=pool.actions_sha256,
        interval_edges_s=np.arange(horizon + 1, dtype=np.float64) / 5,
    )


def test_imagine_and_score_batches_one_pool_and_rejects_mismatched_adapter():
    pool = CandidatePool(np.ones((3, 3, 23), dtype=np.float32), "0" * 64)
    context = {"camera": torch.zeros(1, 2, 1, 1), "states": torch.zeros(1, 2, 23)}
    history = torch.zeros(1, 2, 23)
    result = imagine_and_score(
        _FakeModel(),
        context=context,
        history_actions=history,
        adapted_actions=_adapted(pool),
        task={"text": "turn on the radio"},
        n_history=2,
        candidate_pool=pool,
        device="cpu",
        gamma=0.995,
        criterion="advantage",
        use_autocast=False,
    )

    assert result.candidate_ids == pool.candidate_ids
    assert result.rewards.shape == (3, 3)
    assert result.values.shape == (3, 4)
    assert result.best_index == int(np.argmax(result.scores))
    assert result.source_pool_actions_sha256 == pool.actions_sha256
    assert result.rollout_diagnostics["states"]["future_shape"] == [3, 3, 23]
    assert result.rollout_diagnostics["states"]["finite"] is True

    mismatched = CandidatePool(np.zeros((3, 3, 23), dtype=np.float32), "0" * 64)
    with pytest.raises(ValueError, match="do not belong"):
        imagine_and_score(
            _FakeModel(),
            context=context,
            history_actions=history,
            adapted_actions=_adapted(pool),
            task={"text": "turn on the radio"},
            n_history=2,
            candidate_pool=mismatched,
            device="cpu",
            gamma=0.995,
            use_autocast=False,
        )
