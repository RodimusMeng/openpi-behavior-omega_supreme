import numpy as np
import pytest

from openpi.weaver import candidate_pool


class _FakePolicy:
    def __init__(self):
        self.calls = 0

    def infer(self, obs, *, noise=None):
        self.calls += 1
        offset = self.calls if noise is None else np.asarray(noise)[0, 0]
        return {"actions": np.full((2, 23), offset, dtype=np.float32)}


def test_sample_policy_candidates_reuses_infer_and_freezes_post_transform_chunks():
    policy = _FakePolicy()
    observation = {"state": np.arange(23, dtype=np.float32), "prompt": "test"}
    noises = np.stack([np.full((2, 32), value, dtype=np.float32) for value in (3, 7)])

    pool = candidate_pool.sample_policy_candidates(
        policy,
        observation,
        num_candidates=2,
        noises=noises,
    )

    assert policy.calls == 2
    assert pool.actions.shape == (2, 2, 23)
    assert pool.noises.shape == (2, 2, 32)
    assert pool.candidate_ids == ("candidate-0000", "candidate-0001")
    assert len(pool.actions_sha256) == 64
    assert pool.actions_sha256 == candidate_pool.CandidatePool(
        actions=pool.actions.copy(), observation_digest=pool.observation_digest
    ).actions_sha256
    np.testing.assert_array_equal(pool.actions[:, 0, 0], [3, 7])
    with pytest.raises(ValueError, match="read-only"):
        pool.actions[0, 0, 0] = 0


def test_single_candidate_is_exactly_the_existing_single_infer_result():
    observation = {"state": np.zeros(23, dtype=np.float32)}
    old_policy = _FakePolicy()
    expected = old_policy.infer(observation)["actions"]
    new_policy = _FakePolicy()

    pool = candidate_pool.sample_policy_candidates(new_policy, observation, num_candidates=1)

    np.testing.assert_array_equal(pool.actions[0], expected)
    assert new_policy.calls == 1


def test_internal_padded_action_dimension_is_rejected():
    class _PaddedPolicy:
        def infer(self, obs, *, noise=None):
            return {"actions": np.zeros((2, 32), dtype=np.float32)}

    with pytest.raises(ValueError, match="post-transform 23-D"):
        candidate_pool.sample_policy_candidates(_PaddedPolicy(), {}, num_candidates=1)


def test_observation_digest_is_order_independent_but_content_sensitive():
    first = {"prompt": "task", "state": np.arange(3, dtype=np.float32)}
    reordered = {"state": np.arange(3, dtype=np.float32), "prompt": "task"}
    changed = {"prompt": "task", "state": np.arange(3, dtype=np.float32) + 1}

    assert candidate_pool.observation_sha256(first) == candidate_pool.observation_sha256(reordered)
    assert candidate_pool.observation_sha256(first) != candidate_pool.observation_sha256(changed)


def test_weaver_reward_and_advantage_scores_match_released_formula():
    rewards = np.asarray([[1.0, 2.0], [2.0, 0.0]], dtype=np.float32)

    reward_scores = candidate_pool.weaver_selection_scores(rewards)
    advantage_scores = candidate_pool.weaver_selection_scores(
        rewards,
        gamma=0.5,
        initial_values=np.asarray([0.25, 1.0]),
        bootstrap_values=np.asarray([4.0, 8.0]),
    )

    np.testing.assert_allclose(reward_scores, [3.0, 2.0])
    np.testing.assert_allclose(advantage_scores, [2.75, 3.0])


def test_all_baselines_consume_one_immutable_candidate_pool():
    pool = candidate_pool.CandidatePool(
        actions=np.zeros((3, 2, 23), dtype=np.float32),
        observation_digest="0" * 64,
    )

    selected = candidate_pool.select_same_pool_baselines(
        pool,
        weaver_scores=np.asarray([0.1, 0.9, 0.2]),
        oracle_scores=np.asarray([0.3, 0.1, 1.0]),
        rng=np.random.default_rng(0),
    )

    assert selected.top1 == 0
    assert selected.random in range(pool.num_candidates)
    assert selected.weaver == 1
    assert selected.oracle == 2


@pytest.mark.parametrize("num_candidates", [0, -1, True])
def test_sample_policy_candidates_rejects_invalid_pool_size(num_candidates):
    with pytest.raises(ValueError, match="positive integer"):
        candidate_pool.sample_policy_candidates(_FakePolicy(), {}, num_candidates=num_candidates)
