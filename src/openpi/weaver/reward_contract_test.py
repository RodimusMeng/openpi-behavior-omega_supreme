import numpy as np
import pytest

from openpi.weaver import reward_contract


def test_aggregates_every_source_transition_in_half_open_interval() -> None:
    source = np.asarray([[0.1], [0.2], [-0.1], [0.4], [0.3], [0.0]], dtype=np.float32)
    result = reward_contract.aggregate_progress_rewards(source, np.asarray([0, 3, 5]))

    np.testing.assert_allclose(result.rewards, [0.2, 0.7, 0.0], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(result.transition_valid, [True, True, False])
    assert not result.rewards.flags.writeable
    assert not result.transition_valid.flags.writeable


def test_terminal_marker_is_allowed_only_on_final_source_row() -> None:
    rewards = np.zeros(7, dtype=np.float32)
    terminated = np.zeros((7, 1), dtype=bool)
    terminated[-1] = True
    result = reward_contract.aggregate_progress_rewards(
        rewards,
        np.asarray([0, 3, 6]),
        source_terminated=terminated,
        source_truncated=np.zeros(7, dtype=bool),
    )
    np.testing.assert_array_equal(result.transition_valid, [True, True, False])

    terminated[:] = False
    terminated[2] = True
    with pytest.raises(ValueError, match="final row"):
        reward_contract.aggregate_progress_rewards(
            rewards,
            np.asarray([0, 3, 6]),
            source_terminated=terminated,
        )


def test_rejects_endpoint_sampling_shapes_and_impossible_progress_span() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        reward_contract.aggregate_progress_rewards(np.zeros(5), np.asarray([0, 2, 2]))
    with pytest.raises(ValueError, match="unit-range"):
        reward_contract.aggregate_progress_rewards(
            np.asarray([0.7, 0.7, 0.0]),
            np.asarray([0, 1, 2]),
        )


def test_progress_can_regress_without_being_clipped() -> None:
    result = reward_contract.aggregate_progress_rewards(
        np.asarray([0.5, -0.25, 0.25, 0.0]),
        np.asarray([0, 2, 3]),
    )
    np.testing.assert_allclose(result.rewards, [0.25, 0.25, 0.0])
