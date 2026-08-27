from __future__ import annotations

import numpy as np
import pytest

from openpi.weaver import b1k_dataset_adapter
from openpi.weaver import causal_action_adapter as adapter


def _sample(*, split: str = "train", steps: int = 6, offset: float = 0.0) -> adapter.AdapterSample:
    return adapter.AdapterSample(
        current_state=np.arange(23, dtype=np.float32) + offset,
        policy_actions=np.full((steps, 23), offset + 2.0, dtype=np.float32),
        action_durations_s=np.full(steps, 0.2 / steps, dtype=np.float32),
        target_state_delta=np.full(23, offset + 0.25, dtype=np.float32),
        split=split,
        episode_index=3,
        interval_index=int(offset),
    )


def test_normalization_is_train_only_and_collate_preserves_variable_sequences():
    samples = (_sample(steps=5), _sample(steps=7, offset=1.0))
    normalization = adapter.compute_normalization(samples)
    batch = adapter.collate_samples(samples, normalization)

    assert batch["current_state"].shape == (2, 23)
    assert batch["policy_actions"].shape == (2, 7, 23)
    assert batch["action_durations_s"].shape == (2, 7)
    assert batch["mask"].tolist() == [[True] * 5 + [False, False], [True] * 7]
    assert batch["target_state_delta"].shape == (2, 23)
    assert np.isfinite(batch["policy_actions"]).all()

    with pytest.raises(ValueError, match="train split only"):
        adapter.compute_normalization((_sample(split="val"),))


def test_candidate_segmentation_uses_real_duration_and_never_extends_chunk():
    actions = np.arange(2 * 32 * 23, dtype=np.float32).reshape(2, 32, 23)
    edges = np.arange(33, dtype=np.float64) / 30.0

    segmented = adapter.segment_candidate_actions(actions, edges, target_hz=5.0)

    assert segmented.policy_actions.shape == (2, 5, 6, 23)
    assert segmented.horizon == 5
    assert segmented.mask.all()
    np.testing.assert_allclose(segmented.action_durations_s.sum(axis=2), 0.2, atol=1e-7)
    np.testing.assert_array_equal(segmented.policy_actions[:, 0], actions[:, :6])
    # The two trailing 30 Hz commands form only a partial interval and are not
    # repeated/padded to manufacture a sixth WEAVER action.
    assert segmented.interval_edges_s[-1] == pytest.approx(1.0)


def test_candidate_segmentation_splits_a_command_crossing_wm_boundary():
    actions = np.zeros((1, 3, 23), dtype=np.float32)
    actions[:, 0] = 1
    actions[:, 1] = 2
    actions[:, 2] = 3
    edges = np.array([0.0, 0.15, 0.27, 0.4], dtype=np.float64)

    segmented = adapter.segment_candidate_actions(actions, edges, target_hz=5.0)

    assert segmented.policy_actions.shape == (1, 2, 2, 23)
    np.testing.assert_allclose(segmented.action_durations_s[0, 0, :2], [0.15, 0.05])
    np.testing.assert_allclose(segmented.action_durations_s[0, 1, :2], [0.07, 0.13])
    assert segmented.policy_actions[0, 0, 1, 0] == 2
    assert segmented.policy_actions[0, 1, 0, 0] == 2


def test_state_update_wraps_only_audited_joint_position_coordinates():
    state = np.zeros((1, 23), dtype=np.float32)
    delta = np.zeros_like(state)
    angular = b1k_dataset_adapter.angular_state_indices()
    delta[:, angular] = 4 * np.pi + 0.25
    delta[:, 0] = 4 * np.pi + 0.25

    result = adapter.apply_state_delta(state, delta)

    np.testing.assert_allclose(result[:, angular], 0.25, atol=1e-6)
    # base_qvel is not an angle-valued position and must not be wrapped.
    assert result[0, 0] == pytest.approx(4 * np.pi + 0.25)


def test_plain_state_delta_and_wrapped_update_cross_pi_boundary():
    angular = b1k_dataset_adapter.angular_state_indices()
    before = np.zeros((2, 23), dtype=np.float32)
    before[0, angular] = np.pi - 0.05
    before[1, angular] = -np.pi + 0.05

    # Offline WEAVER targets use the audited plain-subtraction convention.
    deltas = b1k_dataset_adapter.observed_state_delta_actions(before)
    np.testing.assert_allclose(deltas[0, angular], -2 * np.pi + 0.1, atol=1e-6)

    # Autoregressive state advancement still canonicalizes angular positions.
    updated = adapter.apply_state_delta(before[0], deltas[0])
    angular_array = np.asarray(angular)
    np.testing.assert_allclose(updated[angular_array], before[1, angular_array], atol=1e-6)


def test_fixed_rate_candidate_horizon_is_checked_before_rollout():
    assert adapter.available_world_model_horizon(32, control_hz=30, target_hz=5) == 5
    assert adapter.require_world_model_horizon(32, 5, control_hz=30, target_hz=5) == 5
    with pytest.raises(ValueError, match=r"provides 5 complete intervals.*required=8"):
        adapter.require_world_model_horizon(32, 8, control_hz=30, target_hz=5)


def test_adapter_sample_rejects_nonpositive_command_duration():
    with pytest.raises(ValueError, match="positive"):
        adapter.AdapterSample(
            current_state=np.zeros(23),
            policy_actions=np.zeros((1, 23)),
            action_durations_s=np.zeros(1),
            target_state_delta=np.zeros(23),
            split="train",
            episode_index=0,
            interval_index=0,
        )
