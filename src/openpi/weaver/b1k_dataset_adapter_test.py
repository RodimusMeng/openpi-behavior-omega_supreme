import numpy as np
import pytest

from openpi.weaver import b1k_dataset_adapter as adapter
from openpi.weaver.b1k_contract import build_r1pro_contract


def _synthetic_episode(length: int = 31) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.arange(length, dtype=np.float64) / 30.0
    raw_states = np.zeros((length, 61), dtype=np.float32)
    step = np.arange(length, dtype=np.float32)

    raw_states[:, 0:3] = step[:, None] + np.asarray([0.0, 100.0, 200.0])
    raw_states[:, 53:57] = step[:, None] * 0.01 + np.arange(4, dtype=np.float32)
    raw_states[:, 3:10] = step[:, None] * 0.02 + np.arange(7, dtype=np.float32)
    raw_states[:, 24] = step + 1.0
    raw_states[:, 25] = step + 2.0
    raw_states[:, 28:35] = step[:, None] * 0.03 + np.arange(7, dtype=np.float32)
    raw_states[:, 49] = step + 3.0
    raw_states[:, 50] = step + 4.0

    policy_actions = np.arange(length * 23, dtype=np.float32).reshape(length, 23)
    return timestamps, raw_states, policy_actions


def test_two_view_feature_selection_is_contract_derived():
    assert adapter.selected_video_features() == (
        "observation.rgb.left_realsense_link_camera_0",
        "observation.rgb.zed_link_camera_0",
    )
    assert adapter.selected_video_features(include_depth=True) == (
        "observation.rgb.left_realsense_link_camera_0",
        "observation.depth_linear.left_realsense_link_camera_0",
        "observation.rgb.zed_link_camera_0",
        "observation.depth_linear.zed_link_camera_0",
    )


def test_selected_feature_validation_fails_before_decode():
    selected = adapter.selected_video_features(include_depth=True)
    features = {
        key: {"dtype": "video", "info": {"depth_min": 0.0}} if "depth" in key else {"dtype": "video"}
        for key in selected
    }
    assert adapter.validate_selected_video_features(features, include_depth=True) == selected

    del features["observation.rgb.left_realsense_link_camera_0"]
    with pytest.raises(KeyError, match="left_realsense"):
        adapter.validate_selected_video_features(features, include_depth=True)


def test_30hz_episode_is_adapted_to_a_strict_5hz_23d_contract():
    timestamps, raw_states, policy_actions = _synthetic_episode()
    episode = adapter.adapt_episode(
        episode_index=7,
        prompt="move an object",
        timestamps_s=timestamps,
        raw_states=raw_states,
        policy_actions=policy_actions,
    )

    np.testing.assert_array_equal(episode.source_frame_indices, [0, 6, 12, 18, 24, 30])
    assert episode.states.shape == (6, 23)
    assert episode.state_delta_actions.shape == (6, 23)
    assert episode.policy_actions_at_frame.shape == (6, 23)
    np.testing.assert_array_equal(episode.policy_actions_at_frame, policy_actions[[0, 6, 12, 18, 24, 30]])
    np.testing.assert_array_equal(episode.transition_valid, [True, True, True, True, True, False])
    np.testing.assert_array_equal(episode.state_delta_actions[-1], np.zeros(23, dtype=np.float32))

    # Both grippers are reductions of two source joints, not copied scalar slots.
    np.testing.assert_allclose(episode.states[:, 14], raw_states[[0, 6, 12, 18, 24, 30], 24:26].sum(axis=1))
    np.testing.assert_allclose(episode.states[:, 22], raw_states[[0, 6, 12, 18, 24, 30], 49:51].sum(axis=1))


def test_timestamp_jitter_uses_nearest_real_frames():
    timestamps, raw_states, policy_actions = _synthetic_episode(37)
    timestamps = timestamps + np.sin(np.arange(37)) * 0.002
    episode = adapter.adapt_episode(
        episode_index=0,
        prompt="test jitter",
        timestamps_s=timestamps,
        raw_states=raw_states,
        policy_actions=policy_actions,
    )

    target_times = timestamps[0] + np.arange(len(episode.source_frame_indices)) / 5.0
    for target_time, source_index in zip(target_times, episode.source_frame_indices, strict=True):
        assert abs(timestamps[source_index] - target_time) == pytest.approx(np.min(abs(timestamps - target_time)))


def test_timestamp_gap_is_rejected_instead_of_shortening_the_episode():
    timestamps, raw_states, policy_actions = _synthetic_episode(8)
    timestamps = np.asarray([0.0, 0.033, 0.066, 0.099, 0.5, 0.533, 0.566, 0.599])
    with pytest.raises(ValueError, match=r"too far|multiple WEAVER timesteps"):
        adapter.adapt_episode(
            episode_index=0,
            prompt="gap",
            timestamps_s=timestamps,
            raw_states=raw_states,
            policy_actions=policy_actions,
        )


def test_episode_timestamp_reset_cannot_cross_a_boundary():
    timestamps, raw_states, policy_actions = _synthetic_episode(7)
    concatenated_timestamps = np.concatenate([timestamps, timestamps])
    with pytest.raises(ValueError, match="strictly increasing"):
        adapter.adapt_episode(
            episode_index=0,
            prompt="two episodes",
            timestamps_s=concatenated_timestamps,
            raw_states=np.concatenate([raw_states, raw_states]),
            policy_actions=np.concatenate([policy_actions, policy_actions]),
        )


def test_state_delta_matches_official_plain_subtraction():
    states = np.zeros((2, 23), dtype=np.float32)
    states[0, 0] = 1.0
    states[1, 0] = 3.0
    states[0, 3] = np.pi - 0.1
    states[1, 3] = -np.pi + 0.1
    states[0, 14] = 0.9
    states[1, 14] = 0.1

    actions = adapter.observed_state_delta_actions(states)

    assert actions[0, 0] == pytest.approx(2.0)
    assert actions[0, 3] == pytest.approx(-2 * np.pi + 0.2, abs=1e-6)
    assert actions[0, 14] == pytest.approx(-0.8)
    np.testing.assert_array_equal(actions[1], np.zeros(23))


def test_episode_split_is_deterministic_order_independent_and_disjoint():
    tasks = {
        4: ("task-a",),
        2: ("task-a",),
        9: ("task-a",),
        7: ("task-b",),
        1: ("task-b",),
        8: ("held-out",),
    }
    first = adapter.deterministic_episode_split(tasks, val_fraction=0.25, seed=12, held_out_tasks=("held-out",))
    second = adapter.deterministic_episode_split(
        dict(reversed(list(tasks.items()))), val_fraction=0.25, seed=12, held_out_tasks=("held-out",)
    )

    assert first == second
    assert set(first) == set(tasks)
    assert first[8] == "test"
    assert {first[index] for index in (4, 2, 9)} == {"train", "val"}
    assert {first[index] for index in (7, 1)} == {"train", "val"}


def test_episode_split_rejects_unknown_held_out_task():
    with pytest.raises(ValueError, match=r"Unknown held_out_tasks.*typo"):
        adapter.deterministic_episode_split(
            {0: ("task-a",), 1: ("task-b",)},
            held_out_tasks=("typo",),
        )


def test_multi_task_episode_is_rejected_instead_of_silently_taking_first():
    with pytest.raises(ValueError, match="exactly one task"):
        adapter.deterministic_episode_split({0: ("task-a", "task-b")})


def test_task_diverse_smoke_sample_is_deterministic_and_order_independent():
    episode_tasks = {
        4: ("task-b",),
        0: ("task-a",),
        5: ("task-b",),
        2: ("task-a",),
        9: ("task-c",),
        8: ("task-c",),
    }
    first = adapter.deterministic_task_episode_sample(
        episode_tasks,
        num_tasks=2,
        episodes_per_task=2,
        seed=7,
    )
    second = adapter.deterministic_task_episode_sample(
        dict(reversed(tuple(episode_tasks.items()))),
        num_tasks=2,
        episodes_per_task=2,
        seed=7,
    )
    assert first == second
    assert len(first) == 4
    assert len({episode_tasks[index][0] for index in first}) == 2


def test_task_diverse_smoke_sample_rejects_impossible_request():
    with pytest.raises(ValueError, match="only 1"):
        adapter.deterministic_task_episode_sample(
            {0: ("task-a",)},
            num_tasks=1,
            episodes_per_task=2,
        )


def test_manifest_names_both_non_interchangeable_action_representations():
    timestamps, raw_states, policy_actions = _synthetic_episode()
    episode = adapter.adapt_episode(
        episode_index=3,
        prompt="manifest task",
        timestamps_s=timestamps,
        raw_states=raw_states,
        policy_actions=policy_actions,
    )
    manifest = adapter.episode_manifest(
        episode,
        split="train",
        dataset_repo_id="b1k-v3again",
        dataset_metadata_sha256="a" * 64,
    )

    assert manifest["schema_version"] == adapter.SCHEMA_VERSION
    assert manifest["training_action"]["representation"] == adapter.STATE_DELTA_ACTION_REPRESENTATION
    assert manifest["training_action"]["causal"] is False
    assert manifest["training_action"]["online_steering_compatible"] is False
    assert manifest["source_policy_action"]["representation"] == adapter.POLICY_ACTION_REPRESENTATION
    assert manifest["source_policy_action"]["full_rate_preserved"] is True
    assert manifest["source_policy_action"]["five_hz_aggregation"] == "undefined"
    assert manifest["online_steering"]["compatible"] is False
    assert "causal_30hz_policy_action_aggregation" in manifest["online_steering"]["required_next_step"]
    assert manifest["views"]["wrist_left"]["rgb"].endswith("left_realsense_link_camera_0")
    assert manifest["views"]["exterior_1_left"]["rgb"].endswith("zed_link_camera_0")
    assert len(manifest["semantic_sha256"]) == 64


def test_30hz_rewards_are_summed_over_each_5hz_transition():
    timestamps, raw_states, policy_actions = _synthetic_episode()
    rewards = np.arange(len(timestamps), dtype=np.float32) / 1000.0
    terminated = np.zeros(len(timestamps), dtype=bool)
    terminated[24:] = True
    rewards[25:] = 0.0
    episode = adapter.adapt_episode(
        episode_index=0,
        prompt="aggregate progress",
        timestamps_s=timestamps,
        raw_states=raw_states,
        policy_actions=policy_actions,
        rewards=rewards,
        terminated=terminated,
        truncated=np.zeros(len(timestamps), dtype=bool),
    )

    expected = [rewards[start:end].sum() for start, end in zip([0, 6, 12, 18, 24], [6, 12, 18, 24, 30], strict=True)]
    np.testing.assert_allclose(episode.rewards, [*expected, 0.0], rtol=0, atol=1e-6)
    manifest = adapter.episode_manifest(
        episode,
        split="train",
        dataset_repo_id="b1k-v3again",
        dataset_metadata_sha256="a" * 64,
    )
    assert manifest["reward"]["representation"] == "b1k_behavior_recorded_transition_reward_5hz_v3"
    assert manifest["reward"]["allowed_online_input"] is False


def test_terminal_suffix_is_cropped_after_terminal_target_and_keeps_target_observation():
    """Only the interval containing the final done suffix may remain.

    The LeRobot row carrying ``done`` is the transition *into* the next
    observation.  Consequently the adapter keeps the first sampled frame at
    or after ``done + 1`` as the terminal target, but does not expose any
    later absorbing-suffix transitions.
    """
    timestamps, raw_states, policy_actions = _synthetic_episode(31)
    terminated = np.zeros(31, dtype=bool)
    terminated[13:] = True
    # Make post-terminal observations visibly distinct so an accidental tail
    # leak cannot be hidden by a constant absorbing state.
    raw_states[19:, 0] += 10000.0
    rewards = np.zeros(31, dtype=np.float32)

    episode = adapter.adapt_episode(
        episode_index=0,
        prompt="terminal crop",
        timestamps_s=timestamps,
        raw_states=raw_states,
        policy_actions=policy_actions,
        rewards=rewards,
        terminated=terminated,
    )

    np.testing.assert_array_equal(episode.source_frame_indices, [0, 6, 12, 18])
    assert episode.states.shape[0] == 4
    assert episode.transition_valid.tolist() == [True, True, True, False]
    # The target frame 18 is retained; rows 19..30 are not represented.
    assert episode.states[-1, 0] == pytest.approx(raw_states[18, 0])
    assert episode.policy_actions_at_frame[-1, 0] == pytest.approx(policy_actions[18, 0])


def test_intermediate_terminal_run_does_not_crop_episode_or_drop_rewards():
    timestamps, raw_states, policy_actions = _synthetic_episode(31)
    terminated = np.zeros(31, dtype=bool)
    terminated[13:16] = True
    terminated[24:] = True
    rewards = np.zeros(31, dtype=np.float32)
    rewards[13] = 1.0
    rewards[16] = -1.0
    rewards[24] = 1.0

    episode = adapter.adapt_episode(
        episode_index=147,
        prompt="intermediate terminal run",
        timestamps_s=timestamps,
        raw_states=raw_states,
        policy_actions=policy_actions,
        rewards=rewards,
        terminated=terminated,
    )

    np.testing.assert_array_equal(episode.source_frame_indices, [0, 6, 12, 18, 24, 30])
    np.testing.assert_allclose(episode.rewards, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(episode.transition_valid, [True, True, True, True, True, False])


def test_terminal_transition_without_a_later_observation_is_rejected():
    timestamps, raw_states, policy_actions = _synthetic_episode(31)
    terminated = np.zeros(31, dtype=bool)
    terminated[-1] = True
    with pytest.raises(ValueError, match="no later 5 Hz observation"):
        adapter.adapt_episode(
            episode_index=0,
            prompt="terminal at source end",
            timestamps_s=timestamps,
            raw_states=raw_states,
            policy_actions=policy_actions,
            rewards=np.zeros(31, dtype=np.float32),
            terminated=terminated,
        )


def test_nonzero_reward_after_absorbing_terminal_suffix_is_rejected():
    timestamps, raw_states, policy_actions = _synthetic_episode(31)
    terminated = np.zeros(31, dtype=bool)
    terminated[13:] = True
    rewards = np.zeros(31, dtype=np.float32)
    rewards[20] = 1.0
    with pytest.raises(ValueError, match="zero recorded reward after the final terminal"):
        adapter.adapt_episode(
            episode_index=0,
            prompt="bad terminal reward",
            timestamps_s=timestamps,
            raw_states=raw_states,
            policy_actions=policy_actions,
            rewards=rewards,
            terminated=terminated,
        )


def test_adapted_arrays_are_immutable():
    timestamps, raw_states, policy_actions = _synthetic_episode()
    episode = adapter.adapt_episode(
        episode_index=0,
        prompt="immutable",
        timestamps_s=timestamps,
        raw_states=raw_states,
        policy_actions=policy_actions,
    )
    with pytest.raises(ValueError, match="read-only"):
        episode.states[0, 0] = 1.0


def test_contract_still_derives_23d_from_registry():
    contract = build_r1pro_contract()
    assert contract.state_dim == 23
    assert contract.action_dim == 23
