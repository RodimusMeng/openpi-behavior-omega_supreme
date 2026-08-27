import numpy as np
import pytest

from openpi.weaver import b1k_contract


def test_r1pro_contract_is_derived_from_registered_robot_config():
    contract = b1k_contract.build_r1pro_contract()

    assert contract.state_dim == 23
    assert contract.action_dim == 23
    assert [(component.name, component.compact_indices) for component in contract.state_components] == [
        ("base_qvel", (0, 1, 2)),
        ("trunk_qpos", (3, 4, 5, 6)),
        ("left_arm_qpos", (7, 8, 9, 10, 11, 12, 13)),
        ("left_gripper_qpos", (14,)),
        ("right_arm_qpos", (15, 16, 17, 18, 19, 20, 21)),
        ("right_gripper_qpos", (22,)),
    ]
    assert contract.delta_mappings == (
        ((3, 4, 5, 6), (3, 4, 5, 6)),
        ((7, 8, 9, 10, 11, 12, 13), (7, 8, 9, 10, 11, 12, 13)),
        ((15, 16, 17, 18, 19, 20, 21), (15, 16, 17, 18, 19, 20, 21)),
    )


def test_extract_state_reuses_gripper_reduction_and_raw_indices():
    contract = b1k_contract.build_r1pro_contract()
    raw = np.arange(61, dtype=np.float32)

    state = contract.extract_state(raw)

    np.testing.assert_array_equal(state[:3], raw[:3])
    np.testing.assert_array_equal(state[3:7], raw[53:57])
    np.testing.assert_array_equal(state[7:14], raw[3:10])
    assert state[14] == raw[24] + raw[25]
    np.testing.assert_array_equal(state[15:22], raw[28:35])
    assert state[22] == raw[49] + raw[50]


def test_environment_and_training_action_spaces_are_explicitly_distinct_and_invertible():
    contract = b1k_contract.build_r1pro_contract()
    current_state = np.arange(23, dtype=np.float32)
    environment_actions = np.stack([current_state + 10, current_state + 20])

    training_actions = contract.environment_to_openpi_training_actions(environment_actions, current_state)

    np.testing.assert_array_equal(training_actions[:, :3], environment_actions[:, :3])
    np.testing.assert_array_equal(training_actions[:, 3:14], environment_actions[:, 3:14] - current_state[3:14])
    np.testing.assert_array_equal(training_actions[:, 14], environment_actions[:, 14])
    np.testing.assert_array_equal(training_actions[:, 15:22], environment_actions[:, 15:22] - current_state[15:22])
    np.testing.assert_array_equal(training_actions[:, 22], environment_actions[:, 22])
    np.testing.assert_array_equal(
        contract.openpi_training_to_environment_actions(training_actions, current_state),
        environment_actions,
    )


def test_selective_delta_mapping_broadcasts_over_batch_candidate_and_time_axes():
    contract = b1k_contract.build_r1pro_contract()
    current_state = np.stack(
        [np.arange(23, dtype=np.float32), np.arange(23, dtype=np.float32) + 100],
    )
    environment_actions = np.zeros((2, 3, 4, 23), dtype=np.float32)
    environment_actions[..., 3:14] = 500
    environment_actions[..., 15:22] = 700

    training_actions = contract.environment_to_openpi_training_actions(environment_actions, current_state)

    expected_left = environment_actions[..., 3:14] - current_state[:, None, None, 3:14]
    expected_right = environment_actions[..., 15:22] - current_state[:, None, None, 15:22]
    np.testing.assert_array_equal(training_actions[..., 3:14], expected_left)
    np.testing.assert_array_equal(training_actions[..., 15:22], expected_right)
    np.testing.assert_array_equal(
        contract.openpi_training_to_environment_actions(training_actions, current_state),
        environment_actions,
    )


def test_released_two_view_mapping_reuses_left_wrist_and_head():
    contract = b1k_contract.build_r1pro_contract()
    online = {view.online_rgb_key: np.full((2, 3, 3), index) for index, view in enumerate(contract.camera_views)}

    views = contract.extract_online_views(online)

    assert [(view.weaver_key, view.camera_name) for view in contract.camera_views] == [
        ("wrist_left", "left_wrist"),
        ("exterior_1_left", "head"),
    ]
    np.testing.assert_array_equal(views["wrist_left"], 0)
    np.testing.assert_array_equal(views["exterior_1_left"], 1)


def test_world_model_frame_indices_use_timestamps_not_franka_stride():
    timestamps = np.arange(31, dtype=np.float64) / 30.0

    indices = b1k_contract.world_model_frame_indices(timestamps, target_hz=5.0)

    np.testing.assert_array_equal(indices, [0, 6, 12, 18, 24, 30])


def test_future_states_are_shape_only_placeholders():
    placeholders = b1k_contract.future_state_placeholders(np.ones((2, 23), dtype=np.float32), horizon=8)

    assert placeholders.shape == (2, 8, 23)
    assert placeholders.dtype == np.float32
    np.testing.assert_array_equal(placeholders, 0)


def test_post_transform_action_shape_is_enforced():
    contract = b1k_contract.build_r1pro_contract()

    with pytest.raises(ValueError, match="post-transform"):
        contract.validate_environment_actions(np.zeros((8, 32), dtype=np.float32))
