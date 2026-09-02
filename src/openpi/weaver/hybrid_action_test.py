from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from openpi.weaver import causal_action_adapter
from openpi.weaver import hybrid_action
from openpi.weaver.b1k_contract import build_r1pro_contract


def _contract() -> hybrid_action.HybridActionContract:
    components = []
    for component in build_r1pro_contract().action_components:
        if component.name in hybrid_action.GRIPPER_COMPONENTS:
            continue
        components.append(
            hybrid_action.ComponentAffine(
                name=component.name,
                indices=tuple(component.indices),
                phase=0.75,
                scale=np.ones(len(component.indices)),
                bias=np.zeros(len(component.indices)),
            )
        )
    return hybrid_action.HybridActionContract(tuple(components), 0.75, "0" * 64, "/audit.json")


def test_hybrid_interval_uses_affine_delta_and_binary_gripper_command():
    state = np.arange(23, dtype=np.float32)
    commands = np.stack([state + offset for offset in (1, 2, 3, 4, 5)], axis=0)
    left, right = hybrid_action.gripper_indices()
    commands[:, left] = -1
    commands[:, right] = 1

    result = hybrid_action.interval_action(_contract(), state, commands)

    expected_index = hybrid_action.phase_index(len(commands), 0.75)
    non_gripper = sorted(set(range(23)) - {left, right})
    np.testing.assert_allclose(result[non_gripper], commands[expected_index, non_gripper] - state[non_gripper])
    assert result[left] == -1
    assert result[right] == 1


def test_offline_and_online_materialization_share_exact_interval_contract():
    state = np.zeros(23, dtype=np.float32)
    commands = np.ones((6, 23), dtype=np.float32)
    left, right = hybrid_action.gripper_indices()
    commands[:, left] = -1
    commands[:, right] = 1
    numeric = SimpleNamespace(
        length=2,
        states=np.stack([state, state]),
        transition_valid=np.array([True, False]),
        source_frame_indices=np.array([0, 6]),
        source_policy_actions=commands,
    )
    offline = hybrid_action.materialize_episode_actions(numeric, _contract())[0]
    segmented = causal_action_adapter.segment_candidate_actions(
        commands[None], np.arange(7, dtype=np.float64) / 30.0, target_hz=5.0
    )
    online = hybrid_action.predict_candidate_actions(_contract(), state, segmented)[0, 0]

    np.testing.assert_array_equal(online, offline)


def test_offline_interval_does_not_read_command_at_next_boundary():
    state = np.zeros(23, dtype=np.float32)
    commands = np.ones((7, 23), dtype=np.float32)
    left, right = hybrid_action.gripper_indices()
    commands[:, left] = -1
    commands[:, right] = 1
    commands[6] = 999
    numeric = SimpleNamespace(
        length=2,
        states=np.stack([state, state]),
        transition_valid=np.array([True, False]),
        source_frame_indices=np.array([0, 6]),
        source_policy_actions=commands,
    )

    result = hybrid_action.materialize_episode_actions(numeric, _contract())[0]

    assert np.max(np.abs(result)) < 10
