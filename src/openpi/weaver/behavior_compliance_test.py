from __future__ import annotations

import numpy as np
import pytest

from openpi.weaver import behavior_compliance


def test_official_flattened_rgbd_and_proprio_keys_are_allowed() -> None:
    keys = (
        "robot_r1::robot_r1:zed_link:Camera:0::rgb",
        "robot_r1::robot_r1:zed_link:Camera:0::depth_linear",
        "robot_r1::proprio",
    )
    assert behavior_compliance.validate_observation_keys(keys) == ()


def test_privileged_or_unknown_observation_fails_closed() -> None:
    violations = behavior_compliance.validate_observation_keys(
        ("robot_r1::segmentation", "task::object_state", "robot_r1::mystery")
    )
    assert {item.code for item in violations} == {"forbidden_observation", "unknown_observation"}


def test_action_contract_requires_one_finite_controller_action() -> None:
    assert behavior_compliance.validate_action(np.zeros(23, dtype=np.float32), action_dim=23) == ()
    assert {item.code for item in behavior_compliance.validate_action(np.zeros((1, 23)), action_dim=23)} == {
        "action_shape"
    }
    bad = np.zeros(23)
    bad[0] = np.nan
    assert {item.code for item in behavior_compliance.validate_action(bad, action_dim=23)} == {"action_finite"}


def test_submission_manifest_rejects_oracle() -> None:
    common = {
        "behavior_git_sha": "b" * 40,
        "omega_git_sha": "a" * 40,
        "weaver_git_sha": "c" * 40,
        "wrapper": {"path": "omnigibson.eval.wrappers.RGBDFullResWrapper", "sha256": "1" * 64},
        "robot_config": {"path": "r1pro.yaml", "sha256": "2" * 64},
        "observation_keys": ("robot_r1::proprio",),
        "action_dim": 23,
        "task_ids": ("task",),
        "eval_mode": "public_test",
        "instance_indices": tuple(range(10)),
        "num_rollouts": 1,
    }
    with pytest.raises(ValueError, match="oracle"):
        behavior_compliance.build_manifest(**common, oracle_used=True)
    manifest = behavior_compliance.build_manifest(**common, privileged_offline_labels=True)
    assert manifest["selector"] == {"oracle_used": False, "online_inputs_privileged": False}
    assert manifest["training_labels"]["privileged_offline"] is True
    assert behavior_compliance.validate_manifest(manifest, submission=True) == ()


def test_evaluator_metadata_is_not_sensor_input() -> None:
    assert behavior_compliance.validate_observation_keys(
        ("task_id", "cam_rel_poses", "robot_r1::proprio"), include_evaluator_metadata=True
    ) == ()
