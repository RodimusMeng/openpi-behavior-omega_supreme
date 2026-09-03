from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from openpi.weaver import b1k_contract
from scripts.b1k import replay_weaver_pipeline


def _manifest(episode_index: int, task_name: str) -> dict:
    return {
        "episode_index": episode_index,
        "task_names": [task_name],
        "prompt": f"Do {task_name}.",
    }


def test_task_selection_is_deterministic_and_excludes_radio(monkeypatch) -> None:
    monkeypatch.setattr(replay_weaver_pipeline.b1k_dataset_adapter, "verify_episode_manifest", lambda manifest: None)
    manifests = [
        _manifest(0, "turn_on_radio"),
        _manifest(1, "polishing_shoes"),
        _manifest(2, "polishing_shoes"),
        _manifest(3, "folding_towels"),
        _manifest(4, "putting_away_dishes"),
    ]
    selected = replay_weaver_pipeline.select_task_manifests(
        manifests,
        tasks=(),
        num_tasks=3,
        exclude_tasks=("turn_on_radio",),
        seed=7,
    )
    repeated = replay_weaver_pipeline.select_task_manifests(
        manifests,
        tasks=(),
        num_tasks=3,
        exclude_tasks=("turn_on_radio",),
        seed=7,
    )
    assert [(name, item["episode_index"]) for name, item in selected] == [
        (name, item["episode_index"]) for name, item in repeated
    ]
    assert {name for name, _ in selected} == {"polishing_shoes", "folding_towels", "putting_away_dishes"}


def test_explicit_missing_or_excluded_tasks_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(replay_weaver_pipeline.b1k_dataset_adapter, "verify_episode_manifest", lambda manifest: None)
    manifests = [_manifest(0, "turn_on_radio"), _manifest(1, "polishing_shoes")]
    with pytest.raises(ValueError, match="excluded"):
        replay_weaver_pipeline.select_task_manifests(
            manifests,
            tasks=("turn_on_radio",),
            num_tasks=1,
            exclude_tasks=("turn_on_radio",),
            seed=0,
        )
    with pytest.raises(ValueError, match="absent"):
        replay_weaver_pipeline.select_task_manifests(
            manifests,
            tasks=("not_in_split",),
            num_tasks=1,
            exclude_tasks=("turn_on_radio",),
            seed=0,
        )


def test_dataset_row_maps_to_official_online_observation_keys() -> None:
    contract = b1k_contract.build_r1pro_contract()
    row = {
        "episode_index": np.array(4),
        "index": np.array(123),
        "observation.state": np.zeros(61, dtype=np.float32),
    }
    for ordinal, camera_key in enumerate(sorted(contract.robot.observations)):
        camera = contract.robot.observations[camera_key]
        if ordinal == 0:
            row[camera.dataset_key] = np.zeros((3, 12, 10), dtype=np.uint8)
        else:
            row[camera.dataset_key] = np.zeros((12, 10, 3), dtype=np.uint8)
    observation, summary = replay_weaver_pipeline.build_online_observation(
        row,
        contract=contract,
        expected_episode_index=4,
        expected_absolute_index=123,
    )
    assert observation["robot_r1::proprio"].shape == (61,)
    for camera in contract.robot.observations.values():
        assert observation[camera.obs_key].shape == (12, 10, 3)
        assert observation[camera.obs_key].dtype == np.uint8
    assert len(summary["image_shapes"]) == 3


def test_runtime_args_preserve_hybrid_and_device_contract() -> None:
    values = {
        field.name: ("/tmp/value" if field.default is dataclasses.MISSING else field.default)
        for field in dataclasses.fields(replay_weaver_pipeline.Args)
        if field.default_factory is dataclasses.MISSING
    }
    args = replay_weaver_pipeline.Args(**values)
    runtime = replay_weaver_pipeline.runtime_args(args, task="polishing_shoes", prompt="Polish the shoes.")
    assert runtime.task == "b1k/polishing_shoes"
    assert runtime.prompt == "Polish the shoes."
    assert runtime.resize_mode == "stretch"
    assert runtime.device == "cuda:1"
    assert runtime.hybrid_action_contract == args.hybrid_action_contract
