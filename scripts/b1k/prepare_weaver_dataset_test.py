from __future__ import annotations

import importlib.util

# ruff: noqa: SLF001 -- this test exercises the script's resume transaction helpers directly.
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from openpi.weaver import b1k_dataset_adapter as adapter


def _load_prepare_module():
    """Load the script without importing its heavyweight training-config graph."""

    module_name = "_prepare_weaver_dataset_test_target"
    fake_tyro = types.ModuleType("tyro")
    fake_training = types.ModuleType("openpi.training")
    fake_training.__path__ = []
    fake_config = types.ModuleType("openpi.training.config")
    fake_training.config = fake_config
    replacements = {
        "tyro": fake_tyro,
        "openpi.training": fake_training,
        "openpi.training.config": fake_config,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location(
            module_name, Path(__file__).with_name("prepare_weaver_dataset.py")
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load prepare_weaver_dataset.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, old_value in previous.items():
            if old_value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


prepare = _load_prepare_module()


class _TaskTable:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        return iter(enumerate(self._rows))


def test_episode_task_index_overrides_legacy_tasks_and_uses_official_prompt_catalog():
    meta = types.SimpleNamespace(
        tasks=_TaskTable(
            [{"task_index": 7, "task_name": "official_name", "task": "Natural language prompt."}]
        )
    )
    by_index, prompts = prepare._task_catalog(meta)
    tasks = prepare._episode_tasks(
        {"episode_index": 3, "task_index": 7, "tasks": ["stale_legacy_name"]},
        task_names_by_index=by_index,
    )
    assert tasks == ("official_name",)
    assert prompts[tasks[0]] == "Natural language prompt."


def test_task_catalog_prefers_official_tasks_jsonl_and_cross_checks_loader_view(tmp_path):
    meta_root = tmp_path / "meta"
    meta_root.mkdir()
    (meta_root / "tasks.jsonl").write_text(
        '{"task_index": 7, "task_name": "official_name", "task": "Official instruction."}\n',
        encoding="utf-8",
    )
    meta = types.SimpleNamespace(
        tasks=_TaskTable([{"task_index": 7, "task_name": "official_name"}])
    )
    by_index, prompts = prepare._task_catalog(meta, tmp_path)
    assert by_index == {7: "official_name"}
    assert prompts == {"official_name": "Official instruction."}


def test_task_catalog_fails_closed_when_official_prompt_is_missing(tmp_path):
    meta_root = tmp_path / "meta"
    meta_root.mkdir()
    (meta_root / "tasks.jsonl").write_text(
        '{"task_index": 7, "task_name": "official_name"}\n',
        encoding="utf-8",
    )
    meta = types.SimpleNamespace(tasks=_TaskTable([{"task_index": 7, "task_name": "official_name"}]))
    with pytest.raises(ValueError, match="lacks task_name/task natural language"):
        prepare._task_catalog(meta, tmp_path)


def test_metadata_hash_includes_tasks_jsonl(tmp_path):
    meta_root = tmp_path / "meta"
    meta_root.mkdir()
    (meta_root / "info.json").write_text("{}", encoding="utf-8")
    (meta_root / "tasks.jsonl").write_text("{}\n", encoding="utf-8")
    first = prepare._metadata_sha256(tmp_path)
    (meta_root / "tasks.jsonl").write_text('{"changed": true}\n', encoding="utf-8")
    assert prepare._metadata_sha256(tmp_path) != first


def test_default_task_holdout_is_deterministic_and_nonempty():
    tasks = [f"task_{index:03d}" for index in range(100)]
    first = prepare._deterministic_held_out_tasks(tasks, explicit=(), fraction=0.1, seed=42)
    second = prepare._deterministic_held_out_tasks(reversed(tasks), explicit=(), fraction=0.1, seed=42)
    assert first == second
    assert len(first) == 10


def test_reward_diagnostics_preserve_span_and_terminal_provenance():
    diagnostics = prepare._reward_diagnostics(
        np.asarray([1.0, -1.0, -1.0, 0.0]),
        np.asarray([False, True, True, True]),
        np.zeros(4, dtype=bool),
        np.asarray([0, 2]),
    )
    assert diagnostics["cumulative_span"] == 2.0
    assert diagnostics["first_done_row"] == 1
    assert diagnostics["sampled_terminal_target_row"] == 2
    assert diagnostics["post_done_max_abs_reward"] == 1.0


def _artifact(root, episode_index: int = 0):
    states = np.zeros((2, 23), dtype=np.float32)
    episode = adapter.AdaptedEpisode(
        episode_index=episode_index,
        prompt="resume test",
        target_hz=5.0,
        source_frame_indices=np.asarray([0, 1], dtype=np.int64),
        timestamps_s=np.asarray([0.0, 0.2], dtype=np.float64),
        timing_errors_s=np.zeros(2, dtype=np.float64),
        states=states,
        state_delta_actions=adapter.observed_state_delta_actions(states),
        policy_actions_at_frame=np.zeros((2, 23), dtype=np.float32),
        transition_valid=np.asarray([True, False]),
        rewards=np.zeros(2, dtype=np.float32),
    )
    manifest = adapter.episode_manifest(
        episode,
        split="train",
        dataset_repo_id="b1k-v3again",
        dataset_metadata_sha256="a" * 64,
    )
    manifest["source"] = {
        "dataset_from_index": episode_index * 2,
        "dataset_to_index": episode_index * 2 + 2,
        "source_fps": 30,
        "full_rate_count": 2,
        "full_rate_columns": {
            "index": "index",
            "frame_index": "frame_index",
            "timestamp": "source_timestamps_s",
            "policy_action": "source_policy_actions",
            "next.reward": "source_rewards",
            "next.terminated": "source_terminated",
            "next.truncated": "source_truncated",
        },
        "policy_action_aggregation_5hz": "undefined",
        "raw_state_dim": 61,
    }
    manifest["task_names"] = ["resume test"]
    manifest["latent_artifact"] = {"complete": False}
    arrays = {
        "index": np.arange(episode_index * 2, episode_index * 2 + 2, dtype=np.int64),
        "frame_index": np.arange(2, dtype=np.int64),
        "source_timestamps_s": episode.timestamps_s,
        "source_policy_actions": episode.policy_actions_at_frame,
        "source_rewards": np.zeros(2, dtype=np.float32),
        "source_terminated": np.asarray([False, True]),
        "source_truncated": np.zeros(2, dtype=bool),
        "source_frame_indices": episode.source_frame_indices,
        "source_absolute_indices": np.arange(episode_index * 2, episode_index * 2 + 2, dtype=np.int64),
        "timestamps_s": episode.timestamps_s,
        "timing_errors_s": episode.timing_errors_s,
        "states": episode.states,
        "state_delta_actions": episode.state_delta_actions,
        "policy_actions_at_frame": episode.policy_actions_at_frame,
        "transition_valid": episode.transition_valid,
        "rewards": episode.rewards,
    }
    relative_npz = Path("episodes") / "train" / f"{episode_index:06d}.npz"
    npz_path = root / relative_npz
    json_path = npz_path.with_suffix(".json")
    prepare._atomic_write_npz(npz_path, arrays)
    manifest["numeric_artifact"] = {
        "complete": True,
        "path": relative_npz.as_posix(),
        "size_bytes": npz_path.stat().st_size,
        "sha256": prepare._file_sha256(npz_path),
        "keys": sorted(arrays),
    }
    prepare._atomic_write_json(json_path, manifest)
    return manifest, arrays, json_path, npz_path


def test_complete_pair_is_strictly_reused_and_stale_values_fail(tmp_path):
    manifest, arrays, json_path, npz_path = _artifact(tmp_path)

    reused = prepare._validate_existing_pair(
        tmp_path,
        json_path,
        npz_path,
        expected_manifest=manifest,
        expected_arrays=arrays,
    )
    assert reused == manifest
    assert set(prepare._discover_existing_pairs(tmp_path)) == {0}

    stale = {key: np.array(value, copy=True) for key, value in arrays.items()}
    stale["source_policy_actions"][0, 0] = 1.0
    with pytest.raises(prepare.ArtifactResumeError, match=r"source_policy_actions.*stale"):
        prepare._validate_existing_pair(
            tmp_path,
            json_path,
            npz_path,
            expected_manifest=manifest,
            expected_arrays=stale,
        )


def test_orphan_pair_fails_explicitly(tmp_path):
    orphan = tmp_path / "episodes" / "train" / "000000.npz"
    orphan.parent.mkdir(parents=True)
    np.savez_compressed(orphan, value=np.zeros(1))
    with pytest.raises(prepare.ArtifactResumeError, match="Orphan episode artifacts"):
        prepare._discover_existing_pairs(tmp_path)


def test_manifest_index_is_atomically_refreshed_after_each_episode(tmp_path):
    first, _, _, _ = _artifact(tmp_path, episode_index=0)
    second, _, _, _ = _artifact(tmp_path, episode_index=1)
    index_path = tmp_path / "manifest.jsonl"

    prepare._atomic_write_manifest_index(index_path, {0: first})
    assert [json.loads(line)["episode_index"] for line in index_path.read_text().splitlines()] == [0]

    prepare._atomic_write_manifest_index(index_path, {0: first, 1: second})
    assert [json.loads(line)["episode_index"] for line in index_path.read_text().splitlines()] == [0, 1]


def test_subset_resume_rejects_stale_unselected_pair_before_building_global_index(tmp_path):
    current, _, _, _ = _artifact(tmp_path, episode_index=0)
    stale, _, stale_json, _ = _artifact(tmp_path, episode_index=1)
    stale["target_hz"] = 4.0
    stale_semantics = {field: stale[field] for field in adapter.EPISODE_SEMANTIC_FIELDS}
    stale["semantic_sha256"] = adapter.semantic_sha256(stale_semantics)
    prepare._atomic_write_json(stale_json, stale)

    discovered = prepare._discover_existing_pairs(tmp_path)
    assert set(discovered) == {0, 1}
    context = {
        "dataset_repo_id": "b1k-v3again",
        "dataset_metadata_sha256": "a" * 64,
        "target_hz": 5.0,
        "expected_views": current["views"],
        "source_fps": 30.0,
        "all_episode_rows": {
            0: {"dataset_from_index": 0, "dataset_to_index": 2},
            1: {"dataset_from_index": 2, "dataset_to_index": 4},
        },
        "episode_tasks": {0: ("resume test",), 1: ("resume test",)},
        "episode_prompts": {0: "resume test", 1: "resume test"},
        "splits": {0: "train", 1: "train"},
        "dataset_provenance": {},
    }

    # A subset rerun may select only episode 0, but startup must still audit episode 1.
    prepare._validate_existing_context(discovered[0][0], **context)
    with pytest.raises(prepare.ArtifactResumeError, match=r"episode 1 has stale target_hz"):
        prepare._validate_existing_context(discovered[1][0], **context)


def test_authoritative_lerobot_indices_are_not_silently_derived():
    expected = np.arange(10, 13, dtype=np.int64)
    np.testing.assert_array_equal(
        prepare._authoritative_index_column(expected, name="index", expected=expected),
        expected,
    )
    with pytest.raises(ValueError, match="Authoritative LeRobot index"):
        prepare._authoritative_index_column(
            np.asarray([10, 12, 13], dtype=np.int64),
            name="index",
            expected=expected,
        )
