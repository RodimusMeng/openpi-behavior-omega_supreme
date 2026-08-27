import copy
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import scripts.b1k as _b1k_package
from scripts.b1k import prepare_weaver_dataset_test as _serial_test

# ruff: noqa: SLF001 -- this test verifies the parallel entry point's transaction composition.


def _load_parallel_module():
    module_name = "_prepare_weaver_dataset_parallel_test_target"
    fake_tyro = types.ModuleType("tyro")

    replacements = {
        "tyro": fake_tyro,
        "scripts.b1k.prepare_weaver_dataset": _serial_test.prepare,
        "prepare_weaver_dataset": _serial_test.prepare,
    }
    missing = object()
    previous_modules = {name: sys.modules.get(name, missing) for name in replacements}
    previous_attribute = getattr(_b1k_package, "prepare_weaver_dataset", missing)
    try:
        sys.modules.update(replacements)
        _b1k_package.prepare_weaver_dataset = _serial_test.prepare
        spec = importlib.util.spec_from_file_location(
            module_name,
            Path(__file__).with_name("prepare_weaver_dataset_parallel.py"),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load prepare_weaver_dataset_parallel.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, old_value in previous_modules.items():
            if old_value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
        if previous_attribute is missing:
            delattr(_b1k_package, "prepare_weaver_dataset")
        else:
            _b1k_package.prepare_weaver_dataset = previous_attribute


parallel = _load_parallel_module()


def _coordinator_state(dataset_root: Path) -> dict[str, object]:
    return {
        "dataset_root": dataset_root,
        "dataset_repo_id": "b1k-v3again",
        "source_fps": 30.0,
        "columns": ("index", "episode_index"),
        "metadata_sha256": "a" * 64,
        "dataset_provenance": {"source_fps": 30},
        "all_episode_rows": {3: {"episode_index": 3}},
        "episode_tasks": {3: ("task",)},
        "episode_prompts": {3: "Do the task."},
        "splits": {3: "train"},
        # Runtime-only objects must never cross the spawn boundary.
        "dataset": object(),
        "contract": object(),
    }


def test_worker_initializer_contract_is_complete_and_excludes_runtime_handles(tmp_path):
    args = parallel.Args(workers=2, worker_threads=3)
    state = _coordinator_state(tmp_path / "dataset")
    shared = parallel._worker_shared_state(state, staging_root=tmp_path / "staging")
    initargs = parallel._worker_initializer_args(args, state, shared)

    assert initargs == (3, args.config_name, str(state["dataset_root"]), args, shared)
    assert "dataset" not in shared
    assert "contract" not in shared
    assert shared["staging_root"] == str(tmp_path / "staging")


def test_worker_initializer_reopens_and_checks_its_numeric_dataset(tmp_path, monkeypatch):
    args = parallel.Args(workers=1)
    state = _coordinator_state(tmp_path / "dataset")
    shared = parallel._worker_shared_state(state, staging_root=tmp_path / "staging")
    numeric_view = object()
    selected_view = types.SimpleNamespace(with_format=lambda _format: numeric_view)
    hf_dataset = types.SimpleNamespace(select_columns=lambda _columns: selected_view)
    dataset = types.SimpleNamespace(fps=30.0, hf_dataset=hf_dataset)
    data_factory = types.SimpleNamespace(repo_id="b1k-v3again")
    monkeypatch.setattr(
        parallel._serial,
        "_prepare_dataset",
        lambda _args: (types.SimpleNamespace(), data_factory, dataset, state["dataset_root"]),
    )
    monkeypatch.setattr(parallel, "_set_worker_threads", lambda _count: None)

    parallel._worker_initializer(*parallel._worker_initializer_args(args, state, shared))

    assert parallel._WORKER_STATE["dataset"] is numeric_view
    assert parallel._WORKER_STATE["source_fps"] == 30.0
    assert parallel._WORKER_STATE["report_worker_initialization"] is True
    parallel._WORKER_STATE = None


def test_parallel_argument_validation_fails_before_starting_workers(monkeypatch):
    with pytest.raises(ValueError, match="workers must be positive"):
        parallel._validate_args(parallel.Args(workers=0))
    with pytest.raises(ValueError, match="max_in_flight must be at least workers"):
        parallel._validate_args(parallel.Args(workers=4, max_in_flight=3))

    monkeypatch.setattr(parallel.os, "name", "posix")
    monkeypatch.setattr(parallel.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    parallel._validate_args(parallel.Args(workers=2, max_in_flight=4))


def _result_for_staging(staging_path: Path, manifest: dict[str, object]) -> object:
    return parallel.EpisodeResult(
        episode_index=3,
        split="train",
        source_count=2,
        sampled_count=2,
        raw_state_shape=(2, 61),
        policy_action_shape=(2, 23),
        weaver_state_shape=(2, 23),
        weaver_action_shape=(2, 23),
        max_dt_error=0.0,
        manifest=manifest,
        staging_path=str(staging_path),
        staging_size_bytes=staging_path.stat().st_size,
        staging_sha256=_serial_test.prepare._file_sha256(staging_path),
        timings={},
    )


def test_parent_commits_staging_npz_then_json_as_the_only_final_writer(tmp_path):
    staging_path = tmp_path / ".staging" / "worker-3.npz"
    arrays = {"value": np.asarray([1, 2, 3], dtype=np.int64)}
    _serial_test.prepare._atomic_write_npz(staging_path, arrays)
    manifest = {
        "episode_index": 3,
        "numeric_artifact": {
            "complete": True,
            "path": "episodes/train/000003.npz",
            "keys": sorted(arrays),
            "size_bytes": staging_path.stat().st_size,
            "sha256": _serial_test.prepare._file_sha256(staging_path),
        },
    }

    persisted, status, timings = parallel._commit_result(
        _result_for_staging(staging_path, copy.deepcopy(manifest)),
        output_root=tmp_path,
        existing_pairs={},
    )

    final_npz = tmp_path / "episodes" / "train" / "000003.npz"
    assert status == "written"
    assert persisted == manifest
    assert final_npz.is_file()
    assert final_npz.with_suffix(".json").is_file()
    assert not staging_path.exists()
    assert timings["npz_commit_seconds"] >= 0.0
    assert timings["json_write_seconds"] >= 0.0


def test_json_failure_removes_the_already_moved_npz(tmp_path, monkeypatch):
    staging_path = tmp_path / ".staging" / "worker-3.npz"
    _serial_test.prepare._atomic_write_npz(staging_path, {"value": np.asarray([1], dtype=np.int64)})
    manifest = {
        "episode_index": 3,
        "numeric_artifact": {
            "complete": True,
            "path": "episodes/train/000003.npz",
            "keys": ["value"],
        },
    }
    monkeypatch.setattr(
        parallel._serial,
        "_atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated JSON failure")),
    )

    with pytest.raises(OSError, match="simulated JSON failure"):
        parallel._commit_result(
            _result_for_staging(staging_path, manifest),
            output_root=tmp_path,
            existing_pairs={},
        )

    assert not (tmp_path / "episodes" / "train" / "000003.npz").exists()
