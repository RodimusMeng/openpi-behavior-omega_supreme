from __future__ import annotations

# ruff: noqa: SLF001 -- private script helpers are the units under test.
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_script():
    module_name = "_evaluate_weaver_dynamics_transfer_test_target"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).with_name("evaluate_weaver_dynamics_transfer.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load evaluate_weaver_dynamics_transfer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


evaluation = _load_script()


def test_aggregate_reports_mean_delta_relative_delta_and_win_count():
    records = [
        {"fresh": {"Total Loss": 4.0}, "trained": {"Total Loss": 2.0}},
        {"fresh": {"Total Loss": 2.0}, "trained": {"Total Loss": 3.0}},
    ]

    result = evaluation._aggregate(records, metric_names=("Total Loss",))

    assert result == {
        "Total Loss": {
            "fresh_mean": 3.0,
            "trained_mean": 2.5,
            "delta": -0.5,
            "relative_delta": -1 / 6,
            "trained_better_count": 1,
            "comparison_count": 2,
        }
    }


def test_aggregate_by_batch_slot_keeps_episode_comparisons_separate():
    records = [
        {"batch_slot": 0, "fresh": {"Total Loss": 4.0}, "trained": {"Total Loss": 2.0}},
        {"batch_slot": 1, "fresh": {"Total Loss": 2.0}, "trained": {"Total Loss": 3.0}},
    ]

    result = evaluation._aggregate_by_batch_slot(records, metric_names=("Total Loss",))

    assert result["0"]["Total Loss"]["delta"] == -2.0
    assert result["0"]["Total Loss"]["trained_better_count"] == 1
    assert result["1"]["Total Loss"]["delta"] == 1.0
    assert result["1"]["Total Loss"]["trained_better_count"] == 0


def test_data_contract_accepts_relocated_norm_path_and_rejects_semantic_drift():
    class Dataset:
        def __init__(self):
            self.split = "train"
            self.manifests = ({"episode_index": 7, "semantic_sha256": "a" * 64},)
            self.norm_stats_provenance = {
                "path": "/new/norm.json",
                "sha256": "b" * 64,
                "schema_version": "b1k_weaver_norm_v1",
            }
            self.hybrid_action_provenance = {"schema_version": "hybrid-v1", "source_audit_path": "/new/audit.json"}

    config = {
        "weaver_lineage": {
            "split": "train",
            "episode_count": 1,
            "episode_indices": [7],
            "semantic_sha256": ["a" * 64],
        },
        "weaver_norm_provenance": {
            "path": "/old/norm.json",
            "sha256": "b" * 64,
            "schema_version": "b1k_weaver_norm_v1",
        },
        "weaver_hybrid_action_provenance": {
            "schema_version": "hybrid-v1",
            "source_audit_path": "/old/audit.json",
        },
    }

    assert evaluation._validate_data_contract(config, Dataset()) == "in_sample"
    config["weaver_lineage"]["semantic_sha256"] = ["d" * 64]
    with pytest.raises(ValueError, match="overlaps"):
        evaluation._validate_data_contract(config, Dataset())


def test_data_contract_accepts_disjoint_held_out_split_and_rejects_hybrid_drift():
    class Dataset:
        def __init__(self):
            self.split = "val"
            self.manifests = ({"episode_index": 21, "semantic_sha256": "c" * 64},)
            self.norm_stats_provenance = {"path": "/new/norm.json", "sha256": "b" * 64}
            self.hybrid_action_provenance = {
                "schema_version": "hybrid-v1",
                "source_audit_path": "/new/audit.json",
            }

    config = {
        "weaver_lineage": {
            "split": "train",
            "episode_count": 1,
            "episode_indices": [7],
            "semantic_sha256": ["a" * 64],
        },
        "weaver_norm_provenance": {"path": "/old/norm.json", "sha256": "b" * 64},
        "weaver_hybrid_action_provenance": {
            "schema_version": "hybrid-v1",
            "source_audit_path": "/old/audit.json",
        },
    }

    assert evaluation._validate_data_contract(config, Dataset()) == "held_out_episode"
    config["weaver_hybrid_action_provenance"]["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="hybrid action contract"):
        evaluation._validate_data_contract(config, Dataset())


def test_atomic_output_refuses_overwrite(tmp_path):
    path = tmp_path / "evaluation.json"
    evaluation._atomic_write_json(path, {"complete": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluation._atomic_write_json(path, {"complete": False})
    assert not list(tmp_path.glob(".evaluation.json.tmp-*"))


def test_parse_args_requires_two_checkpoint_roots_and_output():
    args = evaluation._parse_args(
        [
            "--config",
            "config.yaml",
            "--released-checkpoint-dir",
            "released",
            "--trained-checkpoint-dir",
            "trained",
            "--task-encoder-model",
            "clip",
            "--dataset-dir",
            "dataset",
            "--norm-stats-path",
            "norm.json",
            "--hybrid-action-contract",
            "hybrid.json",
            "--output-json",
            "result.json",
        ]
    )

    assert args.fresh_device == "cuda:0"
    assert args.trained_device == "cuda:1"
    assert args.evaluation_split == "train"
    assert args.noise_seeds == [101, 202, 303, 404, 505, 606, 707, 808]
