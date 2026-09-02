from __future__ import annotations

# ruff: noqa: SLF001 -- script parsing and transaction helpers are the unit under test.
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


def _load_script():
    module_name = "_train_weaver_b1k_test_target"
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name("train_weaver_b1k.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load train_weaver_b1k.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


train_script = _load_script()


def _released_config() -> dict:
    return {
        "defaults": {
            "horizon": 8,
            "n_history": 2,
            "n_memory_frames": 6,
            "t_memory": 5,
            "use_precomputed_features": True,
            "use_compile": False,
            "use_activation_checkpointing": False,
            "ckpt_freq": 100,
            "dataset": {
                "name": "DROID",
                "path": "/droid",
                "n_workers": 0,
                "n_states": 8,
                "n_actions": 8,
                "normalize": True,
                "img_keys": ["wrist_left", "exterior_1_left"],
                "image_size": [192, 320],
            },
            "model": {"n_embed": 64},
            "im_encoder": {
                "name": "stabilityai/stable-diffusion-3-medium-diffusers",
                "spatial_size": 2,
            },
            "inference": {"pyramid_stagger_width": 1},
            "training": {
                "batch_size": 32,
                "max_steps": 100,
                "max_lr": 1e-4,
                "min_lr": 1e-5,
                "warmup_steps": 10,
                "weight_decay": 0.01,
                "betas": [0.9, 0.99],
                "gradient_accumulation_steps": 1,
            },
            "finetune_cfg": {"batch_size": 8, "max_steps": 25, "max_lr": 2e-5},
        }
    }


def _args(tmp_path, *extra):
    checkpoint_config = tmp_path / "released.yaml"
    checkpoint_config.write_text(yaml.safe_dump(_released_config()), encoding="utf-8")
    return train_script._parse_args(
        [
            "--checkpoint-config",
            str(checkpoint_config),
            "--pretrained-dir",
            str(tmp_path / "pretrained"),
            "--task-encoder-model",
            str(tmp_path / "task-encoder"),
            "--output-dir",
            str(tmp_path / "output"),
            "--dataset-dir",
            str(tmp_path / "dataset"),
            *extra,
        ]
    )


def test_full_is_first_class_default_and_override_order_is_checkpoint_finetune_cli(tmp_path):
    args = _args(
        tmp_path,
        "--override",
        "training.batch_size=4",
        "--override",
        "training.batch_size=2",
        "--override",
        "model.n_embed=128",
    )

    config, report = train_script.load_training_config(args)

    assert args.mode == "full"
    assert config["training"]["batch_size"] == 2
    assert config["training"]["max_steps"] == 25
    assert config["model"]["n_embed"] == 128
    assert config["dataset"]["reward_mode"] == "required"
    assert config["training"]["dynamics_only"] is False
    assert report.changed_fields == ("n_states", "n_actions")


def test_dynamics_mode_changes_only_adapter_supervision_fields(tmp_path):
    args = _args(tmp_path, "--mode", "dynamics")

    config, _ = train_script.load_training_config(args)

    assert config["dataset"]["reward_mode"] == "disabled"
    assert config["training"]["dynamics_only"] is True
    assert config["model"] == _released_config()["defaults"]["model"]
    assert config["inference"] == _released_config()["defaults"]["inference"]


def test_mode_supervision_cannot_be_silently_overridden(tmp_path):
    args = _args(tmp_path, "--mode", "full", "--override", "dataset.reward_mode=disabled")

    with pytest.raises(ValueError, match=r"requires dataset\.reward_mode"):
        train_script.load_training_config(args)


@pytest.mark.parametrize("expression", ["missing-equals", "=3", "training..batch_size=3"])
def test_invalid_override_is_rejected(expression, tmp_path):
    args = _args(tmp_path, "--override", expression)

    with pytest.raises(ValueError, match="Override"):
        train_script.load_training_config(args)


def test_atomic_json_report_replaces_complete_document(tmp_path):
    path = tmp_path / "report.json"
    train_script._atomic_write_json(path, {"step": 1})
    train_script._atomic_write_json(path, {"step": 2, "complete": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 2, "complete": True}
    assert not list(tmp_path.glob(".report.json.tmp-*"))


def _step_report(step: int, loss: float):
    return train_script.training_runtime.StepReport(
        completed_step=step,
        loss=loss,
        learning_rate=1e-5,
        grad_norm=0.5,
        losses={"flow/Total Loss": loss},
    )


def test_training_metrics_are_atomic_and_record_tiny_overfit_slots(tmp_path):
    path = tmp_path / "training_metrics.jsonl"

    metadata = train_script._write_training_metrics(
        path,
        reports=(_step_report(1, 1.0), _step_report(2, 0.8), _step_report(3, 0.6)),
        start_step=0,
        tiny_overfit_batches=2,
    )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["completed_step"] for record in records] == [1, 2, 3]
    assert [record["batch_slot"] for record in records] == [0, 1, 0]
    assert [record["run_step"] for record in records] == [1, 2, 3]
    assert metadata == {
        "path": str(path),
        "records_appended": 3,
        "records_total": 3,
        "first_recorded_step": 1,
        "last_recorded_step": 3,
        "history_complete": True,
    }
    assert not list(tmp_path.glob(".training_metrics.jsonl.tmp-*"))


def test_training_metrics_resume_appends_and_marks_new_run_slots(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    train_script._write_training_metrics(
        path,
        reports=(_step_report(1, 1.0),),
        start_step=0,
        tiny_overfit_batches=4,
    )

    metadata = train_script._write_training_metrics(
        path,
        reports=(_step_report(2, 0.9), _step_report(3, 0.7)),
        start_step=1,
        tiny_overfit_batches=4,
    )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["completed_step"] for record in records] == [1, 2, 3]
    assert [record["run_start_step"] for record in records] == [0, 1, 1]
    assert [record["batch_slot"] for record in records] == [0, 0, 1]
    assert metadata["records_total"] == 3
    assert metadata["history_complete"] is True


def test_training_metrics_reject_checkpoint_history_mismatch(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    train_script._write_training_metrics(
        path,
        reports=(_step_report(1, 1.0),),
        start_step=0,
        tiny_overfit_batches=1,
    )

    with pytest.raises(ValueError, match="metrics end at step 1"):
        train_script._write_training_metrics(
            path,
            reports=(_step_report(3, 0.5),),
            start_step=2,
            tiny_overfit_batches=1,
        )


def test_training_lineage_records_exact_episode_semantics():
    class Dataset:
        manifests = (
            {"episode_index": 3, "semantic_sha256": "a" * 64},
            {"episode_index": 8, "semantic_sha256": "b" * 64},
        )

    lineage = train_script._dataset_lineage(Dataset(), split="train")

    assert lineage == {
        "split": "train",
        "episode_count": 2,
        "episode_indices": [3, 8],
        "semantic_sha256": ["a" * 64, "b" * 64],
    }


def _resume_config() -> dict:
    return {
        "dataset": {
            "path": "/new/dataset",
            "norm_stats_path": "/new/dataset/norm_stats.json",
            "n_states": 23,
            "n_actions": 23,
        },
        "training": {"max_lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.99]},
        "horizon": 8,
    }


def _lineage() -> dict:
    return {"split": "train", "episode_count": 1, "episode_indices": [1], "semantic_sha256": ["a" * 64]}


def _norm_provenance() -> dict:
    return {
        "path": "/new/dataset/norm_stats.json",
        "sha256": "b" * 64,
        "schema_version": "b1k_weaver_norm_v1",
        "source_split": "train",
        "target_hz": 5.0,
        "train_manifest_sha256": "c" * 64,
        "episode_count": 1,
        "dataset": {"repo_id": "b1k-v3again", "metadata_sha256": "d" * 64},
    }


def test_resume_contract_accepts_relocated_data_paths(tmp_path: Path) -> None:
    saved = {**_resume_config(), "weaver_lineage": _lineage(), "weaver_norm_provenance": _norm_provenance()}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(saved), encoding="utf-8")
    train_script._assert_resume_contract(
        tmp_path, config=_resume_config(), lineage=_lineage(), norm_provenance=_norm_provenance()
    )


@pytest.mark.parametrize("field", ["max_lr", "norm_stats", "lineage"])
def test_resume_contract_rejects_semantic_changes(tmp_path: Path, field: str) -> None:
    saved = {**_resume_config(), "weaver_lineage": _lineage(), "weaver_norm_provenance": _norm_provenance()}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(saved), encoding="utf-8")
    current_config = _resume_config()
    current_lineage = _lineage()
    current_norm = _norm_provenance()
    if field == "max_lr":
        current_config["training"]["max_lr"] = 2e-4
    elif field == "norm_stats":
        current_norm["sha256"] = "e" * 64
    else:
        current_lineage["episode_indices"] = [2]
    with pytest.raises(ValueError, match="Resume"):
        train_script._assert_resume_contract(
            tmp_path,
            config=current_config,
            lineage=current_lineage,
            norm_provenance=current_norm,
        )


def test_resume_contract_requires_saved_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing its saved config"):
        train_script._assert_resume_contract(
            tmp_path, config=_resume_config(), lineage=_lineage(), norm_provenance=_norm_provenance()
        )
