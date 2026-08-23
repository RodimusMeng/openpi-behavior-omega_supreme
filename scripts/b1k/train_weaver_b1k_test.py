from __future__ import annotations

# ruff: noqa: SLF001 -- script parsing and transaction helpers are the unit under test.

import importlib.util
import json
import sys
from pathlib import Path

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

    with pytest.raises(ValueError, match="requires dataset.reward_mode"):
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
