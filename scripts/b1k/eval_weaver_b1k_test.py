from __future__ import annotations

# ruff: noqa: SLF001 -- script parser/config helpers are the unit under test.
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


def _load_script():
    module_name = "_eval_weaver_b1k_test_target"
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name("eval_weaver_b1k.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load eval_weaver_b1k.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


eval_weaver_b1k = _load_script()


def _checkpoint_config(path: Path) -> Path:
    value = {
        "defaults": {
            "horizon": 8,
            "eval_horizon": 16,
            "n_history": 2,
            "n_memory_frames": 6,
            "t_memory": 5,
            "use_precomputed_features": True,
            "dataset": {
                "name": "DROID",
                "path": "/source",
                "img_keys": ["wrist_left", "exterior_1_left"],
                "image_size": [192, 320],
                "n_states": 8,
                "n_actions": 8,
                "normalize": True,
            },
            "im_encoder": {"name": "sd3", "spatial_size": 4},
            "model": {"val_steps": 16, "n_layers": 32, "future_upstream_knob": {"kept": True}},
            "inference": {"pyramid_schedule": "linear", "pyramid_stagger_width": 1},
            "training": {"batch_size": 8},
        }
    }
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def _args(tmp_path: Path, *extra: str):
    config = _checkpoint_config(tmp_path / "config.yaml")
    return eval_weaver_b1k._parse_args(
        [
            "--checkpoint-config",
            str(config),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--checkpoint-mode",
            "identity",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "output"),
            *extra,
        ]
    )


def test_config_preserves_upstream_knobs_and_applies_overrides(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        "--nfe",
        "7",
        "--inference-override",
        "pyramid_schedule=cosine",
        "--override",
        "model.n_layers=30",
        "--override",
        "model.val_steps=9",
        "--override",
        "inference.pyramid_stagger_width=0",
    )

    config, report = eval_weaver_b1k.load_evaluation_config(args)

    assert config["dataset"]["n_states"] == 23
    assert config["dataset"]["n_actions"] == 23
    assert config["model"]["val_steps"] == 9
    assert config["model"]["n_layers"] == 30
    assert config["model"]["future_upstream_knob"] == {"kept": True}
    assert config["inference"] == {"pyramid_schedule": "cosine", "pyramid_stagger_width": 0}
    assert report.source.n_states == 8
    assert report.target.n_states == 23


def test_horizons_support_short_and_long_values(tmp_path: Path) -> None:
    args = _args(tmp_path, "--horizon", "4", "--horizon", "32", "--horizon", "4")
    config, _ = eval_weaver_b1k.load_evaluation_config(args)
    assert eval_weaver_b1k._resolve_horizons(args.horizon, config) == (4, 32)


def test_lpips_requires_real_decoded_rgb(tmp_path: Path) -> None:
    args = _args(tmp_path, "--lpips")
    with pytest.raises(ValueError, match="requires --decode-rgb"):
        eval_weaver_b1k._validate_runtime_args(args)


def test_cli_rejects_train_split(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _args(tmp_path, "--split", "train")


def test_effective_nfe_includes_pyramid_stagger(tmp_path: Path) -> None:
    args = _args(tmp_path, "--nfe", "4", "--inference-override", "pyramid_stagger_width=2")
    config, _ = eval_weaver_b1k.load_evaluation_config(args)
    assert eval_weaver_b1k._effective_nfe(config, horizon=5) == 12


def test_lineage_rejects_episode_or_semantic_overlap(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "training_summary.json").write_text(
        '{"dataset_lineage":{"episode_indices":[7],"semantic_sha256":["' + "a" * 64 + '"]}}',
        encoding="utf-8",
    )

    class _Dataset:
        manifests = ({"episode_index": 7, "semantic_sha256": "b" * 64},)

    with pytest.raises(ValueError, match="overlaps training lineage"):
        eval_weaver_b1k._assert_checkpoint_lineage_disjoint(
            _Dataset(), checkpoint_dir=checkpoint_dir, checkpoint_mode="resume"
        )


def test_lineage_is_required_for_resumed_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    class _Dataset:
        manifests = ({"episode_index": 8, "semantic_sha256": "c" * 64},)

    with pytest.raises(ValueError, match="no dataset lineage"):
        eval_weaver_b1k._assert_checkpoint_lineage_disjoint(
            _Dataset(), checkpoint_dir=checkpoint_dir, checkpoint_mode="resume"
        )


def test_checkpoint_mode_must_be_explicit(tmp_path: Path) -> None:
    config = _checkpoint_config(tmp_path / "config.yaml")
    with pytest.raises(SystemExit):
        eval_weaver_b1k._parse_args(
            [
                "--checkpoint-config",
                str(config),
                "--checkpoint-dir",
                str(tmp_path / "checkpoint"),
                "--dataset-dir",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


def test_compliance_manifest_is_validated_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "behavior_2026_compliance_v1",
                "behavior_tag": "v3.9.1",
                "selector": {"oracle_used": False},
                "allowed_observation_keys": ["robot_r1::proprio"],
            }
        ),
        encoding="utf-8",
    )
    result = eval_weaver_b1k._load_compliance_manifest(str(path))
    assert result is not None
    assert result["sha256"] == eval_weaver_b1k._file_sha256(path)
