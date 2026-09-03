from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from openpi.training import config as training_config
from openpi.weaver import hybrid_action
from openpi.weaver.causal_action_adapter import require_world_model_horizon
from scripts.b1k.serve_weaver_b1k import Args
from scripts.b1k.serve_weaver_b1k import _canonical_prompt_catalog_sha256
from scripts.b1k.serve_weaver_b1k import _load_norm
from scripts.b1k.serve_weaver_b1k import _load_task_prompt_catalog
from scripts.b1k.serve_weaver_b1k import _require_checkpoint_policy_contract
from scripts.b1k.serve_weaver_b1k import _resize_batch
from scripts.b1k.serve_weaver_b1k import _resolve_task


def test_server_requires_pinned_task_encoder_model() -> None:
    field = next(field for field in dataclasses.fields(Args) if field.name == "task_encoder_model")
    assert field.default is dataclasses.MISSING


def test_server_defaults_to_the_training_latent_resize_contract() -> None:
    field = next(field for field in dataclasses.fields(Args) if field.name == "resize_mode")
    assert field.default == "stretch"


def test_broad_absolute_checkpoint_config_keeps_controller_actions_absolute() -> None:
    cfg = training_config.get_config("pi05_b1k_full_abs")
    assert cfg.model.action_horizon == 32
    assert cfg.model.action_dim == 32
    assert cfg.data.robot_config_name == "b1k/R1Pro"
    assert cfg.data.repo_id == "b1k-v3again"
    assert cfg.data.assets.asset_id is None
    assert cfg.data.extra_delta_transform is False
    assert _require_checkpoint_policy_contract(cfg, robot="b1k/R1Pro") is cfg
    with pytest.raises(ValueError, match="training data contract"):
        _require_checkpoint_policy_contract(cfg, robot="b1k/Other")


def test_resize_letterbox_matches_target_shape_and_range() -> None:
    source = torch.ones(1, 3, 4, 8)
    result = _resize_batch(source, (8, 8), "letterbox")
    assert result.shape == (1, 3, 8, 8)
    assert torch.all(result[:, :, 0, :] == 0)


def test_resize_center_crop_matches_target_shape() -> None:
    source = torch.zeros(1, 3, 8, 4)
    assert _resize_batch(source, (4, 4), "center-crop").shape == (1, 3, 4, 4)


def test_load_norm_requires_exact_train_representations() -> None:
    payload = {
        "source_split": "train",
        "norm_stats": {
            "state": {"representation": "r1pro_compact_state_v1", "mean": [0.0] * 23, "std": [1.0] * 23},
            "actions": {
                "representation": hybrid_action.HYBRID_REPRESENTATION,
                "mean": [0.0] * 23,
                "std": [1.0] * 23,
            },
        },
        "hybrid_action_contract": {"source_audit_sha256": "a" * 64},
    }
    contract = SimpleNamespace(source_audit_sha256="a" * 64)
    with mock.patch("pathlib.Path.read_text", return_value=json.dumps(payload)):
        values = _load_norm("norm_stats.json", expected_hybrid_contract=contract)
    assert all(value.shape == (23,) for value in values)
    payload["source_split"] = "val"
    with (
        mock.patch("pathlib.Path.read_text", return_value=json.dumps(payload)),
        pytest.raises(ValueError, match="train-split"),
    ):
        _load_norm("norm_stats.json", expected_hybrid_contract=contract)


def test_horizon_contract_rejects_short_policy_chunks() -> None:
    assert require_world_model_horizon(32, 5) == 5
    with pytest.raises(ValueError, match="cannot cover"):
        require_world_model_horizon(32, 6)


def test_resolve_task_accepts_bare_name_and_explicit_prompt() -> None:
    registry = {"b1k": {"demo": "Do the demo."}}
    assert _resolve_task("demo", None, registry) == ("b1k", "demo", "Do the demo.")
    assert _resolve_task("b1k/unknown", "Do the unknown.", registry) == (
        "b1k",
        "unknown",
        "Do the unknown.",
    )
    with pytest.raises(ValueError, match="empty"):
        _resolve_task("", None, registry)


def test_official_prompt_catalog_is_authoritative_and_hash_checked(tmp_path) -> None:
    dataset_hash = "a" * 64
    prompt_hash = _canonical_prompt_catalog_sha256({"official_task": "Place the object."})
    (tmp_path / "task_prompts.json").write_text(
        json.dumps(
            {
                "schema_version": "behavior_task_prompts_v1",
                "dataset_metadata_sha256": dataset_hash,
                "task_catalog_sha256": prompt_hash,
                "prompts": {"official_task": "Place the object."},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"dataset_metadata_sha256": dataset_hash, "task_catalog": {"sha256": prompt_hash}}),
        encoding="utf-8",
    )
    catalog = _load_task_prompt_catalog(tmp_path)
    assert _resolve_task("b1k/official_task", None, {"b1k": {}}, catalog) == (
        "b1k",
        "official_task",
        "Place the object.",
    )
    (tmp_path / "summary.json").write_text(json.dumps({"dataset_metadata_sha256": "b" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _load_task_prompt_catalog(tmp_path)
