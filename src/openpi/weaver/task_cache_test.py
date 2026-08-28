from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.weaver import task_cache


def _catalog() -> dict:
    return {
        "schema_version": task_cache.TASK_PROMPT_SCHEMA_VERSION,
        "dataset_metadata_sha256": "a" * 64,
        "task_catalog_sha256": "b" * 64,
        "prompts": {"turning_on_radio": "Turn on the radio.", "opening_drawer": "Open the drawer."},
    }


def _encoder() -> task_cache.TaskEncoderSpec:
    return task_cache.TaskEncoderSpec(
        model_id="openai/clip-vit-large-patch14",
        revision="c" * 40,
        feature_dim=768,
        snapshot_sha256="d" * 64,
        upstream_git_sha="3e41e907",
    )


def test_task_feature_artifact_round_trip_is_catalog_named_and_float32(tmp_path):
    catalog = _catalog()
    semantics = task_cache.task_feature_semantics(catalog, encoder=_encoder(), adapter_git_sha="deadbeef")
    features = {
        name: np.full(768, index, dtype=np.float64)
        for index, name in enumerate(sorted(catalog["prompts"]), start=1)
    }

    task_cache.write_task_feature_artifact(tmp_path, semantics=semantics, features=features)
    loaded = task_cache.load_task_feature_artifact(tmp_path, prompt_catalog=catalog)

    assert tuple(loaded.by_task_name) == tuple(sorted(catalog["prompts"]))
    assert loaded.by_prompt["Turn on the radio."].shape == (768,)
    assert loaded.by_prompt["Turn on the radio."].dtype == np.float32
    assert not loaded.by_prompt["Turn on the radio."].flags.writeable
    assert loaded.manifest["semantic_sha256"] == semantics["semantic_sha256"]


def test_task_feature_artifact_rejects_changed_prompt_catalog(tmp_path):
    catalog = _catalog()
    semantics = task_cache.task_feature_semantics(catalog, encoder=_encoder(), adapter_git_sha="deadbeef")
    features = {name: np.ones(768, dtype=np.float32) for name in catalog["prompts"]}
    task_cache.write_task_feature_artifact(tmp_path, semantics=semantics, features=features)
    changed = {**catalog, "prompts": {**catalog["prompts"], "turning_on_radio": "Switch on the radio."}}

    with pytest.raises(task_cache.TaskCacheMissError, match="different prompt catalog"):
        task_cache.load_task_feature_artifact(tmp_path, prompt_catalog=changed)


def test_task_feature_artifact_rejects_tampered_bytes(tmp_path):
    catalog = _catalog()
    semantics = task_cache.task_feature_semantics(catalog, encoder=_encoder(), adapter_git_sha="deadbeef")
    features = {name: np.ones(768, dtype=np.float32) for name in catalog["prompts"]}
    task_cache.write_task_feature_artifact(tmp_path, semantics=semantics, features=features)
    path = tmp_path / "features.npz"
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(task_cache.TaskCacheMissError, match="bytes do not match"):
        task_cache.load_task_feature_artifact(tmp_path, prompt_catalog=catalog)


def test_read_task_prompt_catalog_normalizes_order(tmp_path):
    path = tmp_path / "task_prompts.json"
    path.write_text(json.dumps(_catalog()), encoding="utf-8")

    value = task_cache.read_task_prompt_catalog(path)

    assert tuple(value["prompts"]) == ("opening_drawer", "turning_on_radio")
