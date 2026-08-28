"""Contract tests for the released task-feature producer."""

# ruff: noqa: SLF001 -- the imported script module is the contract under test.

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script():
    module_name = "_cache_weaver_task_features_test_target"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).with_name("cache_weaver_task_features.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load cache_weaver_task_features.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cache_tasks = _load_script()


def _fake_encoder(snapshot: Path):
    return SimpleNamespace(
        text_encoder=SimpleNamespace(config=SimpleNamespace(_name_or_path=str(snapshot))),
        tokenizer=SimpleNamespace(name_or_path=str(snapshot)),
    )


def test_loaded_task_model_and_tokenizer_must_share_pinned_snapshot(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    other = tmp_path / "other"
    cache_tasks._assert_loaded_snapshot(_fake_encoder(snapshot), snapshot)

    with pytest.raises(ValueError, match="did not load"):
        cache_tasks._assert_loaded_snapshot(_fake_encoder(other), snapshot)


class _FakeEncoded:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_encode_catalog_preserves_name_to_prompt_mapping_and_float32():
    seen = []

    def encoder(prompts):
        seen.extend(prompts)
        return _FakeEncoded([[len(prompt), index] for index, prompt in enumerate(prompts)])

    result = cache_tasks._encode_catalog(
        encoder,
        {"z_task": "last prompt", "a_task": "first prompt"},
        batch_size=32,
    )

    assert seen == ["first prompt", "last prompt"]
    np.testing.assert_array_equal(result["a_task"], [12, 0])
    np.testing.assert_array_equal(result["z_task"], [11, 1])
    assert result["a_task"].dtype == np.float32
