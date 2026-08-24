"""Contract tests for the production normalization entrypoint."""

# ruff: noqa: SLF001 -- the imported script module is the contract under test.

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_script():
    module_name = "_compute_weaver_norm_stats_test_target"
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name("compute_weaver_norm_stats.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load compute_weaver_norm_stats.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


compute_norm = _load_script()


def test_production_norm_stats_rejects_partial_episode_limit(tmp_path: Path, monkeypatch) -> None:
    index_path = tmp_path / "manifest.jsonl"
    index_path.write_text('{"split":"train"}\n', encoding="utf-8")
    monkeypatch.setattr(compute_norm._episode_adapter, "verify_episode_manifest", lambda _manifest: None)

    with pytest.raises(ValueError, match="complete train split"):
        compute_norm.main(compute_norm.Args(dataset_dir=str(tmp_path), max_episodes=1))
