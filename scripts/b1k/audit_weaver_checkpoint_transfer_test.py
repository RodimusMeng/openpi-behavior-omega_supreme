"""Focused tests for the real-checkpoint audit entrypoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_script():
    if importlib.util.find_spec("torch") is None:
        sys.modules.setdefault("torch", SimpleNamespace())
    if importlib.util.find_spec("yaml") is None:
        sys.modules.setdefault("yaml", SimpleNamespace())
    module_name = "_audit_weaver_checkpoint_transfer_test_target"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).with_name("audit_weaver_checkpoint_transfer.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load audit_weaver_checkpoint_transfer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_script()


def test_make_23d_config_handles_saved_and_repo_style_configs_without_mutation():
    concrete = {
        "horizon": 8,
        "n_history": 2,
        "n_memory_frames": 6,
        "t_memory": 5,
        "dataset": {
            "name": "DROID",
            "img_keys": ["wrist_left", "exterior_1_left"],
            "image_size": [192, 320],
            "n_states": 8,
            "n_actions": 8,
        },
        "im_encoder": {"spatial_size": 2},
    }

    for source in (concrete, {"defaults": concrete}):
        target = audit._make_23d_config(source)  # noqa: SLF001

        assert target["dataset"]["n_states"] == 23
        assert target["dataset"]["n_actions"] == 23
        assert target["dataset"]["img_keys"] == ["wrist_left", "exterior_1_left"]
        assert target["use_precomputed_features"] is True

    assert concrete["dataset"]["n_states"] == 8
    assert concrete["dataset"]["n_actions"] == 8


def test_make_23d_config_materializes_omitted_released_dimensions_and_runtime_mode():
    source = {
        "horizon": 8,
        "n_history": 2,
        "n_memory_frames": 6,
        "t_memory": 5,
        "dataset": {
            "name": "DROID",
            "img_keys": ["wrist_left", "exterior_1_left"],
            "image_size": [192, 320],
        },
        "im_encoder": {"spatial_size": 2},
    }

    target = audit._make_23d_config(source)  # noqa: SLF001

    assert target["dataset"]["n_states"] == 23
    assert target["dataset"]["n_actions"] == 23
    assert target["use_precomputed_features"] is True
