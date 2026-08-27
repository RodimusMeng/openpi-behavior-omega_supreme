from __future__ import annotations

# ruff: noqa: SLF001 -- parser behavior is the public CLI contract under test.
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_script():
    name = "_check_behavior_compliance_test_target"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("check_behavior_compliance.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load compliance checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_script()


def test_checker_fails_on_oracle_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "behavior_2026_compliance_v1",
                "behavior_tag": "v3.9.1",
                "selector": {"oracle_used": True},
                "allowed_observation_keys": ["robot_r1::proprio"],
                "eval_mode": "public_test",
                "instance_indices": list(range(10)),
                "num_rollouts": 1,
                "outputs_unmodified": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_info:
        checker.run(checker._parse_args([str(path), "--submission"]))
    assert exc_info.value.code == 2
