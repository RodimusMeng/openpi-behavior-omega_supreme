from __future__ import annotations

# ruff: noqa: SLF001 -- private audit helpers are the units under test.
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_script():
    module_name = "_audit_weaver_action_semantics_test_target"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).with_name("audit_weaver_action_semantics.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load action-semantics audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_script()


def test_dimension_metrics_recovers_affine_policy_to_realized_mapping():
    policy = np.linspace(-2.0, 2.0, 101)
    realized = 0.25 * policy - 0.4

    result = audit._dimension_metrics(policy, realized)

    assert result["correlation"] == pytest.approx(1.0)
    assert result["affine_scale"] == pytest.approx(0.25)
    assert result["affine_offset"] == pytest.approx(-0.4)
    assert result["affine_r2"] == pytest.approx(1.0)
    assert result["standardized_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert result["direct_nrmse"] > 0
    assert result["origin_nrmse"] > 0


def test_dimension_metrics_reports_sign_inversion_after_independent_normalization():
    policy = np.linspace(-1.0, 1.0, 21)
    realized = -policy

    result = audit._dimension_metrics(policy, realized)

    assert result["correlation"] == pytest.approx(-1.0)
    assert result["standardized_rmse"] == pytest.approx(2.0)
    assert result["affine_r2"] == pytest.approx(1.0)


def test_best_lag_prefers_highest_component_r2_then_smallest_absolute_lag():
    reports = {
        "-1": {"components": {"arm": {"mean_affine_r2": 0.8}}},
        "0": {"components": {"arm": {"mean_affine_r2": 0.8}}},
        "1": {"components": {"arm": {"mean_affine_r2": 0.7}}},
    }

    lag, _ = audit._best_lag(reports, component="arm")

    assert lag == 0


def test_atomic_audit_refuses_overwrite(tmp_path):
    path = tmp_path / "audit.json"
    audit._atomic_write_json(path, {"complete": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit._atomic_write_json(path, {"complete": False})


def test_artifact_identity_changes_with_lag_contract():
    episodes = [{"episode_index": 0, "semantic_sha256": "a" * 64, "sampled_count": 10}]
    args = argparse.Namespace(split="train", min_lag_frames=-1, max_lag_frames=1)
    first = audit._artifact_identity(episodes, args=args)
    args.max_lag_frames = 2

    assert audit._artifact_identity(episodes, args=args) != first
