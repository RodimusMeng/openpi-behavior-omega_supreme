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


def test_command_state_error_subtracts_current_compact_state_from_every_command():
    commands = np.stack([np.arange(23), np.arange(23) + 2.0])
    current = np.arange(23) - 1.0

    result = audit._command_minus_current_state(commands, current)

    np.testing.assert_array_equal(result[0], np.ones(23))
    np.testing.assert_array_equal(result[1], np.full(23, 3.0))


@pytest.mark.parametrize(("count", "phase", "expected"), [(6, 0.0, 0), (6, 0.5, 2), (6, 1.0, 5)])
def test_phase_index_is_strictly_interval_local(count, phase, expected):
    assert audit._phase_index(count, phase) == expected


def test_frozen_affine_is_fit_only_on_fit_episodes():
    class Component:
        name = "all"
        indices = tuple(range(23))

    class Contract:
        action_components = (Component(),)
        action_dim = 23

    rng = np.random.default_rng(7)
    phase_values = {phase: rng.normal(size=(200, 23)) for phase in audit.PHASES}
    fit = {"phase_values": phase_values, "targets": 2 * phase_values[0.5] + 3}
    fitted = audit._fit_component_phase_affine(fit, contract=Contract())

    assert fitted["all"]["phase"] == pytest.approx(0.5)
    assert fitted["all"]["dimensions"][0]["scale"] == pytest.approx(2.0)
    assert fitted["all"]["dimensions"][0]["bias"] == pytest.approx(3.0)


def test_frozen_prediction_report_does_not_refit_held_out_predictions():
    class Component:
        name = "all"
        indices = (0,)

    class Contract:
        action_components = (Component(),)

    target = np.arange(10, dtype=np.float64)[:, None]
    prediction = (2 * target + 3)

    report = audit._frozen_prediction_report(prediction, target, contract=Contract())

    assert report["components"]["all"]["r2"] < 0


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
