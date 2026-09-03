from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).with_name("compare_weaver_prediction_summaries.py")
    spec = importlib.util.spec_from_file_location("compare_weaver_prediction_summaries", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = _load_script()


def _summary(*, episode_indices=(41, 42), state_rmse=2.0, latent_rmse=3.0):
    return {
        "schema_version": "b1k_weaver_prediction_eval_v1",
        "split": "val",
        "horizons": [1],
        "window_count": 8,
        "prediction_record_count": 8,
        "evaluation": {"episode_indices": list(episode_indices)},
        "metrics": {
            "canonical_state": {
                "1": {
                    "overall": {
                        "mae": state_rmse / 2,
                        "rmse": state_rmse,
                        "persistence_rmse": 4.0,
                    }
                }
            },
            "latent": {
                "1": {
                    "camera": {
                        "overall": {"mse": latent_rmse**2, "rmse": latent_rmse},
                    }
                }
            },
        },
    }


def test_comparison_reports_fresh_to_trained_deltas() -> None:
    result = comparison.compare_summaries(
        _summary(state_rmse=2.0, latent_rmse=3.0),
        _summary(state_rmse=1.0, latent_rmse=2.0),
    )

    assert result["canonical_state"]["1"]["rmse"]["delta"] == pytest.approx(-1.0)
    assert result["latent"]["1"]["camera"]["mse"]["delta"] == pytest.approx(-5.0)
    assert result["aggregate"] == {"metrics_compared": 4, "trained_wins": 4}


def test_comparison_rejects_different_held_out_cohorts() -> None:
    with pytest.raises(ValueError, match="exact same held-out windows"):
        comparison.compare_summaries(_summary(), _summary(episode_indices=(41, 99)))
