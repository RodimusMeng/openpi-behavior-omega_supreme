from __future__ import annotations

import json
from unittest import mock

import pytest
import torch

from openpi.weaver.causal_action_adapter import require_world_model_horizon
from scripts.b1k.serve_weaver_b1k import _load_norm
from scripts.b1k.serve_weaver_b1k import _resize_batch


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
                "representation": "r1pro_compact_state_delta_v1",
                "mean": [0.0] * 23,
                "std": [1.0] * 23,
            },
        },
    }
    with mock.patch("pathlib.Path.read_text", return_value=json.dumps(payload)):
        values = _load_norm("norm_stats.json")
    assert all(value.shape == (23,) for value in values)
    payload["source_split"] = "val"
    with (
        mock.patch("pathlib.Path.read_text", return_value=json.dumps(payload)),
        pytest.raises(ValueError, match="train-split"),
    ):
        _load_norm("norm_stats.json")


def test_horizon_contract_rejects_short_policy_chunks() -> None:
    assert require_world_model_horizon(32, 5) == 5
    with pytest.raises(ValueError, match="cannot cover"):
        require_world_model_horizon(32, 6)
