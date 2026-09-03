from __future__ import annotations

from contextlib import AbstractContextManager
import json
from typing import Any

import numpy as np
import pytest

from openpi.weaver import prediction_eval


class _FakeBehaviorPrecomputed:
    split = "val"
    horizon = 3
    n_history = 2
    manifests = (
        {"episode_index": 11, "sampled_count": 6, "semantic_sha256": "a" * 64},
        {"episode_index": 22, "sampled_count": 5, "semantic_sha256": "b" * 64},
    )

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, bool]] = []

    def window_at(self, index: int, t: int, *, collapse: bool) -> dict[str, Any]:
        self.calls.append((index, t, collapse))
        total = self.n_history + self.horizon
        return {
            "obs": {
                "states": np.zeros((total, 23), dtype=np.float32),
                "camera_features": np.zeros((total, 1, 4), dtype=np.float32),
            },
            "actions": np.zeros((total, 23), dtype=np.float32),
            "transition_valid": np.array([True] * (total - 1) + [False]),
            "task": {"text": "test"},
        }


def test_held_out_windows_are_deterministic_and_episode_local() -> None:
    base = _FakeBehaviorPrecomputed()
    dataset = prediction_eval.HeldOutWindowDataset(base, max_horizon=3)

    assert [(ref.episode_index, ref.anchor_t) for ref in dataset.refs] == [(11, 1), (11, 2), (22, 1)]
    item = dataset[1]
    assert base.calls == [(0, 2, False)]
    assert item["eval_provenance"] == {
        "window_index": 1,
        "split": "val",
        "episode_index": 11,
        "anchor_t": 2,
        "first_target_t": 3,
        "last_target_t": 5,
        "manifest_semantic_sha256": "a" * 64,
    }


def test_held_out_windows_reject_train_split() -> None:
    base = _FakeBehaviorPrecomputed()
    base.split = "train"
    with pytest.raises(ValueError, match="held-out"):
        prediction_eval.HeldOutWindowDataset(base, max_horizon=3)


def test_target_transition_mask_starts_at_last_history_action() -> None:
    value = np.array([[False, True, False, True], [True, False, True, False]], dtype=np.bool_)
    result = prediction_eval.target_transition_mask(value, n_history=2, horizon=2)
    np.testing.assert_array_equal(result, [[True, False], [False, True]])


class _EmaContext(AbstractContextManager[None]):
    def __init__(self, owner: _FakeEma) -> None:
        self.owner = owner

    def __enter__(self) -> None:
        self.owner.active = True
        self.owner.entries += 1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.owner.active = False


class _FakeEma:
    def __init__(self) -> None:
        self.active = False
        self.entries = 0

    def use_ema_weights(self) -> _EmaContext:
        return _EmaContext(self)


class _NoGrad(AbstractContextManager[None]):
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None


class _FakeTorch:
    @staticmethod
    def as_tensor(value: Any, *, device: str) -> np.ndarray:
        del device
        return np.asarray(value)

    @staticmethod
    def no_grad() -> _NoGrad:
        return _NoGrad()


class _FakeModel:
    _n_history = 2

    def __init__(self) -> None:
        self.ema = _FakeEma()
        self.evaluating = False
        self.memory_seen: Any = None
        self.expect_ema = True

    def eval(self) -> _FakeModel:
        self.evaluating = True
        return self

    def encode_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {"camera": obs["camera_features"].copy(), "states": obs["states"].copy()}

    def generate_latent_rollouts(
        self,
        x1: dict[str, np.ndarray],
        actions: np.ndarray,
        *,
        memory: Any,
    ) -> dict[str, np.ndarray]:
        assert self.ema.active is self.expect_ema
        assert np.count_nonzero(x1["camera"][:, 2:]) == 0
        assert np.count_nonzero(x1["states"][:, 2:]) == 0
        assert actions.shape == (1, 4, 23)
        self.memory_seen = memory
        result = {key: value.copy() for key, value in x1.items()}
        result["states"][:, 2] = 1.0
        result["states"][:, 3] = 2.0
        result["camera"][:, 2:] = 0.25
        return result

    def decode_obs(self, value: dict[str, np.ndarray], *, chunk_size: int) -> dict[str, np.ndarray]:
        assert not self.ema.active
        assert chunk_size == 7
        scalar = value["camera"][:, :, 0, 0, None, None, None]
        camera = np.broadcast_to(scalar, (*value["camera"].shape[:2], 3, 2, 2)).copy()
        return {"camera": camera, "states": value["states"]}


def test_upstream_predictor_zeros_targets_uses_ema_and_denormalizes() -> None:
    model = _FakeModel()
    predictor = prediction_eval.UpstreamWeaverPredictor(
        model,
        torch=_FakeTorch(),
        device="cpu",
        n_history=2,
        state_mean=np.full(23, 10.0, dtype=np.float32),
        state_std=np.full(23, 2.0, dtype=np.float32),
        use_ema=True,
        decode_rgb=True,
        decode_chunk_size=7,
    )
    states = np.stack([np.full(23, value, dtype=np.float32) for value in (1, 2, 3, 4)])[None]
    batch = {
        "obs": {
            "states": states,
            "camera_features": np.full((1, 4, 1, 1), 0.5, dtype=np.float32),
        },
        "actions": np.zeros((1, 4, 23), dtype=np.float32),
        "transition_valid": np.array([[True, True, False, False]], dtype=np.bool_),
        "memory": {
            "states": np.zeros((1, 1, 23), dtype=np.float32),
            "camera_features": np.zeros((1, 1, 1, 1), dtype=np.float32),
        },
    }

    result = predictor.predict(batch, 2)

    assert model.evaluating
    assert model.ema.entries == 1
    assert model.memory_seen is not None
    np.testing.assert_allclose(result.predicted_states[0, :, 0], [12.0, 14.0])
    np.testing.assert_allclose(result.target_states[0, :, 0], [16.0, 18.0])
    np.testing.assert_allclose(result.persistence_states[0, :, 0], [14.0, 14.0])
    np.testing.assert_array_equal(result.transition_valid, [[True, False]])
    assert result.decoded_predicted is not None
    assert result.decoded_predicted["camera"].shape == (1, 2, 3, 2, 2)


def test_upstream_predictor_can_use_online_non_ema_weights() -> None:
    model = _FakeModel()
    model.expect_ema = False
    predictor = prediction_eval.UpstreamWeaverPredictor(
        model,
        torch=_FakeTorch(),
        device="cpu",
        n_history=2,
        state_mean=np.zeros(23, dtype=np.float32),
        state_std=np.ones(23, dtype=np.float32),
        use_ema=False,
    )
    batch = {
        "obs": {
            "states": np.zeros((1, 4, 23), dtype=np.float32),
            "camera_features": np.zeros((1, 4, 1, 1), dtype=np.float32),
        },
        "actions": np.zeros((1, 4, 23), dtype=np.float32),
        "transition_valid": np.ones((1, 4), dtype=np.bool_),
        "memory": None,
    }

    predictor.predict(batch, 2)

    assert model.ema.entries == 0


class _StaticPredictor:
    def predict(self, batch: dict[str, Any], horizon: int) -> prediction_eval.PredictionBatch:
        del batch
        assert horizon == 2
        target = np.zeros((2, 2, 23), dtype=np.float64)
        predicted = np.empty_like(target)
        predicted[0, 0] = 1.0
        predicted[0, 1] = 2.0
        predicted[1, 0] = 3.0
        predicted[1, 1] = 4.0
        persistence = np.full_like(target, 5.0)
        pred_rgb = np.empty((2, 2, 3, 1, 1), dtype=np.float64)
        pred_rgb[0, 0] = 0.1
        pred_rgb[0, 1] = 0.2
        pred_rgb[1, 0] = 0.3
        pred_rgb[1, 1] = 0.4
        pred_latent = np.empty((2, 2, 2), dtype=np.float64)
        pred_latent[0, 0] = 1.0
        pred_latent[0, 1] = 2.0
        pred_latent[1, 0] = 3.0
        pred_latent[1, 1] = 4.0
        return prediction_eval.PredictionBatch(
            predicted_states=predicted,
            target_states=target,
            persistence_states=persistence,
            transition_valid=np.array([[True, True], [True, False]], dtype=np.bool_),
            predicted_latents={"camera": pred_latent},
            target_latents={"camera": np.zeros_like(pred_latent)},
            decoded_predicted={"camera": pred_rgb},
            decoded_target={"camera": np.zeros_like(pred_rgb)},
        )


def test_metric_summary_uses_real_mask_and_persistence_baseline() -> None:
    batch = {
        "eval_provenance": {
            "window_index": np.array([0, 1]),
            "split": ["val", "val"],
            "episode_index": np.array([11, 22]),
            "anchor_t": np.array([1, 3]),
        }
    }
    records: list[dict[str, Any]] = []

    def lpips(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.abs(predicted - target).reshape(2, 2, -1).mean(axis=2)

    summary = prediction_eval.evaluate_prediction_batches(
        [batch],
        _StaticPredictor(),
        horizons=(2,),
        record_sink=records.append,
        lpips_metric=lpips,
    )

    leads = summary["metrics"]["canonical_state"]["2"]["by_lead"]
    assert leads[0]["valid_samples"] == 2
    assert leads[0]["mae"] == pytest.approx(2.0)
    assert leads[0]["rmse"] == pytest.approx(np.sqrt(5.0))
    assert leads[1]["valid_samples"] == 1
    assert leads[1]["mae"] == pytest.approx(2.0)
    assert leads[1]["persistence_mae"] == pytest.approx(5.0)
    latent = summary["metrics"]["latent"]["2"]["camera"]["by_lead"]
    assert latent[0]["mse"] == pytest.approx(5.0)
    assert latent[1]["mse"] == pytest.approx(4.0)
    rgb = summary["metrics"]["decoded_rgb"]["2"]["camera"]["by_lead"]
    assert rgb[0]["mse"] == pytest.approx(0.05)
    assert rgb[1]["mse"] == pytest.approx(0.04)
    assert rgb[0]["lpips"] == pytest.approx(0.2)
    assert len(records) == 2
    assert records[1]["state_metrics"][1]["transition_valid"] is False
    assert records[1]["state_metrics"][1]["mae"] is None
    assert records[1]["latent_metrics"]["camera"][1]["mse"] is None
    json.dumps({"summary": summary, "records": records}, allow_nan=False)


def test_state_metrics_use_shortest_angle_error() -> None:
    angular_index = prediction_eval.ANGULAR_STATE_INDICES[0]

    class _AngularPredictor:
        def predict(self, batch: dict[str, Any], horizon: int) -> prediction_eval.PredictionBatch:
            del batch
            predicted = np.zeros((1, horizon, 23), dtype=np.float64)
            target = np.zeros_like(predicted)
            predicted[..., angular_index] = -np.pi + 0.1
            target[..., angular_index] = np.pi - 0.1
            return prediction_eval.PredictionBatch(
                predicted_states=predicted,
                target_states=target,
                persistence_states=target.copy(),
                transition_valid=np.ones((1, horizon), dtype=np.bool_),
            )

    summary = prediction_eval.evaluate_prediction_batches(
        [{"eval_provenance": {"window_index": np.array([0]), "episode_index": np.array([1])}}],
        _AngularPredictor(),
        horizons=(1,),
    )

    lead = summary["metrics"]["canonical_state"]["1"]["by_lead"][0]
    assert lead["mae"] == pytest.approx(0.2 / 23)
    assert summary["angular_state_indices"] == list(prediction_eval.ANGULAR_STATE_INDICES)
