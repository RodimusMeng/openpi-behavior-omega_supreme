from __future__ import annotations

import numpy as np
import pytest

from openpi.weaver import causal_action_adapter
from openpi.weaver import steering_policy


class _Policy:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, observation):
        del observation
        self.calls += 1
        return {"actions": np.full((12, 23), self.calls, dtype=np.float32)}

    def reset(self):
        return None


class _Adapter:
    def eval(self):
        return self

    def __call__(self, current_state, policy_actions, action_durations_s, mask):
        del policy_actions, action_durations_s, mask
        return current_state * 0


def test_online_wrapper_executes_selected_original_chunk_and_reuses_pool(monkeypatch):
    del monkeypatch
    policy = _Policy()

    def fake_score(model, **kwargs):
        del model, kwargs
        return type("Score", (), {"best_index": 1})()
    norm = causal_action_adapter.AdapterNormalization(
        np.zeros(23), np.ones(23), np.zeros(23), np.ones(23), np.zeros(23), np.ones(23)
    )
    wrapper = steering_policy.WeaverSteeringPolicy(
        policy=policy,
        adapter_model=_Adapter(),
        adapter_normalization=norm,
        weaver_action_mean=np.zeros(23),
        weaver_action_std=np.ones(23),
        weaver_model=type("Model", (), {"_n_history": 1})(),
        context_fn=lambda obs: {"states": np.zeros((1, 1, 23), dtype=np.float32)},
        history_actions_fn=lambda obs: np.zeros((1, 1, 23), dtype=np.float32),
        task_fn=lambda obs: {"text": "task"},
        current_state_fn=lambda obs: np.zeros(23, dtype=np.float32),
        imagination_horizon=2,
        policy_action_horizon=12,
        num_candidates=2,
        replan_interval=3,
        score_fn=fake_score,
        device="cpu",
    )
    first = wrapper.act({"proprio": np.zeros(23, dtype=np.float32)})
    second = wrapper.act({"proprio": np.zeros(23, dtype=np.float32)})
    np.testing.assert_allclose(first, 2.0)
    np.testing.assert_allclose(second, 2.0)
    assert policy.calls == 2
    assert wrapper.telemetry.selected_indices == (1,)


def test_online_wrapper_rejects_candidate_chunk_shorter_than_imagination_horizon():
    policy = _Policy()
    norm = causal_action_adapter.AdapterNormalization(
        np.zeros(23), np.ones(23), np.zeros(23), np.ones(23), np.zeros(23), np.ones(23)
    )
    with pytest.raises(ValueError, match=r"provides 2 complete intervals.*required=3"):
        steering_policy.WeaverSteeringPolicy(
            policy=policy,
            adapter_model=_Adapter(),
            adapter_normalization=norm,
            weaver_action_mean=np.zeros(23),
            weaver_action_std=np.ones(23),
            weaver_model=type("Model", (), {"_n_history": 1})(),
            context_fn=lambda obs: {"states": np.zeros((1, 1, 23), dtype=np.float32)},
            history_actions_fn=lambda obs: np.zeros((1, 1, 23), dtype=np.float32),
            task_fn=lambda obs: {"text": "task"},
            current_state_fn=lambda obs: np.zeros(23, dtype=np.float32),
            imagination_horizon=3,
            policy_action_horizon=12,
            num_candidates=2,
            score_fn=lambda *args, **kwargs: None,
            device="cpu",
        )


def test_online_wrapper_rejects_unavailable_horizon_at_construction():
    norm = causal_action_adapter.AdapterNormalization(
        np.zeros(23), np.ones(23), np.zeros(23), np.ones(23), np.zeros(23), np.ones(23)
    )
    with pytest.raises(ValueError, match=r"provides 5 complete intervals.*required=8"):
        steering_policy.WeaverSteeringPolicy(
            policy=_Policy(),
            adapter_model=_Adapter(),
            adapter_normalization=norm,
            weaver_action_mean=np.zeros(23),
            weaver_action_std=np.ones(23),
            weaver_model=type("Model", (), {"_n_history": 1})(),
            context_fn=lambda obs: {"states": np.zeros((1, 1, 23), dtype=np.float32)},
            history_actions_fn=lambda obs: np.zeros((1, 1, 23), dtype=np.float32),
            task_fn=lambda obs: {"text": "task"},
            current_state_fn=lambda obs: np.zeros(23, dtype=np.float32),
            imagination_horizon=8,
            policy_action_horizon=32,
            device="cpu",
        )
