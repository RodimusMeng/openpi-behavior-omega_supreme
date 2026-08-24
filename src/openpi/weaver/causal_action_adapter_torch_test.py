from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openpi.weaver import causal_action_adapter as adapter  # noqa: E402
from openpi.weaver.candidate_pool import CandidatePool  # noqa: E402


def _normalization() -> adapter.AdapterNormalization:
    return adapter.AdapterNormalization(
        state_mean=np.zeros(23),
        state_std=np.ones(23),
        policy_action_mean=np.zeros(23),
        policy_action_std=np.ones(23),
        target_delta_mean=np.zeros(23),
        target_delta_std=np.ones(23),
    )


def _samples() -> tuple[adapter.AdapterSample, ...]:
    result = []
    for index, steps in enumerate((4, 6)):
        commands = np.arange(steps * 23, dtype=np.float32).reshape(steps, 23) / 100
        result.append(
            adapter.AdapterSample(
                current_state=np.full(23, index, dtype=np.float32),
                policy_actions=commands,
                action_durations_s=np.full(steps, 0.2 / steps, dtype=np.float32),
                target_state_delta=np.full(23, 0.1 + index, dtype=np.float32),
                split="train",
                episode_index=index,
                interval_index=0,
            )
        )
    return tuple(result)


def test_real_gru_forward_backward_uses_padded_mask():
    torch.manual_seed(7)
    model = adapter.build_torch_model(adapter.AdapterSpec(hidden_dim=16, num_layers=1))
    numpy_batch = adapter.collate_samples(_samples(), _normalization())
    batch = {
        key: torch.as_tensor(value, dtype=torch.bool if key == "mask" else torch.float32)
        for key, value in numpy_batch.items()
    }

    loss, log = adapter.adapter_loss(model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert log["batch_size"].item() == 2
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_gru_is_order_sensitive_instead_of_command_mean_pooling():
    torch.manual_seed(11)
    model = adapter.build_torch_model(adapter.AdapterSpec(hidden_dim=12, num_layers=1)).eval()
    state = torch.zeros(1, 23)
    commands = torch.stack([torch.zeros(23), torch.ones(23), torch.full((23,), 2.0)])[None]
    durations = torch.full((1, 3), 0.2 / 3)
    mask = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        forward = model(state, commands, durations, mask)
        reversed_output = model(state, commands.flip(1), durations, mask)

    assert not torch.allclose(forward, reversed_output)


@pytest.mark.parametrize(
    ("mask", "durations", "match"),
    [
        ([True, False, True], [0.05, 0.0, 0.15], "left-aligned prefix"),
        ([True, True, False], [0.05, 0.0, 0.0], "strictly positive"),
        ([True, True, False], [0.05, float("nan"), 0.0], "finite"),
        ([True, True, False], [0.05, 0.15, 0.01], "exactly zero"),
    ],
)
def test_gru_fails_closed_on_invalid_mask_and_duration_contract(mask, durations, match):
    model = adapter.build_torch_model(adapter.AdapterSpec(hidden_dim=8, num_layers=1))
    with pytest.raises(ValueError, match=match):
        model(
            torch.zeros(1, 23),
            torch.zeros(1, 3, 23),
            torch.tensor([durations], dtype=torch.float32),
            torch.tensor([mask], dtype=torch.bool),
        )


class _DurationWeightedCommands(torch.nn.Module):
    def forward(self, current_state, policy_actions, action_durations_s, mask):
        del current_state
        weights = action_durations_s * mask
        return (policy_actions * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True)


def test_online_prediction_returns_raw_delta_then_separate_weaver_normalization():
    actions = np.zeros((2, 12, 23), dtype=np.float32)
    actions[0] = 2.0
    actions[1] = 4.0
    segmented = adapter.segment_candidate_actions(actions, np.arange(13) / 30, target_hz=5)

    raw = adapter.predict_canonical_deltas(
        _DurationWeightedCommands(),
        _normalization(),
        np.zeros(23, dtype=np.float32),
        segmented,
        device="cpu",
    )
    normalized = adapter.normalize_for_weaver(
        raw,
        weaver_action_mean=np.ones(23),
        weaver_action_std=np.full(23, 2.0),
    )

    assert raw.shape == (2, 2, 23)
    np.testing.assert_allclose(raw[0], 2.0)
    np.testing.assert_allclose(raw[1], 4.0)
    np.testing.assert_allclose(normalized[0], 0.5)
    np.testing.assert_allclose(normalized[1], 1.5)


def test_adapted_actions_are_linked_to_the_exact_candidate_pool():
    actions = np.full((2, 6, 23), 3.0, dtype=np.float32)
    pool = CandidatePool(actions=actions, observation_digest="0" * 64)

    adapted = adapter.adapt_candidate_pool(
        pool,
        _DurationWeightedCommands(),
        _normalization(),
        np.zeros(23),
        np.arange(7) / 30,
        weaver_action_mean=np.zeros(23),
        weaver_action_std=np.ones(23),
        device="cpu",
    )

    assert adapted.source_pool_actions_sha256 == pool.actions_sha256
    assert adapted.source_pool_observation_digest == pool.observation_digest
    assert adapted.source_action_time_edges_sha256 == adapter.action_time_edges_sha256(np.arange(7) / 30)
    assert adapted.normalized_weaver_actions.shape == (2, 1, 23)
