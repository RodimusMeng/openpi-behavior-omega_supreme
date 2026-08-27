import pytest

from openpi.weaver import dynamics_contract


def test_history_future_targets_use_preceding_transition() -> None:
    alignment = dynamics_contract.prediction_alignment(sequence_length=10, n_history=2)
    positions = list(range(10))

    assert positions[alignment.future_target_slice] == list(range(2, 10))
    assert positions[alignment.predecessor_transition_slice] == list(range(1, 9))
    assert positions[alignment.temporal_pair_slice] == list(range(1, 9))
    assert alignment.horizon == 8


def test_terminal_action_is_not_used_for_last_in_window_target() -> None:
    transitions = [True] * 9 + [False]
    alignment = dynamics_contract.prediction_alignment(sequence_length=10, n_history=2)

    assert all(transitions[alignment.predecessor_transition_slice])
    assert transitions[-1] is False


@pytest.mark.parametrize(
    ("sequence_length", "n_history"),
    [(0, 1), (10, 0), (10, 10), (10, 11), (True, 1)],
)
def test_invalid_alignment_is_rejected(sequence_length: int, n_history: int) -> None:
    with pytest.raises(ValueError, match=r"sequence_length|n_history"):
        dynamics_contract.prediction_alignment(sequence_length, n_history)
