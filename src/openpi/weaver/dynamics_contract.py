"""Time-alignment contract for WEAVER dynamics-only training."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PredictionAlignment:
    """Slices connecting future targets to their preceding transitions.

    For an observation/action window ``[t-Hh+1, ..., t+H]``:

    * predicted target frames are positions ``Hh .. T-1``;
    * target frame ``j`` is enabled by the transition at ``j-1``;
    * temporal pairs use the same preceding-transition positions.

    The final action row may therefore be terminal-invalid without invalidating
    the final observation target; it would lead to a frame outside the window.
    """

    sequence_length: int
    n_history: int
    future_target_slice: slice
    predecessor_transition_slice: slice
    temporal_pair_slice: slice

    @property
    def horizon(self) -> int:
        return self.sequence_length - self.n_history


def prediction_alignment(sequence_length: int, n_history: int) -> PredictionAlignment:
    """Return the only supported action/target alignment for a fixed window."""

    for name, value in (("sequence_length", sequence_length), ("n_history", n_history)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if n_history >= sequence_length:
        raise ValueError(
            f"n_history must leave at least one future target, got {n_history}/{sequence_length}"
        )
    predecessor = slice(n_history - 1, sequence_length - 1)
    return PredictionAlignment(
        sequence_length=sequence_length,
        n_history=n_history,
        future_target_slice=slice(n_history, sequence_length),
        predecessor_transition_slice=predecessor,
        temporal_pair_slice=predecessor,
    )
