"""Causal R1Pro policy-command to WEAVER-action adaptation.

Offline WEAVER actions are observed 5 Hz compact-state deltas.  They are not
available for an unexecuted policy candidate.  This module keeps the raw 30 Hz
controller commands ordered and duration-aware; it never applies a component-
wise sum or mean to their mixed velocity/absolute-target/gripper semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from openpi.weaver import b1k_dataset_adapter
from openpi.weaver import candidate_pool
from openpi.weaver.behavior_precomputed import load_numeric_episode

ADAPTER_SCHEMA = "b1k_causal_policy_action_adapter_v1"
STATE_DIM = 23
ACTION_DIM = 23


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


@dataclasses.dataclass(frozen=True)
class AdapterSample:
    """One real episode-internal 30 Hz -> 5 Hz transition."""

    current_state: np.ndarray
    policy_actions: np.ndarray
    action_durations_s: np.ndarray
    target_state_delta: np.ndarray
    split: str
    episode_index: int
    interval_index: int

    def __post_init__(self) -> None:
        state = _finite_array(self.current_state, name="current_state", ndim=1).astype(np.float32, copy=True)
        actions = _finite_array(self.policy_actions, name="policy_actions", ndim=2).astype(np.float32, copy=True)
        durations = _finite_array(self.action_durations_s, name="action_durations_s", ndim=1).astype(
            np.float32, copy=True
        )
        target = _finite_array(self.target_state_delta, name="target_state_delta", ndim=1).astype(
            np.float32, copy=True
        )
        if state.shape != (STATE_DIM,) or target.shape != (STATE_DIM,):
            raise ValueError(f"state/target must be {STATE_DIM}-D, got {state.shape}/{target.shape}")
        if actions.shape[1] != ACTION_DIM or actions.shape[0] == 0:
            raise ValueError(f"policy_actions must have non-empty shape (S,{ACTION_DIM}), got {actions.shape}")
        if durations.shape != (actions.shape[0],) or np.any(durations <= 0):
            raise ValueError("action_durations_s must be positive with one duration per command")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {self.split!r}")
        if self.episode_index < 0 or self.interval_index < 0:
            raise ValueError("episode_index and interval_index must be non-negative")
        for array in (state, actions, durations, target):
            array.setflags(write=False)
        object.__setattr__(self, "current_state", state)
        object.__setattr__(self, "policy_actions", actions)
        object.__setattr__(self, "action_durations_s", durations)
        object.__setattr__(self, "target_state_delta", target)


@dataclasses.dataclass(frozen=True)
class AdapterNormalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    policy_action_mean: np.ndarray
    policy_action_std: np.ndarray
    target_delta_mean: np.ndarray
    target_delta_std: np.ndarray

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = _finite_array(getattr(self, field.name), name=field.name, ndim=1).astype(np.float32, copy=True)
            if value.shape != (STATE_DIM,):
                raise ValueError(f"{field.name} must be {STATE_DIM}-D, got {value.shape}")
            if field.name.endswith("_std") and np.any(value <= 0):
                raise ValueError(f"{field.name} must be strictly positive")
            value.setflags(write=False)
            object.__setattr__(self, field.name, value)

    def to_dict(self) -> dict[str, list[float]]:
        return {field.name: getattr(self, field.name).tolist() for field in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdapterNormalization:
        return cls(**{field.name: np.asarray(value[field.name], dtype=np.float32) for field in dataclasses.fields(cls)})


@dataclasses.dataclass(frozen=True)
class AdapterSpec:
    """Small sequence model; the interface, not this size, is the contract."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "num_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite in [0,1)")


@dataclasses.dataclass(frozen=True)
class SegmentedCandidates:
    policy_actions: np.ndarray  # (N,H,S,23)
    action_durations_s: np.ndarray  # (N,H,S)
    mask: np.ndarray  # (N,H,S)
    interval_edges_s: np.ndarray  # (H+1,)

    @property
    def num_candidates(self) -> int:
        return int(self.policy_actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.policy_actions.shape[1])


@dataclasses.dataclass(frozen=True)
class AdaptedCandidateActions:
    """Causal WM actions byte-linked to one immutable policy pool."""

    canonical_deltas: np.ndarray
    normalized_weaver_actions: np.ndarray
    source_pool_actions_sha256: str
    interval_edges_s: np.ndarray
    source_pool_observation_digest: str | None = None
    source_action_time_edges_sha256: str | None = None

    def __post_init__(self) -> None:
        raw = _finite_array(self.canonical_deltas, name="canonical_deltas", ndim=3).astype(np.float32, copy=True)
        normalized = _finite_array(
            self.normalized_weaver_actions, name="normalized_weaver_actions", ndim=3
        ).astype(np.float32, copy=True)
        if raw.shape != normalized.shape or raw.shape[-1] != ACTION_DIM or 0 in raw.shape:
            raise ValueError(f"adapted candidate actions must share non-empty (N,H,{ACTION_DIM}) shape")
        if len(self.source_pool_actions_sha256) != 64:
            raise ValueError("source_pool_actions_sha256 must be a SHA-256 hex digest")
        try:
            bytes.fromhex(self.source_pool_actions_sha256)
        except ValueError as exc:
            raise ValueError("source_pool_actions_sha256 must be a SHA-256 hex digest") from exc
        for name in ("source_pool_observation_digest", "source_action_time_edges_sha256"):
            value = getattr(self, name)
            if value is not None:
                if len(value) != 64:
                    raise ValueError(f"{name} must be a SHA-256 hex digest")
                try:
                    bytes.fromhex(value)
                except ValueError as exc:
                    raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
        edges = _finite_array(self.interval_edges_s, name="interval_edges_s", ndim=1).astype(np.float64, copy=True)
        if edges.shape != (raw.shape[1] + 1,) or np.any(np.diff(edges) <= 0):
            raise ValueError("interval_edges_s must be strictly increasing with length H+1")
        for array in (raw, normalized, edges):
            array.setflags(write=False)
        object.__setattr__(self, "canonical_deltas", raw)
        object.__setattr__(self, "normalized_weaver_actions", normalized)
        object.__setattr__(self, "interval_edges_s", edges)


def read_manifest_index(dataset_root: str | Path, *, split: str) -> tuple[dict[str, Any], ...]:
    """Read prepared manifests for exactly one declared split."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test, got {split!r}")
    index_path = Path(dataset_root).expanduser().resolve() / "manifest.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing prepared WEAVER manifest index: {index_path}")
    result = []
    with index_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                manifest = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {index_path}:{line_number}") from exc
            if manifest.get("split") == split:
                result.append(manifest)
    if not result:
        raise ValueError(f"Prepared dataset has no episodes for split {split!r}")
    return tuple(result)


def episode_adapter_samples(dataset_root: str | Path, manifest: Mapping[str, Any]) -> tuple[AdapterSample, ...]:
    """Materialize every valid interval from one prepared numeric episode."""

    numeric = load_numeric_episode(dataset_root, manifest)
    samples = []
    for interval_index in range(numeric.length - 1):
        if not bool(numeric.transition_valid[interval_index]):
            continue
        start = int(numeric.source_frame_indices[interval_index])
        end = int(numeric.source_frame_indices[interval_index + 1])
        if not 0 <= start < end < len(numeric.source_timestamps_s):
            raise ValueError(
                f"Episode {manifest['episode_index']} interval {interval_index} has invalid source bounds {start}:{end}"
            )
        time_edges = numeric.source_timestamps_s[start : end + 1]
        samples.append(
            AdapterSample(
                current_state=numeric.states[interval_index],
                policy_actions=numeric.source_policy_actions[start:end],
                action_durations_s=np.diff(time_edges),
                target_state_delta=numeric.actions[interval_index],
                split=str(manifest["split"]),
                episode_index=int(manifest["episode_index"]),
                interval_index=interval_index,
            )
        )
    if not samples:
        raise ValueError(f"Episode {manifest['episode_index']} contains no valid causal adapter intervals")
    return tuple(samples)


def iter_adapter_samples(
    dataset_root: str | Path,
    *,
    split: str,
    max_samples: int | None = None,
) -> Iterable[AdapterSample]:
    if max_samples is not None and (isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0):
        raise ValueError("max_samples must be a positive integer when supplied")
    emitted = 0
    for manifest in read_manifest_index(dataset_root, split=split):
        for sample in episode_adapter_samples(dataset_root, manifest):
            yield sample
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                return


def compute_normalization(samples: Sequence[AdapterSample], *, min_std: float = 1e-6) -> AdapterNormalization:
    """Compute normalization from train samples only; val/test cannot leak."""

    if not samples:
        raise ValueError("Cannot compute adapter normalization from no samples")
    if any(sample.split != "train" for sample in samples):
        raise ValueError("Adapter normalization may be computed from train split only")
    if not math.isfinite(min_std) or min_std <= 0:
        raise ValueError("min_std must be finite and positive")
    states = np.stack([sample.current_state for sample in samples]).astype(np.float64)
    targets = np.stack([sample.target_state_delta for sample in samples]).astype(np.float64)
    commands = np.concatenate([sample.policy_actions for sample in samples], axis=0).astype(np.float64)

    def pair(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return value.mean(axis=0).astype(np.float32), np.maximum(value.std(axis=0), min_std).astype(np.float32)

    state_mean, state_std = pair(states)
    policy_mean, policy_std = pair(commands)
    target_mean, target_std = pair(targets)
    return AdapterNormalization(state_mean, state_std, policy_mean, policy_std, target_mean, target_std)


def collate_samples(samples: Sequence[AdapterSample], normalization: AdapterNormalization) -> dict[str, np.ndarray]:
    """Pad variable command sequences while retaining ordering and duration."""

    if not samples:
        raise ValueError("Cannot collate an empty adapter batch")
    max_steps = max(len(sample.policy_actions) for sample in samples)
    actions = np.zeros((len(samples), max_steps, ACTION_DIM), dtype=np.float32)
    durations = np.zeros((len(samples), max_steps), dtype=np.float32)
    mask = np.zeros((len(samples), max_steps), dtype=np.bool_)
    for index, sample in enumerate(samples):
        length = len(sample.policy_actions)
        actions[index, :length] = (sample.policy_actions - normalization.policy_action_mean) / (
            normalization.policy_action_std
        )
        durations[index, :length] = sample.action_durations_s
        mask[index, :length] = True
    states = (np.stack([sample.current_state for sample in samples]) - normalization.state_mean) / (
        normalization.state_std
    )
    targets = (np.stack([sample.target_state_delta for sample in samples]) - normalization.target_delta_mean) / (
        normalization.target_delta_std
    )
    return {
        "current_state": states.astype(np.float32),
        "policy_actions": actions,
        "action_durations_s": durations,
        "mask": mask,
        "target_state_delta": targets.astype(np.float32),
    }


def build_torch_model(spec: AdapterSpec) -> Any:
    """Build the learned causal adapter without importing torch for data audits."""

    torch = _require_torch()

    class CausalR1ProActionAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.command_encoder = torch.nn.GRU(
                ACTION_DIM + 1,
                spec.hidden_dim,
                num_layers=spec.num_layers,
                dropout=spec.dropout if spec.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.state_encoder = torch.nn.Sequential(
                torch.nn.Linear(STATE_DIM, spec.hidden_dim),
                torch.nn.SiLU(),
            )
            self.output = torch.nn.Sequential(
                torch.nn.Linear(spec.hidden_dim * 2, spec.hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(spec.hidden_dim, STATE_DIM),
            )

        def forward(self, current_state, policy_actions, action_durations_s, mask):
            if policy_actions.ndim != 3 or policy_actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"policy_actions must be (B,S,{ACTION_DIM})")
            if current_state.shape != (policy_actions.shape[0], STATE_DIM):
                raise ValueError(f"current_state must be (B,{STATE_DIM})")
            if action_durations_s.shape != policy_actions.shape[:2] or mask.shape != policy_actions.shape[:2]:
                raise ValueError("duration/mask must match the policy-action batch/time axes")
            if mask.dtype != torch.bool or not bool(mask.any(dim=1).all()):
                raise ValueError("every adapter sample must contain at least one valid command")
            lengths = mask.sum(dim=1)
            prefix = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
            if not torch.equal(mask, prefix):
                raise ValueError("adapter command mask must be a contiguous left-aligned prefix")
            if not bool(torch.isfinite(action_durations_s).all()):
                raise ValueError("action_durations_s must contain only finite values")
            if not bool((action_durations_s[mask] > 0).all()):
                raise ValueError("valid action durations must be strictly positive")
            if bool((action_durations_s[~mask] != 0).any()):
                raise ValueError("padded action durations must be exactly zero")
            sequence = torch.cat([policy_actions, action_durations_s.unsqueeze(-1)], dim=-1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sequence,
                mask.sum(dim=1).to(device="cpu"),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.command_encoder(packed)
            encoded_state = self.state_encoder(current_state)
            return self.output(torch.cat([encoded_state, hidden[-1]], dim=-1))

    model = CausalR1ProActionAdapter()
    model.adapter_spec = dataclasses.asdict(spec)
    return model


def adapter_loss(model: Any, batch: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Run the real normalized-delta regression loss."""

    torch = _require_torch()
    required = {
        "current_state",
        "policy_actions",
        "action_durations_s",
        "mask",
        "target_state_delta",
    }
    if set(batch) != required:
        raise ValueError(f"Adapter batch keys must be exactly {sorted(required)}, got {sorted(batch)}")
    predicted = model(
        batch["current_state"],
        batch["policy_actions"],
        batch["action_durations_s"],
        batch["mask"],
    )
    target = batch["target_state_delta"]
    if predicted.shape != target.shape or predicted.shape[-1] != STATE_DIM:
        raise ValueError(f"Adapter prediction/target must share (B,{STATE_DIM}) shape")
    loss = torch.nn.functional.mse_loss(predicted, target)
    return loss, {
        "loss": loss.detach(),
        "normalized_mae": (predicted - target).abs().mean().detach(),
        "batch_size": predicted.new_tensor(predicted.shape[0]),
    }


def segment_candidate_actions(
    candidate_actions: np.ndarray,
    action_time_edges_s: np.ndarray,
    *,
    target_hz: float = 5.0,
    max_horizon: int | None = None,
) -> SegmentedCandidates:
    """Group a frozen ``(N,T,23)`` pool into complete WM intervals.

    A command that straddles a boundary is retained in both intervals with its
    clipped active duration. A trailing partial interval is excluded rather
    than filled with commands the policy never proposed.
    """

    actions = _finite_array(candidate_actions, name="candidate_actions", ndim=3).astype(np.float32, copy=False)
    if actions.shape[2] != ACTION_DIM or actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError(f"candidate_actions must have non-empty shape (N,T,{ACTION_DIM}), got {actions.shape}")
    edges = _finite_array(action_time_edges_s, name="action_time_edges_s", ndim=1).astype(np.float64, copy=False)
    if edges.shape != (actions.shape[1] + 1,) or np.any(np.diff(edges) <= 0):
        raise ValueError("action_time_edges_s must be strictly increasing with length T+1")
    if not math.isfinite(target_hz) or target_hz <= 0:
        raise ValueError("target_hz must be finite and positive")
    if max_horizon is not None and (isinstance(max_horizon, bool) or not isinstance(max_horizon, int) or max_horizon <= 0):
        raise ValueError("max_horizon must be a positive integer when supplied")
    interval = 1.0 / target_hz
    horizon = math.floor((float(edges[-1] - edges[0]) + 1e-9) / interval)
    if max_horizon is not None:
        horizon = min(horizon, max_horizon)
    if horizon <= 0:
        raise ValueError("Candidate chunk contains no complete world-model interval")
    wm_edges = edges[0] + np.arange(horizon + 1, dtype=np.float64) * interval
    segments: list[list[tuple[int, float]]] = []
    for wm_start, wm_end in itertools.pairwise(wm_edges):
        row = []
        for action_index, (action_start, action_end) in enumerate(itertools.pairwise(edges)):
            overlap = min(wm_end, action_end) - max(wm_start, action_start)
            if overlap > 1e-12:
                row.append((action_index, float(overlap)))
        if not row or not math.isclose(sum(duration for _, duration in row), interval, rel_tol=0, abs_tol=1e-7):
            raise ValueError("Candidate timing does not fully cover a requested world-model interval")
        segments.append(row)
    max_steps = max(len(row) for row in segments)
    output = np.zeros((actions.shape[0], horizon, max_steps, ACTION_DIM), dtype=np.float32)
    durations = np.zeros((actions.shape[0], horizon, max_steps), dtype=np.float32)
    mask = np.zeros((actions.shape[0], horizon, max_steps), dtype=np.bool_)
    for wm_index, row in enumerate(segments):
        for step, (action_index, duration) in enumerate(row):
            output[:, wm_index, step] = actions[:, action_index]
            durations[:, wm_index, step] = duration
            mask[:, wm_index, step] = True
    return SegmentedCandidates(output, durations, mask, wm_edges)


def available_world_model_horizon(
    action_horizon: int,
    *,
    control_hz: float = 30.0,
    target_hz: float = 5.0,
) -> int:
    """Return complete WM intervals covered by a fixed-rate policy chunk."""

    if isinstance(action_horizon, bool) or not isinstance(action_horizon, int) or action_horizon <= 0:
        raise ValueError("action_horizon must be a positive integer")
    for name, value in (("control_hz", control_hz), ("target_hz", target_hz)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return math.floor((action_horizon / control_hz + 1e-9) * target_hz)


def require_world_model_horizon(
    action_horizon: int,
    required_horizon: int,
    *,
    control_hz: float = 30.0,
    target_hz: float = 5.0,
) -> int:
    """Fail before policy inference when a chunk cannot cover a WM rollout."""

    if isinstance(required_horizon, bool) or not isinstance(required_horizon, int) or required_horizon <= 0:
        raise ValueError("required_horizon must be a positive integer")
    available = available_world_model_horizon(
        action_horizon,
        control_hz=control_hz,
        target_hz=target_hz,
    )
    if available < required_horizon:
        raise ValueError(
            "Policy candidate chunk cannot cover the configured WEAVER imagination horizon: "
            f"action_horizon={action_horizon} at {control_hz:g} Hz provides {available} complete "
            f"intervals at {target_hz:g} Hz, required={required_horizon}"
        )
    return available


def predict_canonical_deltas(
    model: Any,
    normalization: AdapterNormalization,
    current_state: np.ndarray,
    segmented: SegmentedCandidates,
    *,
    device: str,
) -> np.ndarray:
    """Autoregressively predict raw canonical deltas for every candidate.

    The current state is updated only with adapter predictions. No observed
    future state or privileged simulator value enters this online path.
    """

    torch = _require_torch()
    state = _finite_array(current_state, name="current_state")
    if state.shape == (STATE_DIM,):
        state = np.repeat(state[None], segmented.num_candidates, axis=0)
    expected_state_shape = (segmented.num_candidates, STATE_DIM)
    if state.shape != expected_state_shape:
        raise ValueError(f"current_state must be ({STATE_DIM},) or {expected_state_shape}, got {state.shape}")
    canonical_state = state.astype(np.float32, copy=True)
    predictions = []
    model.eval()
    with torch.no_grad():
        for interval_index in range(segmented.horizon):
            commands = segmented.policy_actions[:, interval_index]
            normalized_commands = (commands - normalization.policy_action_mean) / (
                normalization.policy_action_std
            )
            normalized_state = (canonical_state - normalization.state_mean) / normalization.state_std
            predicted_normalized = model(
                torch.as_tensor(normalized_state, dtype=torch.float32, device=device),
                torch.as_tensor(normalized_commands, dtype=torch.float32, device=device),
                torch.as_tensor(
                    segmented.action_durations_s[:, interval_index],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(segmented.mask[:, interval_index], dtype=torch.bool, device=device),
            )
            if predicted_normalized.shape != expected_state_shape:
                raise ValueError(
                    f"Adapter inference must return {expected_state_shape}, got {tuple(predicted_normalized.shape)}"
                )
            delta = predicted_normalized.detach().cpu().numpy() * normalization.target_delta_std
            delta += normalization.target_delta_mean
            if not np.isfinite(delta).all():
                raise FloatingPointError("Causal action adapter produced non-finite state deltas")
            delta = delta.astype(np.float32, copy=False)
            predictions.append(delta)
            canonical_state = apply_state_delta(canonical_state, delta)
    return np.stack(predictions, axis=1)


def normalize_for_weaver(
    canonical_deltas: np.ndarray,
    *,
    weaver_action_mean: np.ndarray,
    weaver_action_std: np.ndarray,
) -> np.ndarray:
    """Apply the world model's action stats, not the adapter's target stats."""

    deltas = _finite_array(canonical_deltas, name="canonical_deltas")
    if deltas.ndim != 3 or deltas.shape[-1] != ACTION_DIM:
        raise ValueError(f"canonical_deltas must have shape (N,H,{ACTION_DIM}), got {deltas.shape}")
    mean = _finite_array(weaver_action_mean, name="weaver_action_mean", ndim=1)
    std = _finite_array(weaver_action_std, name="weaver_action_std", ndim=1)
    if mean.shape != (ACTION_DIM,) or std.shape != (ACTION_DIM,) or np.any(std <= 0):
        raise ValueError(f"WEAVER action mean/std must be {ACTION_DIM}-D with positive std")
    normalized = (deltas - mean) / std
    if not np.isfinite(normalized).all():
        raise FloatingPointError("WEAVER action normalization produced non-finite values")
    return normalized.astype(np.float32)


def adapt_candidate_pool(
    pool: candidate_pool.CandidatePool,
    model: Any,
    normalization: AdapterNormalization,
    current_state: np.ndarray,
    action_time_edges_s: np.ndarray,
    *,
    weaver_action_mean: np.ndarray,
    weaver_action_std: np.ndarray,
    device: str,
    target_hz: float = 5.0,
    max_horizon: int | None = None,
) -> AdaptedCandidateActions:
    """Convert exactly one frozen candidate pool into causal WEAVER actions."""

    segmented = segment_candidate_actions(
        pool.actions,
        action_time_edges_s,
        target_hz=target_hz,
        max_horizon=max_horizon,
    )
    canonical = predict_canonical_deltas(model, normalization, current_state, segmented, device=device)
    normalized = normalize_for_weaver(
        canonical,
        weaver_action_mean=weaver_action_mean,
        weaver_action_std=weaver_action_std,
    )
    return AdaptedCandidateActions(
        canonical_deltas=canonical,
        normalized_weaver_actions=normalized,
        source_pool_actions_sha256=pool.actions_sha256,
        interval_edges_s=segmented.interval_edges_s,
        source_pool_observation_digest=pool.observation_digest,
        source_action_time_edges_sha256=action_time_edges_sha256(action_time_edges_s),
    )


def apply_state_delta(state: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Advance compact state and wrap audited torso/arm joint coordinates."""

    state_value = _finite_array(state, name="state")
    delta_value = _finite_array(delta, name="delta")
    if state_value.shape != delta_value.shape or state_value.shape[-1] != STATE_DIM:
        raise ValueError(f"state/delta must share (...,{STATE_DIM}) shape")
    result = np.asarray(state_value + delta_value, dtype=np.float32)
    angular = b1k_dataset_adapter.angular_state_indices()
    result[..., angular] = np.arctan2(np.sin(result[..., angular]), np.cos(result[..., angular]))
    return result


def checkpoint_payload(
    model: Any,
    optimizer: Any,
    *,
    spec: AdapterSpec,
    normalization: AdapterNormalization,
    step: int,
    dataset_semantics: Mapping[str, Any],
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the only supported resumable adapter checkpoint schema."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if not dataset_semantics:
        raise ValueError("dataset_semantics must be non-empty")
    payload = {
        "schema_version": ADAPTER_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "spec": dataclasses.asdict(spec),
        "normalization": normalization.to_dict(),
        "dataset_semantics": dict(dataset_semantics),
    }
    if rng_state is not None:
        payload["rng_state"] = dict(rng_state)
    return payload


def _require_torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("The causal R1Pro action adapter requires PyTorch") from exc
    return torch


def action_time_edges_sha256(value: Any) -> str:
    """Digest one candidate-pool timing vector for downstream linkage."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = __import__("hashlib").sha256()
    digest.update(array.dtype.str.encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()
