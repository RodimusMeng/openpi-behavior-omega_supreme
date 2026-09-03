"""Shared offline/online causal action contract for BEHAVIOR WEAVER.

The 21 base/torso/arm dimensions are frozen phase-affine estimates of the
realized 5 Hz state delta.  The two contact-dependent gripper dimensions stay
as binary controller commands.  No observed future state enters this module.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.weaver import b1k_dataset_adapter
from openpi.weaver.b1k_contract import build_r1pro_contract

AUDIT_SCHEMA = "b1k_weaver_action_semantics_audit_v3"
HYBRID_SCHEMA = "b1k_weaver_hybrid_causal_action_v1"
HYBRID_REPRESENTATION = "r1pro_21d_phase_affine_delta_plus_2d_binary_gripper_command_v1"
GRIPPER_COMPONENTS = frozenset(("left_gripper", "right_gripper"))
MIN_HELD_OUT_R2 = {"base": 0.70, "torso": 0.95, "left_arm": 0.90, "right_arm": 0.90}


@dataclasses.dataclass(frozen=True)
class ComponentAffine:
    name: str
    indices: tuple[int, ...]
    phase: float
    scale: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        if self.name in GRIPPER_COMPONENTS:
            raise ValueError("Gripper components must remain commands, not affine deltas")
        if not 0.0 <= self.phase <= 1.0:
            raise ValueError(f"Invalid phase for {self.name}: {self.phase}")
        scale = np.asarray(self.scale, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        expected = (len(self.indices),)
        if scale.shape != expected or bias.shape != expected or not np.isfinite(scale).all() or not np.isfinite(bias).all():
            raise ValueError(f"Invalid affine parameters for {self.name}")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "bias", bias)
        scale.setflags(write=False)
        bias.setflags(write=False)


@dataclasses.dataclass(frozen=True)
class HybridActionContract:
    components: tuple[ComponentAffine, ...]
    gripper_phase: float
    source_audit_sha256: str
    source_audit_path: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.gripper_phase <= 1.0:
            raise ValueError("gripper_phase must be in [0,1]")
        covered = sorted(index for component in self.components for index in component.indices)
        expected = sorted(set(range(23)) - set(gripper_indices()))
        if covered != expected:
            raise ValueError(f"Hybrid affine components cover {covered}, expected {expected}")
        if len(self.source_audit_sha256) != 64:
            raise ValueError("source_audit_sha256 must be a SHA-256 digest")

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": HYBRID_SCHEMA,
            "representation": HYBRID_REPRESENTATION,
            "source_audit_path": self.source_audit_path,
            "source_audit_sha256": self.source_audit_sha256,
            "gripper_phase": self.gripper_phase,
            "components": [
                {
                    "name": value.name,
                    "indices": list(value.indices),
                    "phase": value.phase,
                    "scale": value.scale.tolist(),
                    "bias": value.bias.tolist(),
                }
                for value in self.components
            ],
        }


def gripper_indices() -> tuple[int, int]:
    contract = build_r1pro_contract()
    values = tuple(
        component.indices[0]
        for component in contract.action_components
        if component.name in GRIPPER_COMPONENTS
    )
    if len(values) != 2:
        raise RuntimeError(f"Expected two R1Pro gripper dimensions, got {values}")
    return values


def load_audited_contract(path: str | Path, *, gripper_phase: float = 0.75) -> HybridActionContract:
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError(f"Unsupported causal audit schema: {payload.get('schema_version')!r}")
    if payload.get("phase_definition", {}).get("interval_local") is not True:
        raise ValueError("Hybrid contract requires an interval-local phase audit")
    fit_ids = payload.get("fit_episode_indices")
    held_ids = payload.get("held_out_episode_indices")
    if not fit_ids or not held_ids or set(fit_ids) & set(held_ids):
        raise ValueError("Hybrid contract requires a disjoint episode-level held-out audit")
    _validate_audit_gate(payload)

    robot = build_r1pro_contract()
    expected = {component.name: tuple(component.indices) for component in robot.action_components}
    fitted = payload.get("fitted_adapter")
    if not isinstance(fitted, Mapping):
        raise ValueError("Causal audit has no fitted_adapter")
    components = []
    for name, indices in expected.items():
        if name in GRIPPER_COMPONENTS:
            continue
        spec = fitted.get(name)
        if not isinstance(spec, Mapping):
            raise ValueError(f"Causal audit is missing {name}")
        dimensions = spec.get("dimensions")
        if not isinstance(dimensions, list) or tuple(int(value["index"]) for value in dimensions) != indices:
            raise ValueError(f"Causal audit has wrong dimensions for {name}")
        components.append(
            ComponentAffine(
                name=name,
                indices=indices,
                phase=float(spec["phase"]),
                scale=np.asarray([value["scale"] for value in dimensions]),
                bias=np.asarray([value["bias"] for value in dimensions]),
            )
        )
    return HybridActionContract(
        components=tuple(components),
        gripper_phase=float(gripper_phase),
        source_audit_sha256=hashlib.sha256(raw).hexdigest(),
        source_audit_path=str(resolved),
    )


def _validate_audit_gate(payload: Mapping[str, Any]) -> None:
    held = payload.get("evaluation", {}).get("held_out", {}).get("components", {})
    for component, minimum in MIN_HELD_OUT_R2.items():
        result = held.get(component)
        if not isinstance(result, Mapping) or result.get("r2") is None or float(result["r2"]) < minimum:
            raise ValueError(f"Held-out causal gate failed for {component}: required R2 >= {minimum}")
    events = payload.get("gripper_events", {}).get("held_out", {})
    for component in GRIPPER_COMPONENTS:
        result = events.get(component)
        if not isinstance(result, Mapping):
            raise ValueError(f"Held-out causal gate has no {component} event audit")
        commands = {float(value) for value in result.get("command_unique", ())}
        if commands != {-1.0, 1.0} or int(result.get("switch_count", 0)) <= 0:
            raise ValueError(f"Held-out {component} must prove binary -1/+1 commands with switch events")


def phase_index(command_count: int, phase: float) -> int:
    if command_count < 1 or not 0.0 <= phase <= 1.0:
        raise ValueError("Phase indexing requires a non-empty interval and phase in [0,1]")
    return round(phase * (command_count - 1))


def interval_action(
    contract: HybridActionContract,
    current_state: np.ndarray,
    policy_commands: np.ndarray,
) -> np.ndarray:
    state = np.asarray(current_state, dtype=np.float32)
    commands = np.asarray(policy_commands, dtype=np.float32)
    if state.shape != (23,) or commands.ndim != 2 or commands.shape[1] != 23 or not len(commands):
        raise ValueError("Hybrid action requires current_state (23,) and non-empty commands (S,23)")
    if not np.isfinite(state).all() or not np.isfinite(commands).all():
        raise ValueError("Hybrid action inputs must be finite")
    result = np.zeros(23, dtype=np.float32)
    for component in contract.components:
        selected = commands[phase_index(len(commands), component.phase), list(component.indices)]
        error = selected - state[list(component.indices)]
        result[list(component.indices)] = component.scale * error + component.bias
    grip_command = commands[phase_index(len(commands), contract.gripper_phase)]
    for index in gripper_indices():
        value = float(grip_command[index])
        if not (np.isclose(value, -1.0, atol=1e-5) or np.isclose(value, 1.0, atol=1e-5)):
            if abs(value - 1.0) <= 0.10:
                value = 1.0
            elif abs(value + 1.0) <= 0.10:
                value = -1.0
            else:
                raise ValueError(
                    f"Gripper command at index {index} is not close to binary -1/+1: {value}"
                )
        result[index] = -1.0 if value < 0 else 1.0
    return result


def materialize_episode_actions(numeric: Any, contract: HybridActionContract) -> np.ndarray:
    actions = np.zeros((numeric.length, 23), dtype=np.float32)
    for index in range(numeric.length - 1):
        if not bool(numeric.transition_valid[index]):
            continue
        start = int(numeric.source_frame_indices[index])
        end = int(numeric.source_frame_indices[index + 1])
        actions[index] = interval_action(contract, numeric.states[index], numeric.source_policy_actions[start:end])
    return actions


def predict_candidate_actions(contract: HybridActionContract, current_state: np.ndarray, segmented: Any) -> np.ndarray:
    state = np.asarray(current_state, dtype=np.float32)
    if state.shape == (23,):
        state = np.repeat(state[None], segmented.num_candidates, axis=0)
    if state.shape != (segmented.num_candidates, 23):
        raise ValueError("current_state has the wrong candidate/state shape")
    state = state.copy()
    output = np.zeros((segmented.num_candidates, segmented.horizon, 23), dtype=np.float32)
    grip = set(gripper_indices())
    for horizon in range(segmented.horizon):
        for candidate in range(segmented.num_candidates):
            length = int(np.sum(segmented.mask[candidate, horizon]))
            output[candidate, horizon] = interval_action(
                contract,
                state[candidate],
                segmented.policy_actions[candidate, horizon, :length],
            )
        state += output[:, horizon]
        for index in b1k_dataset_adapter.angular_state_indices():
            state[:, index] = np.arctan2(np.sin(state[:, index]), np.cos(state[:, index]))
        for index in grip:
            state[:, index] -= output[:, horizon, index]
    return output
