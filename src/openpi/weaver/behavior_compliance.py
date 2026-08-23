"""Machine-checkable BEHAVIOR 2026 challenge-track compliance primitives.

The official evaluator is the source of truth.  This module only validates the
boundary around a policy server: onboard RGB/depth/proprio observations in,
one finite robot action out, and no privileged oracle labels in a submission
artifact.  Offline BDDL/progress labels may be recorded separately as
privileged training supervision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ALLOWED_OBSERVATION_ROOTS = frozenset({"rgb", "depth", "depth_linear", "proprio"})
ALLOWED_EVALUATOR_METADATA = frozenset({"task_id", "cam_rel_poses", "need_new_action"})
FORBIDDEN_OBSERVATION_TOKENS = frozenset(
    {"segmentation", "object_state", "target_pose", "pointcloud", "full_scene", "global_pose"}
)


@dataclasses.dataclass(frozen=True)
class ComplianceViolation:
    code: str
    detail: str


def validate_observation_keys(
    keys: Iterable[str], *, include_evaluator_metadata: bool = False
) -> tuple[ComplianceViolation, ...]:
    """Reject simulator-only observations and unknown roots (fail closed).

    ``keys`` should be the keys actually consumed by the policy.  The official
    websocket evaluator also injects transport metadata (``task_id``, camera
    relative poses, and action-cache hints); callers may validate those keys
    separately with ``include_evaluator_metadata=True`` without treating them
    as onboard sensor modalities.
    """

    violations: list[ComplianceViolation] = []
    for raw_key in keys:
        key = str(raw_key).lower()
        # Official flatten_obs_dict joins nested names with ``::``; the final
        # component is the sensor modality (rgb/depth_linear/proprio).
        modality = key.rsplit("::", 1)[-1].split(".", 1)[-1].split("/", 1)[-1]
        if include_evaluator_metadata and modality in ALLOWED_EVALUATOR_METADATA:
            continue
        if any(token in key for token in FORBIDDEN_OBSERVATION_TOKENS):
            violations.append(ComplianceViolation("forbidden_observation", str(raw_key)))
        elif modality not in ALLOWED_OBSERVATION_ROOTS:
            violations.append(ComplianceViolation("unknown_observation", str(raw_key)))
    return tuple(violations)


def validate_action(action: Any, *, action_dim: int) -> tuple[ComplianceViolation, ...]:
    """Validate the websocket action payload before it reaches OmniGibson."""

    violations: list[ComplianceViolation] = []
    value = np.asarray(action)
    if value.shape != (action_dim,):
        violations.append(ComplianceViolation("action_shape", f"expected {(action_dim,)}, got {value.shape}"))
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        violations.append(ComplianceViolation("action_finite", "action must be finite numeric data"))
    return tuple(violations)


def build_manifest(
    *,
    behavior_git_sha: str,
    omega_git_sha: str,
    weaver_git_sha: str,
    wrapper: Mapping[str, Any],
    robot_config: Mapping[str, Any],
    observation_keys: Sequence[str],
    action_dim: int,
    task_ids: Sequence[str],
    eval_mode: str,
    instance_indices: Sequence[int],
    num_rollouts: int,
    candidate_pool: Mapping[str, Any] | None = None,
    oracle_used: bool = False,
    privileged_offline_labels: bool = False,
    behavior_tag: str = "v3.9.1",
) -> dict[str, Any]:
    """Create an auditable manifest; callers must reject violations first."""

    observation_violations = validate_observation_keys(observation_keys)
    if observation_violations:
        raise ValueError(f"Observation allowlist violation: {observation_violations}")
    if oracle_used:
        raise ValueError("Simulator oracle is diagnostic-only and cannot enter a submission manifest")
    if behavior_tag != "v3.9.1":
        raise ValueError(f"BEHAVIOR challenge evaluation must use v3.9.1, got {behavior_tag!r}")
    if not all(_is_sha(value) for value in (behavior_git_sha, omega_git_sha, weaver_git_sha)):
        raise ValueError("All source revisions are required")
    if action_dim <= 0 or num_rollouts <= 0:
        raise ValueError("action_dim and num_rollouts must be positive")
    return {
        "schema_version": "behavior_2026_compliance_v1",
        "track": "behavior_2026_single_onboard",
        "behavior_tag": behavior_tag,
        "behavior_git_sha": behavior_git_sha,
        "omega_git_sha": omega_git_sha,
        "weaver_git_sha": weaver_git_sha,
        "wrapper": dict(wrapper),
        "robot_config": {**dict(robot_config), "action_dim": int(action_dim)},
        "allowed_observation_keys": sorted({str(key) for key in observation_keys}),
        "forbidden_observation_tokens": sorted(FORBIDDEN_OBSERVATION_TOKENS),
        "task_ids": [str(task) for task in task_ids],
        "eval_mode": str(eval_mode),
        "instance_indices": [int(index) for index in instance_indices],
        "num_rollouts": int(num_rollouts),
        "candidate_pool": dict(candidate_pool) if candidate_pool is not None else None,
        "selector": {"oracle_used": False, "online_inputs_privileged": False},
        "training_labels": {"privileged_offline": bool(privileged_offline_labels)},
        "outputs_unmodified": True,
    }


def validate_manifest(manifest: Mapping[str, Any], *, submission: bool = False) -> tuple[ComplianceViolation, ...]:
    """Validate a manifest before handing it to the official evaluator."""

    violations: list[ComplianceViolation] = []
    if manifest.get("schema_version") != "behavior_2026_compliance_v1":
        violations.append(ComplianceViolation("schema_version", "unsupported compliance manifest"))
    if manifest.get("behavior_tag") != "v3.9.1":
        violations.append(ComplianceViolation("behavior_version", "official evaluator requires v3.9.1"))
    if manifest.get("selector", {}).get("oracle_used") is not False:
        violations.append(ComplianceViolation("oracle", "oracle scores must be isolated from submissions"))
    observations = manifest.get("allowed_observation_keys", ())
    violations.extend(validate_observation_keys(observations))
    if submission:
        if manifest.get("eval_mode") != "public_test":
            violations.append(ComplianceViolation("eval_mode", "reported challenge results must be public_test"))
        if list(manifest.get("instance_indices", ())) != list(range(10)):
            violations.append(ComplianceViolation("instances", "reported public instances must be indices 0..9"))
        if manifest.get("num_rollouts") != 1:
            violations.append(ComplianceViolation("rollouts", "submission requires one rollout per instance"))
        if manifest.get("outputs_unmodified") is not True:
            violations.append(ComplianceViolation("outputs", "evaluation JSON/video must remain unmodified"))
    return tuple(violations)


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    """Atomically write and return the manifest SHA-256."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _is_sha(value: Any) -> bool:
    if not isinstance(value, str) or len(value) not in {40, 64}:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
