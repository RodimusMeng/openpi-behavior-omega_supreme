"""Build an auditable BEHAVIOR configuration from a released WEAVER config.

The downloaded checkpoint configuration is the source of truth for the model
architecture.  This module changes only the embodiment/data fields required by
the first BEHAVIOR transfer and records both sides of that change.  It has no
PyTorch or upstream-WEAVER dependency, so configuration mistakes are caught
before a GPU model or checkpoint is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import dataclasses
from pathlib import Path
from typing import Any

from openpi.weaver.behavior_precomputed import DEFAULT_IMAGE_KEYS

BEHAVIOR_DATASET_NAME = "BEHAVIOR"
BEHAVIOR_STATE_DIM = 23
BEHAVIOR_ACTION_DIM = 23
BEHAVIOR_STATE_REPRESENTATION = "r1pro_compact_state_v1"
BEHAVIOR_ACTION_REPRESENTATION = "r1pro_compact_state_delta_v1"


@dataclasses.dataclass(frozen=True)
class ArchitectureContract:
    """The checkpoint fields that determine WEAVER parameter shapes."""

    image_keys: tuple[str, ...]
    image_size: tuple[int, int]
    n_states: int
    n_actions: int
    spatial_size: int
    horizon: int
    n_history: int
    n_memory_frames: int
    t_memory: int


@dataclasses.dataclass(frozen=True)
class BehaviorConfigReport:
    """Source and target architecture contracts for one transfer build."""

    source: ArchitectureContract
    target: ArchitectureContract
    changed_fields: tuple[str, ...]

    @property
    def is_identity(self) -> bool:
        return not self.changed_fields


def checkpoint_defaults(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copied concrete config from repo-style or saved YAML.

    The upstream repository config stores the runnable configuration under a
    ``defaults`` key, whereas ``save_checkpoint`` writes the concrete config
    directly.  Released checkpoints may therefore use either representation.
    """

    if not isinstance(raw_config, Mapping):
        raise TypeError("WEAVER checkpoint config must be a mapping")
    concrete = raw_config.get("defaults", raw_config)
    if not isinstance(concrete, Mapping):
        raise ValueError("WEAVER checkpoint config defaults must be a mapping")
    return copy.deepcopy(dict(concrete))


def architecture_contract(config: Mapping[str, Any]) -> ArchitectureContract:
    """Extract and validate every currently known shape-bearing contract."""

    cfg = checkpoint_defaults(config)
    try:
        dataset = cfg["dataset"]
        encoder = cfg["im_encoder"]
        image_keys = _string_tuple(dataset["img_keys"], name="dataset.img_keys")
        image_size = _positive_int_tuple(dataset["image_size"], length=2, name="dataset.image_size")
        n_states = _positive_int(dataset["n_states"], name="dataset.n_states")
        n_actions = _positive_int(dataset["n_actions"], name="dataset.n_actions")
        spatial_size = _positive_int(encoder["spatial_size"], name="im_encoder.spatial_size")
        horizon = _positive_int(cfg["horizon"], name="horizon")
        n_history = _positive_int(cfg["n_history"], name="n_history")
        n_memory_frames = _nonnegative_int(cfg["n_memory_frames"], name="n_memory_frames")
        t_memory = _positive_int(cfg["t_memory"], name="t_memory")
    except KeyError as exc:
        raise ValueError(f"WEAVER checkpoint config is missing {exc.args[0]!r}") from exc
    if not image_keys:
        raise ValueError("dataset.img_keys must be non-empty")
    if len(set(image_keys)) != len(image_keys):
        raise ValueError(f"dataset.img_keys must be unique, got {image_keys}")
    return ArchitectureContract(
        image_keys=image_keys,
        image_size=image_size,
        n_states=n_states,
        n_actions=n_actions,
        spatial_size=spatial_size,
        horizon=horizon,
        n_history=n_history,
        n_memory_frames=n_memory_frames,
        t_memory=t_memory,
    )


def build_behavior_config(
    checkpoint_config: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    norm_stats_path: str | Path,
    image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
    n_states: int = BEHAVIOR_STATE_DIM,
    n_actions: int = BEHAVIOR_ACTION_DIM,
    dynamics_only: bool = True,
) -> tuple[dict[str, Any], BehaviorConfigReport]:
    """Override only the explicit BEHAVIOR transfer/data fields.

    All transformer, encoder, horizon, history, memory and inference settings
    remain inherited from the downloaded checkpoint.  The initial transfer is
    intentionally restricted to the released two-view order.  A three-view
    bimanual build is a separate architecture change and must be requested
    explicitly after the two-view identity/transfer gates pass.
    """

    source_cfg = checkpoint_defaults(checkpoint_config)
    source_contract = architecture_contract(source_cfg)
    target_keys = _string_tuple(image_keys, name="image_keys")
    if target_keys != tuple(DEFAULT_IMAGE_KEYS):
        raise ValueError(
            f"The first BEHAVIOR transfer must preserve released two-view order {DEFAULT_IMAGE_KEYS}, got {target_keys}"
        )
    target_state_dim = _positive_int(n_states, name="n_states")
    target_action_dim = _positive_int(n_actions, name="n_actions")
    if target_state_dim != BEHAVIOR_STATE_DIM or target_action_dim != BEHAVIOR_ACTION_DIM:
        raise ValueError(
            f"R1Pro transfer requires {BEHAVIOR_STATE_DIM}D state/action, got {target_state_dim}/{target_action_dim}"
        )

    dataset_path = Path(dataset_root).expanduser()
    norm_path = Path(norm_stats_path).expanduser()
    if not str(dataset_path):
        raise ValueError("dataset_root must be non-empty")
    if not str(norm_path):
        raise ValueError("norm_stats_path must be non-empty")

    target_cfg = copy.deepcopy(source_cfg)
    dataset = target_cfg.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("checkpoint dataset config must be a mapping")
    dataset.update(
        {
            "name": BEHAVIOR_DATASET_NAME,
            "path": str(dataset_path),
            "img_keys": list(target_keys),
            "n_states": target_state_dim,
            "n_actions": target_action_dim,
            "normalize": True,
            "norm_stats_path": str(norm_path),
            "relabel_actions": True,
            "state_representation": BEHAVIOR_STATE_REPRESENTATION,
            "action_representation": BEHAVIOR_ACTION_REPRESENTATION,
            "reward_mode": "disabled" if dynamics_only else "required",
            "collapse_prob": 0.0,
        }
    )
    training = target_cfg.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("checkpoint training config must be a mapping")
    training["dynamics_only"] = bool(dynamics_only)

    target_contract = architecture_contract(target_cfg)
    changed = tuple(
        field.name
        for field in dataclasses.fields(ArchitectureContract)
        if getattr(source_contract, field.name) != getattr(target_contract, field.name)
    )
    return target_cfg, BehaviorConfigReport(
        source=source_contract,
        target=target_contract,
        changed_fields=changed,
    )


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _positive_int_tuple(value: Any, *, length: int, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integers")
    try:
        result = tuple(_positive_int(item, name=f"{name}[]") for item in value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values, got {result}")
    return result


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{name} must contain non-empty strings, got {result}")
    return result
