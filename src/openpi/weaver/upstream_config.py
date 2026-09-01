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
# The released DROID WEAVER checkpoint is an 8D model. Some published
# checkpoint configs were materialized without these two metadata fields even
# though the tensors still have the 8D contract.
WEAVER_RELEASE_STATE_DIM = 8
WEAVER_RELEASE_ACTION_DIM = 8
WEAVER_RELEASE_IMAGE_KEYS = ("wrist_left", "exterior_1_left")
WEAVER_RELEASE_IMAGE_SIZE = (192, 320)
WEAVER_RELEASE_HORIZON = 8
WEAVER_RELEASE_N_HISTORY = 2
WEAVER_RELEASE_N_MEMORY_FRAMES = 6
WEAVER_RELEASE_T_MEMORY = 5
WEAVER_RELEASE_USE_PRECOMPUTED_FEATURES = True
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
    inferred_source_fields: tuple[str, ...] = ()

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


def released_checkpoint_config(raw_config: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return a validated concrete release config with omitted 8D metadata materialized.

    The published DROID release may omit ``dataset.n_states`` and
    ``dataset.n_actions``. Inference is allowed only when every other frozen
    release architecture field matches; arbitrary incomplete configs still
    fail closed.
    """

    concrete = checkpoint_defaults(raw_config)
    return _materialize_released_metadata(concrete)


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

    source_cfg, inferred_source_fields = released_checkpoint_config(checkpoint_config)
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
        inferred_source_fields=inferred_source_fields,
    )


def _materialize_released_metadata(
    config: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Materialize omitted runtime metadata from the known released WEAVER format.

    ``architecture_contract`` intentionally remains fail-closed. This helper
    is only used by the BEHAVIOR transfer builder, where the input is the
    published DROID release; the contract parser still validates every other
    shape-bearing field immediately afterwards.
    """

    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return config, ()
    missing_dimensions = tuple(name for name in ("n_states", "n_actions") if name not in dataset)
    missing_precomputed = ("use_precomputed_features",) if "use_precomputed_features" not in config else ()
    missing = (*missing_dimensions, *missing_precomputed)
    if not missing:
        if config["use_precomputed_features"] is not WEAVER_RELEASE_USE_PRECOMPUTED_FEATURES:
            raise ValueError("Frozen DROID release requires use_precomputed_features=True")
        return config, ()
    try:
        release_shape = (
            dataset.get("name") == "DROID",
            tuple(dataset["img_keys"]) == WEAVER_RELEASE_IMAGE_KEYS,
            tuple(dataset["image_size"]) == WEAVER_RELEASE_IMAGE_SIZE,
            config["horizon"] == WEAVER_RELEASE_HORIZON,
            config["n_history"] == WEAVER_RELEASE_N_HISTORY,
            config["n_memory_frames"] == WEAVER_RELEASE_N_MEMORY_FRAMES,
            config["t_memory"] == WEAVER_RELEASE_T_MEMORY,
        )
    except (KeyError, TypeError):
        release_shape = ()
    if release_shape != (True, True, True, True, True, True, True):
        raise ValueError(
            "Cannot infer omitted WEAVER state/action metadata: config does not "
            "match the frozen DROID release architecture"
        )
    for name in ("n_states", "n_actions"):
        expected_dimension = WEAVER_RELEASE_STATE_DIM if name == "n_states" else WEAVER_RELEASE_ACTION_DIM
        if name not in missing_dimensions and dataset[name] != expected_dimension:
            raise ValueError(
                f"Cannot infer omitted WEAVER release metadata: present dataset.{name}="
                f"{dataset[name]!r} is not the frozen 8D release value"
            )
    materialized = copy.deepcopy(config)
    materialized_dataset = materialized["dataset"]
    defaults = {
        "n_states": WEAVER_RELEASE_STATE_DIM,
        "n_actions": WEAVER_RELEASE_ACTION_DIM,
    }
    for name in missing:
        if name in defaults:
            materialized_dataset[name] = defaults[name]
    if "use_precomputed_features" in missing_precomputed:
        materialized["use_precomputed_features"] = WEAVER_RELEASE_USE_PRECOMPUTED_FEATURES
    elif config["use_precomputed_features"] is not WEAVER_RELEASE_USE_PRECOMPUTED_FEATURES:
        raise ValueError("Frozen DROID release requires use_precomputed_features=True")
    return materialized, missing


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
