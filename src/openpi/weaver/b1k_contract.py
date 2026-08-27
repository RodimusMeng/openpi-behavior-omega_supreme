"""Audited R1Pro data contract at the OpenPI/WEAVER boundary."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import math

import numpy as np

from openpi.configs.robots import ROBOT_REGISTRY
from openpi.configs.robots.base_config import RobotConfig
from openpi.configs.robots.base_config import build_delta_action_mappings
from openpi.configs.robots.base_config import extract_compact_state
from openpi.configs.robots.base_config import state_output_slices

R1PRO_ROBOT_KEY = "b1k/R1Pro"

# Keep the released two-view projector names for the first transfer smoke test.
# A third view changes additional reward/critic tensors, not only one projector.
RELEASED_TWO_VIEW_MAPPING: tuple[tuple[str, str], ...] = (
    ("wrist_left", "image_1"),
    ("exterior_1_left", "image_0"),
)


@dataclasses.dataclass(frozen=True)
class StateComponent:
    name: str
    raw_indices: tuple[int, ...]
    compact_indices: tuple[int, ...]
    reduction: str


@dataclasses.dataclass(frozen=True)
class ActionComponent:
    name: str
    indices: tuple[int, ...]
    environment_semantics: str
    openpi_training_semantics: str
    compact_state_indices: tuple[int, ...] | None


@dataclasses.dataclass(frozen=True)
class CameraView:
    weaver_key: str
    robot_camera_key: str
    camera_name: str
    online_rgb_key: str
    dataset_rgb_key: str


@dataclasses.dataclass(frozen=True)
class B1KWeaverContract:
    """One source-derived description of the R1Pro/WEAVER boundary.

    ``environment actions`` are the post-transform chunks returned by
    ``Policy.infer`` and consumed by the B1K controller. They intentionally keep
    their mixed command semantics. ``OpenPI training actions`` are a separate
    internal representation in which only torso and arm targets are converted to
    deltas from the current compact state.
    """

    robot_key: str
    robot: RobotConfig
    state_components: tuple[StateComponent, ...]
    action_components: tuple[ActionComponent, ...]
    camera_views: tuple[CameraView, ...]
    delta_mappings: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]

    @property
    def state_dim(self) -> int:
        return sum(len(component.compact_indices) for component in self.state_components)

    @property
    def action_dim(self) -> int:
        return self.robot.action_dim

    def extract_state(self, raw_proprio: np.ndarray) -> np.ndarray:
        """Reuse the B1K policy transform to produce the audited compact state."""
        state = extract_compact_state(np.asarray(raw_proprio), self.robot)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected compact state dim {self.state_dim}, got {state.shape}")
        return state

    def extract_online_views(self, observation: Mapping[str, object]) -> dict[str, np.ndarray]:
        """Map raw evaluator RGB keys to the released WEAVER two-view names."""
        views: dict[str, np.ndarray] = {}
        for view in self.camera_views:
            if view.online_rgb_key not in observation:
                raise KeyError(f"Missing online RGB observation: {view.online_rgb_key}")
            views[view.weaver_key] = np.asarray(observation[view.online_rgb_key])
        return views

    def extract_dataset_views(self, sample: Mapping[str, object]) -> dict[str, np.ndarray]:
        """Map LeRobot dataset RGB keys to the same WEAVER view names."""
        views: dict[str, np.ndarray] = {}
        for view in self.camera_views:
            if view.dataset_rgb_key not in sample:
                raise KeyError(f"Missing dataset RGB observation: {view.dataset_rgb_key}")
            views[view.weaver_key] = np.asarray(sample[view.dataset_rgb_key])
        return views

    def validate_environment_actions(self, actions: np.ndarray) -> np.ndarray:
        """Validate post-transform B1K chunks without changing their semantics."""
        actions = np.asarray(actions)
        if actions.ndim < 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected post-transform actions with shape (...,T,{self.action_dim}), got {actions.shape}"
            )
        if not np.issubdtype(actions.dtype, np.number) or not np.isfinite(actions).all():
            raise ValueError("Environment actions must be finite numeric values")
        return np.array(actions, copy=True)

    def _expand_state_slice(
        self,
        current_state: np.ndarray,
        *,
        action_rank: int,
        state_indices: tuple[int, ...],
    ) -> np.ndarray:
        state = np.asarray(current_state)
        if state.ndim < 1 or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected current state with shape (...,{self.state_dim}), got {state.shape}")
        if not np.issubdtype(state.dtype, np.number) or not np.isfinite(state).all():
            raise ValueError("Current state must contain finite numeric values")

        inserted_axes = action_rank - state.ndim
        if inserted_axes < 1:
            raise ValueError(
                "Environment actions must have at least one candidate/time axis beyond current state: "
                f"action rank {action_rank}, state rank {state.ndim}"
            )
        selected = state[..., state_indices]
        return selected.reshape(*selected.shape[:-1], *((1,) * inserted_axes), selected.shape[-1])

    def environment_to_openpi_training_actions(
        self,
        actions: np.ndarray,
        current_state: np.ndarray,
    ) -> np.ndarray:
        """Apply the exact selective delta transform used by B1K policy training."""
        transformed = self.validate_environment_actions(actions)
        for action_indices, state_indices in self.delta_mappings:
            transformed[..., action_indices] -= self._expand_state_slice(
                current_state,
                action_rank=transformed.ndim,
                state_indices=state_indices,
            )
        return transformed

    def openpi_training_to_environment_actions(
        self,
        actions: np.ndarray,
        current_state: np.ndarray,
    ) -> np.ndarray:
        """Invert the selective delta transform before controller execution."""
        actions = np.asarray(actions)
        if actions.ndim < 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected OpenPI actions with shape (...,T,{self.action_dim}), got {actions.shape}")
        transformed = np.array(actions, copy=True)
        for action_indices, state_indices in self.delta_mappings:
            transformed[..., action_indices] += self._expand_state_slice(
                current_state,
                action_rank=transformed.ndim,
                state_indices=state_indices,
            )
        return transformed


def build_r1pro_contract(
    view_mapping: tuple[tuple[str, str], ...] = RELEASED_TWO_VIEW_MAPPING,
) -> B1KWeaverContract:
    """Build the contract from the registered R1Pro config, not copied indices."""
    robot = ROBOT_REGISTRY[R1PRO_ROBOT_KEY]
    compact_slices = state_output_slices(robot)
    state_components = tuple(
        StateComponent(
            name=config.name,
            raw_indices=tuple(config.indices),
            compact_indices=compact_slice,
            reduction="sum" if config.is_eef else "identity",
        )
        for config, compact_slice in zip(robot.proprio, compact_slices, strict=True)
    )

    delta_mappings = tuple(build_delta_action_mappings(robot))
    state_by_action = dict(delta_mappings)
    action_components: list[ActionComponent] = []
    for config in robot.action:
        indices = tuple(config.indices)
        compact_state_indices = state_by_action.get(indices)
        if config.name == "base":
            environment_semantics = "robot_frame_velocity_command"
        elif config.is_eef:
            environment_semantics = "gripper_command"
        else:
            environment_semantics = "absolute_joint_position_target"
        action_components.append(
            ActionComponent(
                name=config.name,
                indices=indices,
                environment_semantics=environment_semantics,
                openpi_training_semantics=(
                    "delta_from_current_compact_state" if compact_state_indices is not None else environment_semantics
                ),
                compact_state_indices=compact_state_indices,
            )
        )

    camera_views: list[CameraView] = []
    seen_weaver_keys: set[str] = set()
    seen_robot_keys: set[str] = set()
    for weaver_key, robot_camera_key in view_mapping:
        if weaver_key in seen_weaver_keys:
            raise ValueError(f"Duplicate WEAVER view key: {weaver_key}")
        if robot_camera_key in seen_robot_keys:
            raise ValueError(f"Robot camera mapped more than once: {robot_camera_key}")
        if robot_camera_key not in robot.observations:
            raise KeyError(f"Unknown R1Pro camera key: {robot_camera_key}")
        seen_weaver_keys.add(weaver_key)
        seen_robot_keys.add(robot_camera_key)
        observation = robot.observations[robot_camera_key]
        camera_views.append(
            CameraView(
                weaver_key=weaver_key,
                robot_camera_key=robot_camera_key,
                camera_name=observation.name,
                online_rgb_key=observation.obs_key,
                dataset_rgb_key=observation.dataset_key,
            )
        )

    return B1KWeaverContract(
        robot_key=R1PRO_ROBOT_KEY,
        robot=robot,
        state_components=state_components,
        action_components=tuple(action_components),
        camera_views=tuple(camera_views),
        delta_mappings=delta_mappings,
    )


def world_model_frame_indices(timestamps_s: np.ndarray, *, target_hz: float = 5.0) -> np.ndarray:
    """Choose nearest real frames on a regular world-model time grid.

    This avoids baking WEAVER's Franka-specific ``RGB_SKIP=3`` into B1K. At the
    evaluator's nominal 30 Hz, a 5 Hz grid normally selects every sixth frame,
    but real timestamps remain the source of truth.
    """
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError(f"timestamps_s must be a non-empty vector, got {timestamps.shape}")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    if not math.isfinite(target_hz) or target_hz <= 0:
        raise ValueError(f"target_hz must be positive and finite, got {target_hz}")

    target_times = np.arange(timestamps[0], timestamps[-1] + 1e-12, 1.0 / target_hz)
    right = np.searchsorted(timestamps, target_times, side="left")
    right = np.clip(right, 0, timestamps.size - 1)
    left = np.clip(right - 1, 0, timestamps.size - 1)
    choose_left = np.abs(target_times - timestamps[left]) <= np.abs(timestamps[right] - target_times)
    indices = np.where(choose_left, left, right)
    return np.unique(indices).astype(np.int64)


def future_state_placeholders(current_state: np.ndarray, horizon: int) -> np.ndarray:
    """Create only the future-state shape required by WEAVER rollout inference.

    At the frozen upstream implementation, future values are replaced by random
    noise before denoising; callers must not treat these zeros as hard conditions
    or dynamics predictions.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    state = np.asarray(current_state)
    if state.ndim < 1 or state.shape[-1] == 0:
        raise ValueError(f"current_state must have shape (...,D), got {state.shape}")
    return np.zeros((*state.shape[:-1], horizon, state.shape[-1]), dtype=state.dtype)
