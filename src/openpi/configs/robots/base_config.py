"""Base configuration dataclasses for robot configs."""

from dataclasses import dataclass

import numpy as np

# Global registry for robot configurations
ROBOT_REGISTRY: dict[str, "RobotConfig"] = {}


def register_robot(name: str, config: "RobotConfig") -> "RobotConfig":
    """Register a robot config in the global registry.

    Args:
        name: Robot identifier in format 'bucket/robot_type' (e.g., 'i3l/A1')
        config: RobotConfig instance to register

    Returns:
        The same config instance (for chaining)
    """
    ROBOT_REGISTRY[name] = config
    return config


@dataclass
class ObservationConfig:
    """Configuration for a single camera observation view."""

    name: str  # unique identifier for the observation
    obs_key: str  # full name in the obsevation dict during inference
    dataset_key: str  # this is the name of the key in LeRobot datasets
    resolution: list[int]  # [width, height]

    # RGBD optional fields
    depth_obs_key: str | None = None
    depth_dataset_key: str | None = None


@dataclass
class StateActionConfig:
    """Configuration for action / proprioception data."""

    name: str  # Name for this action/proprio component
    indices: list[int] | None = None  # Indices in the full action/proprio array
    is_eef: bool = False  # Whether this corresponds to an end-effector command (gripper, dexhand, etc.)
    needs_delta_comp: bool = False  # [Action Only] Whether to compute delta from previous step


@dataclass
class RobotConfig:
    """Base configuration for a robot.

    This matches the structure defined in i3l.yaml robot configurations.
    """

    name: str
    robot_type: str
    observations: dict[str, ObservationConfig]
    action_key: str
    action_dim: int
    action: list[StateActionConfig]
    proprio: list[StateActionConfig]

    def __post_init__(self):
        """Convert dictionaries to proper dataclass instances if needed."""
        # Convert observation configs if they're still dicts
        if self.observations:
            self.observations = {
                key: ObservationConfig(**val) if isinstance(val, dict) else val
                for key, val in self.observations.items()
            }

        # Convert action and proprio configs if they're still dicts
        self.proprio = [StateActionConfig(**item) if isinstance(item, dict) else item for item in self.proprio]
        self.action = [StateActionConfig(**item) if isinstance(item, dict) else item for item in self.action]

        # Validate action_dim matches sum of action indices
        total_dim = sum([len(a.indices) for a in self.action])
        if total_dim != self.action_dim:
            raise ValueError(f"Sum of action indices ({total_dim}) doesn't match " f"action_dim ({self.action_dim})")


def state_output_slices(robot_config: RobotConfig) -> list[tuple[int, ...]]:
    """Return each configured proprio group in the compact policy-state space.

    ``RobotConfig.proprio`` stores indices into the raw simulator proprio vector.
    End-effectors are reduced to one scalar by the robot policy transform, so the
    compact output slices cannot be read directly from those raw indices.
    """
    slices: list[tuple[int, ...]] = []
    offset = 0
    for proprio_config in robot_config.proprio:
        width = 1 if proprio_config.is_eef else len(proprio_config.indices)
        slices.append(tuple(range(offset, offset + width)))
        offset += width
    return slices


def extract_compact_state(raw_proprio: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """Extract the compact policy state described by ``RobotConfig.proprio``."""
    raw_proprio = np.asarray(raw_proprio)
    components = []
    for proprio_config in robot_config.proprio:
        component = raw_proprio[..., proprio_config.indices]
        if proprio_config.is_eef:
            component = component.sum(axis=-1, keepdims=True)
        components.append(component)
    return np.concatenate(components, axis=-1)


def build_delta_action_mappings(robot_config: RobotConfig) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Build the action/state mappings used by OpenPI's selective delta transform.

    Only action groups marked ``needs_delta_comp`` participate. The matching
    behavior intentionally preserves the existing B1K transform contract: groups
    are matched to compact state slices in order and by width.
    """
    state_slices = state_output_slices(robot_config)
    mappings: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    state_slice_index = 0
    for action_config in robot_config.action:
        if action_config.is_eef or not action_config.needs_delta_comp:
            continue

        action_indices = tuple(action_config.indices)
        while state_slice_index < len(state_slices) and len(state_slices[state_slice_index]) != len(action_indices):
            state_slice_index += 1
        if state_slice_index >= len(state_slices):
            raise ValueError(
                f"Could not find a state slice for delta action group {action_config.name!r} "
                f"with dim {len(action_indices)} in robot config {robot_config.robot_type!r}."
            )
        mappings.append((action_indices, state_slices[state_slice_index]))
        state_slice_index += 1
    return mappings
