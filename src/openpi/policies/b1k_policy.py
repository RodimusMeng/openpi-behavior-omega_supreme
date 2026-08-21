import dataclasses
from typing import Literal

import einops
import numpy as np

from openpi import transforms
from openpi.configs.robots.base_config import RobotConfig
from openpi.configs.robots.base_config import extract_compact_state
from openpi.models import model as _model

_MIN_DEPTH_M = 0.01
_MAX_DEPTH_M = 10.0


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/image_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(23),
        "prompt": "do something",
    }


def extract_state_from_proprio(proprio_data, robot_config: RobotConfig) -> np.ndarray:
    """Extract state from proprioception data based on robot configuration.

    We assume perfect correlation for the two gripper fingers.

    Args:
        proprio_data: Raw proprioception data
        robot_config: RobotConfig instance containing robot configuration

    Returns:
        Extracted state array
    """
    return extract_compact_state(proprio_data, robot_config)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def normalize_depth(
    depth,
    *,
    input_unit: Literal["m", "mm"],
    min_depth_m: float = _MIN_DEPTH_M,
    max_depth_m: float = _MAX_DEPTH_M,
) -> np.ndarray:
    """Convert metric depth to a channel-last float32 image in [-1, 1]."""
    if not 0 < min_depth_m < max_depth_m:
        raise ValueError(f"Expected 0 < min_depth_m < max_depth_m, got {min_depth_m}, {max_depth_m}")

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 2:
        depth = depth[..., None]
    elif depth.ndim >= 3 and depth.shape[-1] == 1:
        pass
    elif depth.ndim >= 3 and depth.shape[-3] == 1:
        depth = np.moveaxis(depth, -3, -1)
    else:
        raise ValueError(f"Expected depth shape [H,W], [H,W,1], or [...,1,H,W], got {depth.shape}")

    if input_unit == "mm":
        depth = depth / 1000.0
    elif input_unit != "m":
        raise ValueError(f"Unsupported depth unit: {input_unit!r}")

    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, max_depth_m)
    depth = np.clip(depth, min_depth_m, max_depth_m)

    log_min = np.log(min_depth_m)
    log_max = np.log(max_depth_m)
    return (2.0 * (np.log(depth) - log_min) / (log_max - log_min) - 1.0).astype(np.float32)


def prepare_rgbd(rgb, depth, *, depth_unit: Literal["m", "mm"]) -> np.ndarray:
    """Parse one paired RGB/depth observation into a channel-last RGBD tensor."""
    rgb = _parse_image(rgb).astype(np.float32) / 127.5 - 1.0
    depth = normalize_depth(depth, input_unit=depth_unit)
    if rgb.shape[:-1] != depth.shape[:-1]:
        raise ValueError(f"RGB and depth spatial shapes do not match: {rgb.shape} vs {depth.shape}")
    return np.concatenate([rgb, depth], axis=-1)


@dataclasses.dataclass(frozen=True)
class B1KInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Robot configuration object
    robot_config: RobotConfig = dataclasses.field(default=None)
    include_depth: bool = False
    depth_unit: Literal["m", "mm"] = "mm"

    def __call__(self, data: dict) -> dict:
        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data, self.robot_config)
        if "actions" in data:
            action = data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        images, image_masks = [], []
        for camera_id in self.robot_config.observations:
            rgb = data[f"observation/{camera_id}"]
            if self.include_depth:
                if not camera_id.startswith("image_"):
                    raise ValueError(f"Cannot infer paired depth key for camera {camera_id!r}")
                depth_id = camera_id.replace("image_", "depth_", 1)
                images.append(
                    prepare_rgbd(
                        rgb,
                        data[f"observation/{depth_id}"],
                        depth_unit=self.depth_unit,
                    )
                )
            else:
                images.append(_parse_image(rgb))
            image_masks.append(np.True_)
        while len(images) < 3:
            images.append(np.zeros_like(images[0]))
            image_masks.append(np.False_)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = tuple(images)
                image_masks = tuple(image_masks)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = tuple(images)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class B1KOutputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
