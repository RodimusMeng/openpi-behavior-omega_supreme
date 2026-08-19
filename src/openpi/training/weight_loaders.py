import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads checkpoint weights, optionally expanding the RGB stem to RGBD.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    adapt_rgbd_stem: bool = False

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path),
            restore_type=np.ndarray,
        )

        if self.adapt_rgbd_stem:
            loaded_params = _adapt_rgbd_stem(
                loaded_params,
                params,
            )

        return _merge_params(
            loaded_params,
            params,
            missing_regex=".*lora.*",
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _adapt_rgbd_stem(
    loaded_params: at.Params,
    reference_params: at.Params,
) -> at.Params:
    """Expand the SigLIP input stem from RGB to RGBD."""

    stem_key = "PaliGemma/img/embedding/kernel"

    flat_loaded = flax.traverse_util.flatten_dict(
        loaded_params,
        sep="/",
    )
    flat_reference = flax.traverse_util.flatten_dict(
        reference_params,
        sep="/",
    )

    if stem_key not in flat_loaded:
        raise KeyError(f"Checkpoint has no {stem_key}")
    if stem_key not in flat_reference:
        raise KeyError(f"Model has no {stem_key}")

    loaded_stem = flat_loaded[stem_key]
    reference_stem = flat_reference[stem_key]

    if loaded_stem.shape == reference_stem.shape:
        return loaded_params

    if loaded_stem.ndim != 4 or reference_stem.ndim != 4:
        raise ValueError(f"Cannot adapt SigLIP stem from {loaded_stem.shape} to {reference_stem.shape}")

    expected_shape = (*loaded_stem.shape[:-2], 4, loaded_stem.shape[-1])
    if loaded_stem.shape[-2] != 3 or reference_stem.shape != expected_shape:
        raise ValueError(f"Cannot adapt SigLIP stem from {loaded_stem.shape} to {reference_stem.shape}")

    rgbd_stem = np.zeros(
        reference_stem.shape,
        dtype=loaded_stem.dtype,
    )
    rgbd_stem[..., :3, :] = loaded_stem
    flat_loaded[stem_key] = rgbd_stem

    return flax.traverse_util.unflatten_dict(
        flat_loaded,
        sep="/",
    )


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            if v.dtype == flat_ref[k].dtype:
                result[k] = v
            else:
                print(f"Warning: {k} has dtype {v.dtype} but reference has dtype {flat_ref[k].dtype}")
                result[k] = v.astype(flat_ref[k].dtype)
    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
