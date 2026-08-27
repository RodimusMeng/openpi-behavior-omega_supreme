"""Shape-safe transfer of WEAVER PyTorch checkpoints across architectures.

PyTorch's ``strict=False`` still rejects a checkpoint tensor when its key is
present in the target model but its shape differs.  WEAVER's EMA checkpoint is
also positional rather than name-addressable, so it must not be copied after a
state/action dimension or view-layout change.  This module keeps both choices
explicit and testable without constructing WEAVER or loading real weights.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

import torch

DEFAULT_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


@dataclasses.dataclass(frozen=True)
class KeyRewrite:
    """A wrapper-prefixed state-dict key and its canonical spelling."""

    original: str
    canonical: str


@dataclasses.dataclass(frozen=True)
class ShapeMismatch:
    """A same-name source/target tensor pair that cannot be transferred."""

    key: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class TransferReport:
    """Structured result of a shape-filtered checkpoint transfer."""

    loaded_keys: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]
    missing_source_keys: tuple[str, ...]
    unexpected_source_keys: tuple[str, ...]
    source_key_rewrites: tuple[KeyRewrite, ...] = ()
    target_key_rewrites: tuple[KeyRewrite, ...] = ()

    @property
    def is_identity_compatible(self) -> bool:
        """Whether every source tensor maps one-to-one onto the target architecture."""

        return not (self.shape_mismatches or self.missing_source_keys or self.unexpected_source_keys)

    def assert_identity_compatible(self) -> None:
        """Raise if transfer would keep any target initialization or drop a source tensor."""

        if not self.is_identity_compatible:
            raise IdentityTransferError(self)

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded_keys)}, "
            f"shape_mismatches={len(self.shape_mismatches)}, "
            f"missing_source={len(self.missing_source_keys)}, "
            f"unexpected_source={len(self.unexpected_source_keys)}"
        )


class IdentityTransferError(ValueError):
    """Raised when an identity transfer was requested for incompatible architectures."""

    def __init__(self, report: TransferReport):
        self.report = report
        details = []
        if report.shape_mismatches:
            details.append("shape=" + ",".join(item.key for item in report.shape_mismatches))
        if report.missing_source_keys:
            details.append("missing=" + ",".join(report.missing_source_keys))
        if report.unexpected_source_keys:
            details.append("unexpected=" + ",".join(report.unexpected_source_keys))
        suffix = f" ({'; '.join(details)})" if details else ""
        super().__init__(f"Checkpoint transfer is not identity-compatible: {report.summary()}{suffix}")


class KeyNormalizationError(ValueError):
    """Raised when wrapper-prefix removal makes two state-dict keys ambiguous."""


@dataclasses.dataclass(frozen=True)
class EmaResetReport:
    """Named target parameters copied into a fresh positional EMA shadow list."""

    parameter_names: tuple[str, ...]
    shadow_shapes: tuple[tuple[int, ...], ...]


def normalize_state_dict_key(
    key: str,
    prefixes: Sequence[str] = DEFAULT_WRAPPER_PREFIXES,
) -> str:
    """Strip any repeated leading DDP/compile wrapper prefixes from ``key``."""

    if any(not prefix for prefix in prefixes):
        raise ValueError("State-dict wrapper prefixes must be non-empty")

    canonical = key
    while True:
        for prefix in prefixes:
            if canonical.startswith(prefix):
                canonical = canonical[len(prefix) :]
                break
        else:
            return canonical


def adapt_state_dict(
    source_state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
    *,
    prefixes: Sequence[str] = DEFAULT_WRAPPER_PREFIXES,
    require_identity: bool = False,
) -> tuple[collections.OrderedDict[str, torch.Tensor], TransferReport]:
    """Merge compatible source tensors into a target-initialized state dict.

    Keys are compared after removing repeated wrapper prefixes.  A tensor is
    copied only when both its canonical key and shape match.  The returned keys
    retain the target state's spelling and insertion order, so the result can be
    passed directly to that target model's ``load_state_dict``.

    Args:
        source_state_dict: Checkpoint model state.
        target_state_dict: Fresh target model state, used for both its schema and
            initialization of new or shape-incompatible tensors.
        prefixes: Leading wrapper prefixes to normalize before matching.
        require_identity: If true, reject any missing, unexpected, or
            shape-incompatible key.  This is the same-architecture regression
            guard: a successful result then consists entirely of source tensors.
    """

    source, source_rewrites = _canonicalize_state_dict(source_state_dict, prefixes, label="source")
    target, target_rewrites = _canonicalize_state_dict(target_state_dict, prefixes, label="target")

    merged: collections.OrderedDict[str, torch.Tensor] = collections.OrderedDict()
    loaded_keys = []
    shape_mismatches = []
    missing_source_keys = []

    for canonical_key, (target_key, target_tensor) in target.items():
        source_item = source.get(canonical_key)
        if source_item is None:
            merged[target_key] = target_tensor
            missing_source_keys.append(canonical_key)
            continue

        _, source_tensor = source_item
        if source_tensor.shape != target_tensor.shape:
            merged[target_key] = target_tensor
            shape_mismatches.append(
                ShapeMismatch(
                    key=canonical_key,
                    source_shape=tuple(source_tensor.shape),
                    target_shape=tuple(target_tensor.shape),
                )
            )
            continue

        merged[target_key] = source_tensor
        loaded_keys.append(canonical_key)

    unexpected_source_keys = [key for key in source if key not in target]
    report = TransferReport(
        loaded_keys=tuple(loaded_keys),
        shape_mismatches=tuple(shape_mismatches),
        missing_source_keys=tuple(missing_source_keys),
        unexpected_source_keys=tuple(unexpected_source_keys),
        source_key_rewrites=source_rewrites,
        target_key_rewrites=target_rewrites,
    )
    if require_identity:
        report.assert_identity_compatible()
    return merged, report


@torch.no_grad()
def reset_ema_from_model(ema_model: Any, model: torch.nn.Module) -> EmaResetReport:
    """Reset a Diffusers-style EMA object from the target model's trainable parameters.

    Pass WEAVER's inner ``model.ema.ema`` object, not the outer wrapper.  This
    deliberately does not consume a source EMA checkpoint: Diffusers stores EMA
    shadows positionally, so an architecture change can otherwise associate a
    valid-looking tensor with the wrong parameter.  The order here matches
    WEAVER's EMA construction (all and only ``requires_grad`` parameters).
    """

    if not hasattr(ema_model, "shadow_params"):
        raise TypeError("ema_model must expose a mutable shadow_params attribute")
    if not hasattr(ema_model, "optimization_step"):
        raise TypeError("ema_model must expose an optimization_step attribute")

    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    ema_model.shadow_params = [parameter.detach().clone() for _, parameter in named_parameters]
    ema_model.optimization_step = 0

    return EmaResetReport(
        parameter_names=tuple(name for name, _ in named_parameters),
        shadow_shapes=tuple(tuple(parameter.shape) for _, parameter in named_parameters),
    )


def _canonicalize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: Sequence[str],
    *,
    label: str,
) -> tuple[collections.OrderedDict[str, tuple[str, torch.Tensor]], tuple[KeyRewrite, ...]]:
    canonical: collections.OrderedDict[str, tuple[str, torch.Tensor]] = collections.OrderedDict()
    rewrites = []

    for original_key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} state_dict value for {original_key!r} is not a torch.Tensor")
        canonical_key = normalize_state_dict_key(original_key, prefixes)
        if canonical_key in canonical:
            previous_key = canonical[canonical_key][0]
            raise KeyNormalizationError(
                f"{label} keys {previous_key!r} and {original_key!r} both normalize to {canonical_key!r}"
            )
        canonical[canonical_key] = (original_key, tensor)
        if canonical_key != original_key:
            rewrites.append(KeyRewrite(original=original_key, canonical=canonical_key))

    return canonical, tuple(rewrites)
