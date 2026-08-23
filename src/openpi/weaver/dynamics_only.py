"""Reward-free WEAVER flow training that reuses the upstream model internals.

The released ``WEAVER.forward`` always constructs reward/critic inputs and
requires reward labels, even when their coefficients are zero.  This adapter
executes the upstream dynamics path directly and never calls ``task_encoder``,
``rm`` or ``critic``.  It is intentionally an external adapter: the frozen
upstream checkout remains an identity-testable reference.
"""

# This adapter deliberately mirrors frozen-upstream WEAVER internals.  The
# upstream class exposes the required training contract only through these
# attributes, so each access is audited against the pinned upstream commit.
# ruff: noqa: SLF001

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openpi.weaver.dynamics_contract import prediction_alignment


def dynamics_only_forward(
    model: Any,
    *,
    obs: Mapping[str, Any],
    actions: Any,
    transition_valid: Any,
    memory: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run one WEAVER flow-loss forward without reward or critic computation.

    Args:
        model: An upstream ``weaver.wm.model.WEAVER`` instance.
        obs: Batched precomputed views plus ``states``.
        actions: Batched normalized actions with shape ``(B,T,n_actions)``.
        transition_valid: Boolean ``(B,T)`` mask. Target frame ``j`` consumes
            the validity of the transition at ``j-1``.
        memory: Optional sparse real-observation memory batch.
    """

    torch = _require_torch()
    if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
        raise ValueError(f"actions must be a rank-3 torch tensor, got {type(actions)!r}")
    batch_size, sequence_length, _ = actions.shape
    alignment = prediction_alignment(sequence_length, int(model._n_history))
    _validate_obs(obs, batch_size=batch_size, sequence_length=sequence_length, torch=torch)
    target_mask = _target_mask(
        transition_valid,
        batch_size=batch_size,
        sequence_length=sequence_length,
        alignment=alignment,
        device=actions.device,
        torch=torch,
    )

    memory_tokens = None
    if model._use_memory:
        if memory is None:
            raise ValueError("model uses sparse memory but memory=None was passed")
        _validate_memory(memory, batch_size=batch_size, model=model, torch=torch)
        memory_tokens = model.encode_memory_obs(memory)
        if model.training and model._history_noise_std > 0:
            memory_tokens = memory_tokens + torch.randn_like(memory_tokens) * model._history_noise_std
    elif memory is not None:
        raise ValueError("memory was provided but the target WEAVER model has memory disabled")

    x1 = model.encode_obs(obs)
    x0 = model.sample_noise(x1)
    timestep = model.sample_timestep(batch_size, sequence_length)
    xt = model.interpolate(x1, x0, timestep)
    if model.training and model._history_noise_std > 0:
        for key, value in tuple(xt.items()):
            history = value[:, : model._n_history]
            future = value[:, model._n_history :]
            noised_history = history + torch.randn_like(history) * model._history_noise_std
            xt[key] = torch.cat([noised_history, future], dim=1)

    predicted = model.wm(xt, actions, timestep, memory=memory_tokens)
    flow_losses: dict[str, Any] = {}
    for key, prediction in predicted.items():
        if key not in x1 or key not in x0:
            raise ValueError(f"upstream dynamics returned unknown prediction key {key!r}")
        expanded_timestep = _expanded_timestep(model, key, timestep)
        elementwise = model.compute_flow_loss(x1[key], x0[key], prediction, expanded_timestep)
        per_timestep = _mean_feature_dims(elementwise)
        flow_losses[key] = _masked_mean(
            per_timestep[:, alignment.future_target_slice],
            target_mask,
            name=f"flow/{key}",
        )

    expected_keys = {*model._img_keys, "states"}
    if set(flow_losses) != expected_keys:
        raise ValueError(
            f"dynamics prediction keys must be exactly {sorted(expected_keys)}, got {sorted(flow_losses)}"
        )
    image_flow = sum(flow_losses[key] for key in model._img_keys)
    total_flow = image_flow + model._state_loss_scale * flow_losses["states"]
    total_loss = total_flow

    temporal_losses: dict[str, Any] = {}
    temporal_total = total_flow.new_zeros(())
    if model._use_temporal_loss:
        for key, prediction in predicted.items():
            elementwise = model.compute_temporal_loss(x1[key], x0[key], prediction)
            per_pair = _mean_feature_dims(elementwise)
            temporal_losses[key] = _masked_mean(
                per_pair[:, alignment.temporal_pair_slice],
                target_mask,
                name=f"temporal/{key}",
            )
        temporal_total = sum(temporal_losses.values())
        total_loss = total_loss + model._temporal_loss_coeff * temporal_total

    if model._train_decoder:
        raise NotImplementedError(
            "The first BEHAVIOR dynamics-only transfer requires the released frozen SD3 decoder path"
        )

    zero = total_loss.new_zeros(())
    log = {f"flow/{key}": value.detach() for key, value in flow_losses.items()}
    log.update({f"temporal/{key}": value.detach() for key, value in temporal_losses.items()})
    log.update(
        {
            "flow/Total Loss": total_flow.detach(),
            "RM loss": zero.detach(),
            "Critic Loss": zero.detach(),
            "decoder/Total Loss": zero.detach(),
            "Temporal/Total Loss": temporal_total.detach(),
            "Total Loss": total_loss.detach(),
            "valid_future_targets": target_mask.sum().detach(),
        }
    )
    return total_loss, log


def dynamics_optimizer_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    """Return trainable parameters excluding heads unused by dynamics-only loss."""

    excluded_prefixes = ("rm.", "critic.", "task_encoder.")
    selected = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(excluded_prefixes)
    )
    if not selected:
        raise ValueError("No trainable WEAVER dynamics parameters were selected")
    return selected


def _target_mask(
    transition_valid: Any,
    *,
    batch_size: int,
    sequence_length: int,
    alignment: Any,
    device: Any,
    torch: Any,
) -> Any:
    if not isinstance(transition_valid, torch.Tensor):
        raise TypeError("transition_valid must be a torch tensor after DataLoader collation")
    if transition_valid.shape != (batch_size, sequence_length):
        raise ValueError(
            f"transition_valid must have shape {(batch_size, sequence_length)}, "
            f"got {tuple(transition_valid.shape)}"
        )
    if transition_valid.dtype is not torch.bool:
        raise TypeError(f"transition_valid must be torch.bool, got {transition_valid.dtype}")
    mask = transition_valid[:, alignment.predecessor_transition_slice].to(device=device)
    if not bool(mask.any()):
        raise ValueError("dynamics batch contains no valid future prediction targets")
    return mask


def _validate_obs(obs: Mapping[str, Any], *, batch_size: int, sequence_length: int, torch: Any) -> None:
    if not isinstance(obs, Mapping) or "states" not in obs:
        raise ValueError("obs must be a mapping containing states")
    for key, value in obs.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"obs[{key!r}] must be a torch tensor, got {type(value)!r}")
        if value.shape[:2] != (batch_size, sequence_length):
            raise ValueError(
                f"obs[{key!r}] must start with {(batch_size, sequence_length)}, got {tuple(value.shape)}"
            )


def _validate_memory(memory: Mapping[str, Any], *, batch_size: int, model: Any, torch: Any) -> None:
    if not isinstance(memory, Mapping) or "states" not in memory:
        raise ValueError("memory must be a mapping containing states")
    for key, value in memory.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"memory[{key!r}] must be a torch tensor, got {type(value)!r}")
        if value.shape[:2] != (batch_size, model._n_memory_frames):
            raise ValueError(
                f"memory[{key!r}] must start with {(batch_size, model._n_memory_frames)}, "
                f"got {tuple(value.shape)}"
            )


def _expanded_timestep(model: Any, key: str, timestep: Any) -> Any:
    if key in model._img_keys:
        return timestep[:, :, None, None]
    return timestep[:, :, None]


def _mean_feature_dims(value: Any) -> Any:
    if value.ndim < 3:
        raise ValueError(f"elementwise dynamics loss must have feature dimensions, got {tuple(value.shape)}")
    dims = tuple(range(2, value.ndim))
    return value.mean(dim=dims)


def _masked_mean(value: Any, mask: Any, *, name: str) -> Any:
    if value.shape != mask.shape:
        raise ValueError(f"{name} value/mask shape mismatch: {tuple(value.shape)} vs {tuple(mask.shape)}")
    weights = mask.to(dtype=value.dtype)
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise ValueError(f"{name} has no valid entries")
    return (value * weights).sum() / denominator


def _require_torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "dynamics_only_forward requires the official WEAVER/OpenPI PyTorch environment"
        ) from exc
    return torch
