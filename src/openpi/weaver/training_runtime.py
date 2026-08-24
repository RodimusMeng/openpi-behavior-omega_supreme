"""Single-process training runtime for BEHAVIOR-adapted WEAVER models.

The runtime deliberately keeps model construction and checkpoint transfer behind
injected callables.  Those operations belong to the frozen-upstream adapter;
this module owns only the shared ``full``/``dynamics`` training semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from contextlib import nullcontext
import dataclasses
import inspect
from pathlib import Path
import time
from typing import Any, Literal

from openpi.weaver import dynamics_only

TrainingMode = Literal["full", "dynamics"]


@dataclasses.dataclass(frozen=True)
class UpstreamTools:
    """Official WEAVER utility functions used by the training loop."""

    move_tensors_to_device: Callable[..., Mapping[str, Any]]
    save_checkpoint: Callable[..., None]
    get_lr: Callable[..., float]


@dataclasses.dataclass(frozen=True)
class LoopConfig:
    """Resolved optimizer/loop settings, after applying upstream overrides."""

    mode: TrainingMode
    device: str
    max_steps: int
    gradient_accumulation_steps: int
    max_lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float
    betas: tuple[float, float]
    grad_clip_norm: float = 1.0
    checkpoint_every: int = 0
    tiny_overfit_batches: int = 0
    tiny_overfit_steps: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"full", "dynamics"}:
            raise ValueError(f"Unsupported WEAVER training mode: {self.mode!r}")
        for name in ("max_steps", "gradient_accumulation_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("warmup_steps", "checkpoint_every", "tiny_overfit_batches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if self.tiny_overfit_steps is not None and (
            isinstance(self.tiny_overfit_steps, bool)
            or not isinstance(self.tiny_overfit_steps, int)
            or self.tiny_overfit_steps <= 0
        ):
            raise ValueError("tiny_overfit_steps must be a positive integer when supplied")
        if self.max_lr <= 0 or self.min_lr < 0 or self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("learning rates, weight decay, and grad_clip_norm are outside their valid range")
        if len(self.betas) != 2 or any(not 0 <= beta < 1 for beta in self.betas):
            raise ValueError(f"betas must contain two values in [0,1), got {self.betas}")


@dataclasses.dataclass(frozen=True)
class StepReport:
    completed_step: int
    loss: float
    learning_rate: float
    grad_norm: float
    losses: dict[str, float]


@dataclasses.dataclass(frozen=True)
class TrainResult:
    start_step: int
    completed_step: int
    reports: tuple[StepReport, ...]
    telemetry: dict[str, Any]


def load_upstream_tools() -> UpstreamTools:
    """Import the official helpers lazily inside the frozen WEAVER environment."""

    try:
        from weaver.utils.tools import get_lr  # noqa: PLC0415
        from weaver.utils.tools import move_tensors_to_device  # noqa: PLC0415
        from weaver.utils.tools import save_checkpoint  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Official WEAVER utilities are unavailable; add the frozen WEAVER checkout to PYTHONPATH"
        ) from exc
    return UpstreamTools(
        move_tensors_to_device=move_tensors_to_device,
        save_checkpoint=save_checkpoint,
        get_lr=get_lr,
    )


def build_optimizer(
    model: Any,
    *,
    mode: TrainingMode,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device: str,
) -> Any:
    """Build the official optimizer while excluding unused heads in dynamics mode."""

    torch = _require_torch()
    if mode == "full":
        if not hasattr(model, "configure_optimizers"):
            raise TypeError("A full WEAVER model must expose configure_optimizers")
        return model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=torch.device(device).type,
        )

    selected = dynamics_only.dynamics_optimizer_parameters(model)
    selected_ids = {id(parameter) for _, parameter in selected}
    if hasattr(model, "configure_optimizers"):
        # Reuse upstream's exact AdamW grouping/fused selection.  The temporary
        # flags affect construction only; the external dynamics forward already
        # guarantees that excluded heads receive no graph or gradient.
        changed = []
        for _, parameter in model.named_parameters():
            if parameter.requires_grad and id(parameter) not in selected_ids:
                changed.append(parameter)
                parameter.requires_grad_(requires_grad=False)
        try:
            optimizer = model.configure_optimizers(
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                betas=betas,
                device_type=torch.device(device).type,
            )
        finally:
            for parameter in changed:
                parameter.requires_grad_(requires_grad=True)
    else:
        optimizer = _equivalent_upstream_adamw(
            tuple(parameter for _, parameter in selected),
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=torch.device(device).type,
            torch=torch,
        )

    actual_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual_ids != selected_ids:
        raise ValueError("Dynamics optimizer parameters do not exactly match dynamics_optimizer_parameters")
    return optimizer


def restore_optimizer_for_resume(optimizer: Any, checkpoint: MutableMapping[str, Any]) -> int:
    """Restore optimizer/step and consume the large CPU checkpoint mapping."""

    if "optimizer" not in checkpoint or "step" not in checkpoint:
        raise ValueError("Resume checkpoint must contain optimizer and step")
    step = checkpoint["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"Resume checkpoint step must be a non-negative integer, got {step!r}")
    optimizer_state = checkpoint.pop("optimizer")
    checkpoint.pop("step")
    try:
        optimizer.load_state_dict(optimizer_state)
    finally:
        del optimizer_state
        checkpoint.clear()
    return step


def compute_loss(model: Any, batch: Mapping[str, Any], *, mode: TrainingMode) -> tuple[Any, dict[str, Any]]:
    """Execute the selected real WEAVER loss path on one device-resident batch."""

    if not isinstance(batch, Mapping):
        raise TypeError("WEAVER batch must be a mapping")
    for key in ("obs", "actions", "transition_valid"):
        if key not in batch:
            raise ValueError(f"WEAVER batch is missing required field {key!r}")

    if mode == "dynamics":
        return dynamics_only.dynamics_only_forward(
            model,
            obs=batch["obs"],
            actions=batch["actions"],
            transition_valid=batch["transition_valid"],
            memory=batch.get("memory"),
        )
    if mode != "full":
        raise ValueError(f"Unsupported WEAVER training mode: {mode!r}")

    _validate_full_batch(model, batch)
    # This is intentionally the frozen upstream forward, including its real
    # reward and critic losses.  No zero-filled reward compatibility path exists.
    return model.forward(
        obs=batch["obs"],
        actions=batch["actions"],
        tasks=batch["task"],
        gt_rewards=batch["rewards"],
        memory=batch.get("memory"),
    )


def train(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: Any,
    *,
    loop: LoopConfig,
    upstream: UpstreamTools,
    checkpoint_dir: str | Path | None,
    saved_config: Mapping[str, Any],
    start_step: int = 0,
) -> TrainResult:
    """Run real forward/backward updates and atomically checkpoint progress."""

    torch = _require_torch()
    if isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0:
        raise ValueError(f"start_step must be a non-negative integer, got {start_step!r}")
    if start_step > loop.max_steps:
        raise ValueError(f"start_step {start_step} exceeds max_steps {loop.max_steps}")
    raw_model = _unwrap_model(model)
    if not hasattr(raw_model, "ema") or not callable(getattr(raw_model.ema, "update", None)):
        raise TypeError("WEAVER model must expose ema.update()")

    stop_step = loop.max_steps
    if loop.tiny_overfit_steps is not None:
        stop_step = min(stop_step, start_step + loop.tiny_overfit_steps)
    batches = _training_batches(dataloader, cache_count=loop.tiny_overfit_batches)
    optimizer_parameters = _unique_optimizer_parameters(optimizer)
    if not optimizer_parameters:
        raise ValueError("Optimizer contains no parameters")

    model.train()
    telemetry_state = _start_telemetry(loop.device, torch=torch)
    reports: list[StepReport] = []
    completed_step = start_step
    last_saved_step: int | None = None

    for step in range(start_step, stop_step):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        last_log: Mapping[str, Any] = {}
        for _ in range(loop.gradient_accumulation_steps):
            batch = next(batches)
            device_batch = upstream.move_tensors_to_device(batch, device=loop.device)
            with _autocast_context(loop.device, torch=torch):
                loss, last_log = compute_loss(model, device_batch, mode=loop.mode)
                scaled_loss = loss / loop.gradient_accumulation_steps
            if not bool(torch.isfinite(scaled_loss).all()):
                raise FloatingPointError(f"Non-finite {loop.mode} loss at optimizer step {step}")
            accumulated_loss += float(scaled_loss.detach().item())
            scaled_loss.backward()

        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(optimizer_parameters, loop.grad_clip_norm)
        learning_rate = float(
            upstream.get_lr(
                step,
                loop.warmup_steps,
                loop.max_steps,
                loop.max_lr,
                loop.min_lr,
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        raw_model.ema.update()
        completed_step = step + 1
        reports.append(
            StepReport(
                completed_step=completed_step,
                loss=accumulated_loss,
                learning_rate=learning_rate,
                grad_norm=float(grad_norm_tensor.detach().item()),
                losses=_detached_scalars(last_log),
            )
        )

        if (
            checkpoint_dir is not None
            and loop.checkpoint_every > 0
            and completed_step % loop.checkpoint_every == 0
        ):
            _save_atomic(
                upstream,
                checkpoint_dir=checkpoint_dir,
                model=raw_model,
                optimizer=optimizer,
                saved_config=saved_config,
                completed_step=completed_step,
            )
            last_saved_step = completed_step

    if checkpoint_dir is not None and completed_step != last_saved_step:
        _save_atomic(
            upstream,
            checkpoint_dir=checkpoint_dir,
            model=raw_model,
            optimizer=optimizer,
            saved_config=saved_config,
            completed_step=completed_step,
        )
    telemetry = _finish_telemetry(telemetry_state, loop.device, torch=torch)
    return TrainResult(
        start_step=start_step,
        completed_step=completed_step,
        reports=tuple(reports),
        telemetry=telemetry,
    )


def _validate_full_batch(model: Any, batch: Mapping[str, Any]) -> None:
    torch = _require_torch()
    for key in ("rewards", "task"):
        if key not in batch:
            raise ValueError(f"Full WEAVER training requires real {key}; no synthetic fallback is allowed")
    rewards = batch["rewards"]
    actions = batch["actions"]
    if not isinstance(rewards, torch.Tensor) or rewards.shape != actions.shape[:2]:
        raise ValueError(
            f"Full WEAVER rewards must have shape {tuple(actions.shape[:2])}, got {getattr(rewards, 'shape', None)}"
        )
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("Full WEAVER rewards contain non-finite values")
    task = batch["task"]
    if not isinstance(task, Mapping) or not ({"text", "features"} & set(task)):
        raise ValueError("Full WEAVER task must contain text or precomputed CLIP features")
    if "features" in task:
        features = task["features"]
        expected_dim = int(model.task_encoder.feature_dim)
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 2
            or features.shape != (actions.shape[0], expected_dim)
            or not bool(torch.isfinite(features).all())
        ):
            raise ValueError(
                "Precomputed CLIP task features must be finite with shape "
                f"{(actions.shape[0], expected_dim)}"
            )
    else:
        text = task["text"]
        if (
            not isinstance(text, list | tuple)
            or len(text) != actions.shape[0]
            or any(not isinstance(item, str) or not item.strip() for item in text)
        ):
            raise ValueError("Full WEAVER task text must contain one non-empty prompt per batch item")

    transition_valid = batch["transition_valid"]
    if not isinstance(transition_valid, torch.Tensor) or transition_valid.dtype != torch.bool:
        raise TypeError("transition_valid must be a torch.bool tensor")
    if transition_valid.shape != actions.shape[:2]:
        raise ValueError(
            f"transition_valid must have shape {tuple(actions.shape[:2])}, got {tuple(transition_valid.shape)}"
        )
    # Frozen upstream RewardModel.compute_loss applies an unmasked MSE to every
    # row, while Critic.compute_loss has no terminal/bootstrap mask.  Full-mode
    # windows must therefore contain only real episode-internal transitions,
    # including the final row used by the reward head.  BehaviorPrecomputed
    # enforces this by reserving one real successor beyond the selected window.
    if not bool(transition_valid.all()):
        raise ValueError(
            "Frozen full WEAVER.forward has no reward/critic transition mask; every row in a full-mode window "
            "must be a real episode-internal transition"
        )


def _equivalent_upstream_adamw(
    parameters: tuple[Any, ...],
    *,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
    torch: Any,
) -> Any:
    decay = [parameter for parameter in parameters if parameter.dim() >= 2]
    no_decay = [parameter for parameter in parameters if parameter.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra = {"fused": True} if fused_available and device_type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)


def _training_batches(
    dataloader: Iterable[Mapping[str, Any]],
    *,
    cache_count: int,
) -> Iterator[Mapping[str, Any]]:
    if cache_count > 0:
        iterator = iter(dataloader)
        cached = []
        for _ in range(cache_count):
            try:
                cached.append(next(iterator))
            except StopIteration:
                break
        if not cached:
            raise ValueError("Tiny-overfit requested batches from an empty dataloader")
        while True:
            yield from cached
    else:
        while True:
            emitted = False
            for batch in dataloader:
                emitted = True
                yield batch
            if not emitted:
                raise ValueError("WEAVER dataloader is empty")


def _unique_optimizer_parameters(optimizer: Any) -> tuple[Any, ...]:
    result = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                result.append(parameter)
                seen.add(id(parameter))
    return tuple(result)


def _autocast_context(device: str, *, torch: Any) -> Any:
    device_type = torch.device(device).type
    if device_type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _detached_scalars(values: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in values.items():
        if hasattr(value, "numel") and value.numel() == 1:
            result[str(key)] = float(value.detach().item())
    return result


def _save_atomic(
    upstream: UpstreamTools,
    *,
    checkpoint_dir: str | Path,
    model: Any,
    optimizer: Any,
    saved_config: Mapping[str, Any],
    completed_step: int,
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    upstream.save_checkpoint(
        str(path),
        model=model,
        optimizer=optimizer,
        cfg=saved_config,
        step=completed_step,
        atomic=True,
    )


def _unwrap_model(model: Any) -> Any:
    """Return the checkpoint-owning model behind compile wrappers."""

    return getattr(model, "_orig_mod", model)


def _start_telemetry(device: str, *, torch: Any) -> dict[str, Any]:
    target = torch.device(device)
    state: dict[str, Any] = {"device": str(target), "started_at": time.perf_counter()}
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is false: {target}")
        torch.cuda.set_device(target)
        torch.cuda.reset_peak_memory_stats(target)
        properties = torch.cuda.get_device_properties(target)
        state.update(gpu_name=properties.name, gpu_total_bytes=int(properties.total_memory))
    return state


def _finish_telemetry(state: Mapping[str, Any], device: str, *, torch: Any) -> dict[str, Any]:
    result = dict(state)
    started_at = float(result.pop("started_at"))
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        result.update(
            allocated_bytes=int(torch.cuda.memory_allocated(target)),
            reserved_bytes=int(torch.cuda.memory_reserved(target)),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(target)),
        )
    result["elapsed_seconds"] = time.perf_counter() - started_at
    return result


def _require_torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("WEAVER training requires a PyTorch environment") from exc
    return torch
