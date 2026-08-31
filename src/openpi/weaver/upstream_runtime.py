"""Construct and migrate the frozen upstream WEAVER model without forking it.

The upstream package is imported lazily so configuration and report tooling can
run in the Omega environment. ``build_model`` mirrors the frozen WEAVER
construction boundary and passes the complete model/inference configuration
through unchanged. As in upstream ``finetune.py``, checkpoint migration happens
before optional ``torch.compile`` or activation-checkpoint wrappers are applied
by the training entrypoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from pathlib import Path
from typing import Any, Literal, NamedTuple

from openpi.weaver import upstream_config

IDENTITY_MODE = "identity"
RESUME_MODE = "resume"
B1K_23D_FRESH_MODE = "b1k_23d_fresh"
LoadMode = Literal["identity", "resume", "b1k_23d_fresh"]

EXPECTED_8D_TO_23D_MISMATCHES = (
    "wm.inp_prj.states.c_fc.weight",
    "wm.inp_prj.actions.c_fc.weight",
    "wm.mem_inp_prj.states.c_fc.weight",
    "wm.mem_inp_prj.actions.c_fc.weight",
    "wm.out_prj.states.weight",
    "rm.net.inp_prj.states.c_fc.weight",
    "rm.net.inp_prj.actions.c_fc.weight",
    "critic.net.inp_prj.states.c_fc.weight",
    "critic.net.inp_prj.actions.c_fc.weight",
)


class UpstreamRuntimeError(RuntimeError):
    """Raised when the frozen upstream package cannot satisfy its contract."""


class CheckpointContractError(ValueError):
    """Raised when a checkpoint is incompatible with the requested load mode."""


class _UpstreamApi(NamedTuple):
    get_encoder: Any
    get_task_encoder: Any
    model_type: Any
    dict_to_namespace: Any
    load_checkpoint: Any


@dataclasses.dataclass(frozen=True)
class ModelTransferReport:
    loaded_keys: tuple[str, ...]
    shape_mismatches: tuple[dict[str, Any], ...]
    missing_source_keys: tuple[str, ...]
    unexpected_source_keys: tuple[str, ...]
    source_key_rewrites: tuple[dict[str, str], ...]
    target_key_rewrites: tuple[dict[str, str], ...]


@dataclasses.dataclass(frozen=True)
class EmaTransferReport:
    strategy: Literal["loaded_full", "reset_from_target_model"]
    source_state_keys: tuple[str, ...]
    shadow_parameter_count: int
    reset_parameter_names: tuple[str, ...] = ()
    reset_shadow_shapes: tuple[tuple[int, ...], ...] = ()


@dataclasses.dataclass(frozen=True)
class CheckpointLoadReport:
    mode: LoadMode
    checkpoint_path: str
    checkpoint_step: int | float | str | None
    model: ModelTransferReport
    ema: EmaTransferReport

    def to_dict(self) -> dict[str, Any]:
        """Return a value accepted by ``json.dumps`` without custom encoders."""

        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LoadResult:
    """Loaded weights plus the checkpoint state allowed for training resume.

    ``checkpoint`` retains optimizer/step only when ``resumed`` is true. Fresh
    transfer and identity-audit callers therefore cannot accidentally restore
    the released source optimizer or advance their target training step.
    """

    checkpoint: Mapping[str, Any]
    report: CheckpointLoadReport
    resumed: bool


def _load_upstream_api() -> _UpstreamApi:
    try:
        from weaver.utils.config import dict_to_namespace  # noqa: PLC0415
        from weaver.utils.tools import load_checkpoint  # noqa: PLC0415
        from weaver.wm.encoders import get_encoder  # noqa: PLC0415
        from weaver.wm.encoders import get_task_encoder  # noqa: PLC0415
        from weaver.wm.model import WEAVER  # noqa: PLC0415
    except ImportError as exc:
        raise UpstreamRuntimeError(
            "Frozen WEAVER is not importable. Add its 3e41e907 checkout to PYTHONPATH "
            "and run model construction in the official Python 3.11 environment."
        ) from exc
    return _UpstreamApi(
        get_encoder=get_encoder,
        get_task_encoder=get_task_encoder,
        model_type=WEAVER,
        dict_to_namespace=dict_to_namespace,
        load_checkpoint=load_checkpoint,
    )


def _load_transfer_api():
    # checkpoint_transfer imports PyTorch, so keep it out of config-only tools.
    from openpi.weaver import checkpoint_transfer  # noqa: PLC0415

    return checkpoint_transfer


def _image_keys_from_config(cfg: Any) -> tuple[str, ...]:
    # This is the exact frozen upstream generate/finetune selection rule.
    if bool(getattr(cfg.dataset, "sample_aux_from_left_right", False)):
        return ("aux", "wrist")
    return tuple(cfg.dataset.img_keys)


def _load_clip_encoder_type():
    try:
        from weaver.wm.encoders import ClipEncoder  # noqa: PLC0415
    except ImportError as exc:
        raise UpstreamRuntimeError("Frozen WEAVER ClipEncoder is not importable from weaver.wm.encoders") from exc
    return ClipEncoder


def _build_task_encoder(
    api: _UpstreamApi,
    *,
    device: str,
    task_encoder_model: str | Path | None,
):
    """Build the frozen WEAVER task encoder, optionally from a pinned snapshot."""

    if task_encoder_model is None:
        return api.get_task_encoder(config=None, device=device)

    snapshot = Path(task_encoder_model).expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned task-encoder snapshot does not exist: {snapshot}")

    clip_encoder_type = _load_clip_encoder_type()
    return clip_encoder_type(
        model_name=str(snapshot),
        device=device,
    )


def build_model(
    config: Mapping[str, Any],
    *,
    device: str,
    task_encoder_model: str | Path | None = None,
):
    """Build the full original WEAVER for either released 8D or B1K 23D.

    No model/inference/memory/SPRINT setting is reconstructed locally. The
    concrete checkpoint config is deep-copied, converted with upstream's own
    namespace helper, and its complete ``model`` and ``inference`` objects are
    handed to the original ``WEAVER`` constructor. Top-level compile and
    activation-checkpoint flags intentionally remain post-load training-loop
    concerns, matching frozen upstream ordering.
    """

    concrete = upstream_config.checkpoint_defaults(config)
    contract = upstream_config.architecture_contract(concrete)
    if (contract.n_states, contract.n_actions) not in {(8, 8), (23, 23)}:
        raise ValueError(
            "Frozen transfer runtime supports released identity 8D or B1K 23D state/action, "
            f"got {contract.n_states}/{contract.n_actions}"
        )
    api = _load_upstream_api()
    cfg = api.dict_to_namespace(concrete)
    image_keys = _image_keys_from_config(cfg)
    if not image_keys or len(set(image_keys)) != len(image_keys):
        raise ValueError(f"WEAVER image views must be non-empty and unique, got {image_keys}")

    image_size: tuple[int, int] | int = cfg.dataset.image_size
    normalized_image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    im_encoder, train_decoder = api.get_encoder(
        config=cfg.im_encoder,
        image_size=cfg.dataset.image_size,
        device=device,
    )
    task_encoder = _build_task_encoder(
        api,
        device=device,
        task_encoder_model=task_encoder_model,
    )
    model = api.model_type(
        img_keys=list(image_keys),
        im_encoder=im_encoder,
        train_decoder=train_decoder,
        task_encoder=task_encoder,
        n_history=cfg.n_history,
        n_horizon=cfg.horizon,
        config=cfg.model,
        use_precomputed_features=cfg.use_precomputed_features,
        n_states=cfg.dataset.n_states,
        n_actions=cfg.dataset.n_actions,
        image_size=normalized_image_size,
        device=device,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        inference_config=cfg.inference,
    ).to(device)
    if not hasattr(model, "ema") or not hasattr(model.ema, "to"):
        raise UpstreamRuntimeError("Frozen WEAVER model did not construct its EMA wrapper")
    model.ema.to(device)
    return model


def load_checkpoint_into_model(
    model: Any,
    checkpoint_dir: str | Path,
    *,
    mode: LoadMode,
    device: str,
) -> LoadResult:
    """Strictly load identity weights or perform the audited 8D-to-23D transfer.

    ``identity`` and ``resume`` require a complete one-to-one model state and a
    structurally complete EMA state. Only ``resume`` exposes optimizer/step to
    the caller. ``b1k_23d_fresh`` accepts exactly the nine known 8D-to-23D
    projection mismatches, requires no missing/unexpected names, and discards
    the positional source EMA in favor of a full target-model reset.
    """

    if mode not in {IDENTITY_MODE, RESUME_MODE, B1K_23D_FRESH_MODE}:
        raise ValueError(f"Unsupported WEAVER checkpoint mode: {mode!r}")
    if not hasattr(model, "ema"):
        raise TypeError("load_checkpoint_into_model requires the raw upstream WEAVER model before compile/DDP")

    api = _load_upstream_api()
    transfer_api = _load_transfer_api()
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    # Keep the multi-gigabyte source state on CPU during construction. PyTorch's
    # load_state_dict copies compatible tensors into the already placed target;
    # retaining a second CUDA copy would inflate peak memory for no benefit.
    checkpoint = api.load_checkpoint(str(checkpoint_root), device="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise CheckpointContractError("Upstream checkpoint must contain a mapping")
    source_model = checkpoint.get("model")
    source_ema = checkpoint.get("ema")
    if not isinstance(source_model, Mapping):
        raise CheckpointContractError("Upstream checkpoint has no model state mapping")
    if not isinstance(source_ema, Mapping):
        raise CheckpointContractError("Upstream checkpoint has no EMA state mapping")
    if mode == RESUME_MODE and ("optimizer" not in checkpoint or "step" not in checkpoint):
        raise CheckpointContractError("Resume checkpoint must contain optimizer and step state")

    strict_identity = mode in {IDENTITY_MODE, RESUME_MODE}
    merged, transfer = transfer_api.adapt_state_dict(
        source_model,
        model.state_dict(),
        require_identity=strict_identity,
    )
    if strict_identity:
        if not transfer.is_identity_compatible:
            raise CheckpointContractError("Identity model transfer was not one-to-one")
    else:
        _assert_exact_8d_to_23d_transfer(transfer)

    incompatible = model.load_state_dict(merged, strict=True)
    missing = tuple(getattr(incompatible, "missing_keys", ()))
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    if missing or unexpected:
        raise CheckpointContractError(f"Strict merged model load returned missing={missing}, unexpected={unexpected}")

    if strict_identity:
        target_ema_state = model.ema.state_dict()
        _assert_same_tree_schema(source_ema, target_ema_state, path="ema")
        model.ema.load_state_dict(source_ema)
        model.ema.to(device)
        _assert_same_tree_schema(source_ema, model.ema.state_dict(), path="ema.loaded")
        ema_report = EmaTransferReport(
            strategy="loaded_full",
            source_state_keys=tuple(source_ema),
            shadow_parameter_count=_shadow_parameter_count(source_ema),
        )
    else:
        if not hasattr(model.ema, "ema"):
            raise CheckpointContractError("Upstream EMA wrapper has no inner Diffusers EMA object")
        reset = transfer_api.reset_ema_from_model(model.ema.ema, model)
        target_parameter_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if tuple(reset.parameter_names) != target_parameter_names:
            raise CheckpointContractError("Cross-dimension EMA reset did not cover every trainable parameter")
        if len(model.ema.ema.shadow_params) != len(target_parameter_names):
            raise CheckpointContractError("Cross-dimension EMA reset produced the wrong shadow-parameter count")
        ema_report = EmaTransferReport(
            strategy="reset_from_target_model",
            source_state_keys=tuple(source_ema),
            shadow_parameter_count=len(model.ema.ema.shadow_params),
            reset_parameter_names=tuple(reset.parameter_names),
            reset_shadow_shapes=tuple(tuple(shape) for shape in reset.shadow_shapes),
        )

    report = CheckpointLoadReport(
        mode=mode,
        checkpoint_path=str(checkpoint_root / "checkpoint.pt"),
        checkpoint_step=_json_scalar(checkpoint.get("step")),
        model=_model_report(transfer),
        ema=ema_report,
    )
    resumed = mode == RESUME_MODE
    # Never retain duplicate model/EMA tensors after loading. Resume exposes
    # only the optimizer state and step needed by the caller; fresh/identity
    # modes structurally cannot restore the source optimizer.
    allowed_checkpoint = {"optimizer": checkpoint["optimizer"], "step": checkpoint["step"]} if resumed else {}
    return LoadResult(checkpoint=allowed_checkpoint, report=report, resumed=resumed)


def _assert_exact_8d_to_23d_transfer(report: Any) -> None:
    mismatch_by_key = {mismatch.key: mismatch for mismatch in report.shape_mismatches}
    if len(report.shape_mismatches) != len(EXPECTED_8D_TO_23D_MISMATCHES) or tuple(sorted(mismatch_by_key)) != tuple(
        sorted(EXPECTED_8D_TO_23D_MISMATCHES)
    ):
        raise CheckpointContractError(
            "B1K transfer must contain exactly the nine audited 8D-to-23D shape mismatches; "
            f"got {tuple(mismatch.key for mismatch in report.shape_mismatches)}"
        )
    if report.missing_source_keys or report.unexpected_source_keys:
        raise CheckpointContractError(
            "B1K transfer may not add/drop parameter names: "
            f"missing={report.missing_source_keys}, unexpected={report.unexpected_source_keys}"
        )
    for key in EXPECTED_8D_TO_23D_MISMATCHES:
        mismatch = mismatch_by_key[key]
        source_shape = tuple(mismatch.source_shape)
        target_shape = tuple(mismatch.target_shape)
        axis = 0 if key == "wm.out_prj.states.weight" else -1
        source_other = source_shape[1:] if axis == 0 else source_shape[:-1]
        target_other = target_shape[1:] if axis == 0 else target_shape[:-1]
        if (
            len(source_shape) != len(target_shape)
            or source_shape[axis] != 8
            or target_shape[axis] != 23
            or source_other != target_other
        ):
            raise CheckpointContractError(
                f"Unexpected dimension transition for {key}: {source_shape} -> {target_shape}"
            )


def _assert_same_tree_schema(source: Any, target: Any, *, path: str) -> None:
    if isinstance(source, Mapping):
        if not isinstance(target, Mapping) or set(source) != set(target):
            source_keys = tuple(source)
            target_keys = tuple(target) if isinstance(target, Mapping) else type(target).__name__
            raise CheckpointContractError(
                f"EMA mapping schema differs at {path}: source={source_keys}, target={target_keys}"
            )
        for key in source:
            _assert_same_tree_schema(source[key], target[key], path=f"{path}.{key}")
        return
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        if not isinstance(target, Sequence) or isinstance(target, (str, bytes, bytearray)):
            raise CheckpointContractError(f"EMA sequence type differs at {path}")
        if len(source) != len(target):
            raise CheckpointContractError(f"EMA sequence length differs at {path}: {len(source)} != {len(target)}")
        for index, (source_item, target_item) in enumerate(zip(source, target, strict=True)):
            _assert_same_tree_schema(source_item, target_item, path=f"{path}[{index}]")
        return

    source_shape = getattr(source, "shape", None)
    target_shape = getattr(target, "shape", None)
    if source_shape is not None or target_shape is not None:
        if source_shape is None or target_shape is None or tuple(source_shape) != tuple(target_shape):
            raise CheckpointContractError(f"EMA tensor shape differs at {path}: {source_shape!r} != {target_shape!r}")
        return
    if type(source) is not type(target):
        raise CheckpointContractError(
            f"EMA scalar type differs at {path}: {type(source).__name__} != {type(target).__name__}"
        )


def _shadow_parameter_count(ema_state: Mapping[str, Any]) -> int:
    shadow_params = ema_state.get("shadow_params")
    if not isinstance(shadow_params, Sequence) or isinstance(shadow_params, (str, bytes, bytearray)):
        raise CheckpointContractError("EMA state has no shadow_params sequence")
    return len(shadow_params)


def _model_report(report: Any) -> ModelTransferReport:
    return ModelTransferReport(
        loaded_keys=tuple(report.loaded_keys),
        shape_mismatches=tuple(
            {
                "key": mismatch.key,
                "source_shape": list(mismatch.source_shape),
                "target_shape": list(mismatch.target_shape),
            }
            for mismatch in report.shape_mismatches
        ),
        missing_source_keys=tuple(report.missing_source_keys),
        unexpected_source_keys=tuple(report.unexpected_source_keys),
        source_key_rewrites=tuple(
            {"original": rewrite.original, "canonical": rewrite.canonical} for rewrite in report.source_key_rewrites
        ),
        target_key_rewrites=tuple(
            {"original": rewrite.original, "canonical": rewrite.canonical} for rewrite in report.target_key_rewrites
        ),
    )


def _json_scalar(value: Any) -> int | float | str | None:
    if value is None or isinstance(value, (int, float, str)):
        return value
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, (int, float, str)):
            return item
    return str(value)
