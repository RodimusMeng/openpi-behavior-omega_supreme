"""Train a 23-D WEAVER model on prepared BEHAVIOR-1K episode artifacts.

This entrypoint preserves the downloaded checkpoint configuration, applies the
upstream finetune block, then applies each explicit ``--override key=value`` in
command-line order.  ``--mode full`` calls the original reward/critic forward;
``--mode dynamics`` selects only the external masked flow-loss adapter.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml

from openpi.weaver import training_runtime
from openpi.weaver import upstream_config
from openpi.weaver.behavior_precomputed import BehaviorPrecomputed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-config", required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--task-encoder-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--norm-stats-path")
    parser.add_argument("--hybrid-action-contract", required=True)
    parser.add_argument("--mode", choices=("full", "dynamics"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--tiny-overfit-batches", type=int, default=0)
    parser.add_argument("--tiny-overfit-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable nested override applied after checkpoint and finetune configuration",
    )
    return parser.parse_args(argv)


def load_training_config(args: argparse.Namespace) -> tuple[dict[str, Any], upstream_config.BehaviorConfigReport]:
    """Resolve checkpoint -> BEHAVIOR -> finetune -> explicit override order."""

    checkpoint_path = Path(args.checkpoint_config).expanduser().resolve()
    try:
        raw = yaml.safe_load(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read checkpoint config {checkpoint_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"Checkpoint config must contain a mapping: {checkpoint_path}")

    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    norm_stats_path = (
        Path(args.norm_stats_path).expanduser().resolve()
        if args.norm_stats_path is not None
        else dataset_root / "norm_stats.json"
    )
    config, report = upstream_config.build_behavior_config(
        raw,
        dataset_root=dataset_root,
        norm_stats_path=norm_stats_path,
        dynamics_only=args.mode == "dynamics",
    )
    config = _apply_finetune_config(config)
    for override in args.override:
        _apply_override(config, override)
    _validate_mode_config(config, mode=args.mode)
    return config, report


def run(
    args: argparse.Namespace,
    *,
    build_model: Callable[..., Any],
    load_checkpoint_into_model: Callable[..., Any],
    upstream_tools: training_runtime.UpstreamTools | None = None,
    dataset_factory: Callable[..., Any] = BehaviorPrecomputed,
) -> tuple[training_runtime.TrainResult, Any]:
    """Construct the real dataset/model, load or resume, and execute training."""

    torch = _require_torch()
    config, config_report = load_training_config(args)
    _seed_everything(args.seed, torch=torch)
    dataset_cfg = config["dataset"]
    dataset = dataset_factory(
        dataset_cfg["path"],
        split=args.split,
        horizon=int(config["horizon"]),
        image_keys=tuple(dataset_cfg["img_keys"]),
        n_history=int(config["n_history"]),
        n_memory_frames=int(config["n_memory_frames"]),
        t_memory=int(config["t_memory"]),
        normalize=bool(dataset_cfg["normalize"]),
        norm_stats_path=dataset_cfg.get("norm_stats_path"),
        reward_mode="required" if args.mode == "full" else "disabled",
        collapse_prob=float(dataset_cfg.get("collapse_prob", 0.0)),
        episode_indices=None if args.episode_indices is None else tuple(args.episode_indices),
        max_trajectories=args.max_trajectories,
        latent_root=dataset_cfg.get("latent_root"),
        hybrid_action_contract_path=args.hybrid_action_contract,
    )
    if args.split != "train":
        raise ValueError("WEAVER production training is train-split only; held-out val/test are evaluation-only")
    lineage = _dataset_lineage(dataset, split=args.split)
    norm_provenance = getattr(dataset, "norm_stats_provenance", None)
    if not isinstance(norm_provenance, Mapping):
        raise ValueError("Normalized WEAVER training requires auditable norm_stats provenance")
    hybrid_provenance = getattr(dataset, "hybrid_action_provenance", None)
    if not isinstance(hybrid_provenance, Mapping):
        raise ValueError("Formal BEHAVIOR WEAVER training requires a hybrid causal action contract")

    training_cfg = config["training"]
    batch_size = int(training_cfg["batch_size"])
    n_workers = int(dataset_cfg.get("n_workers", 0))
    worker_kwargs = {"persistent_workers": True, "prefetch_factor": 4} if n_workers > 0 else {}
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=torch.device(args.device).type == "cuda",
        drop_last=len(dataset) >= batch_size,
        **worker_kwargs,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_checkpoint = output_dir / "checkpoint.pt"
    if args.resume and not output_checkpoint.is_file():
        raise FileNotFoundError(f"--resume requires an existing checkpoint: {output_checkpoint}")
    if not args.resume and output_checkpoint.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing training checkpoint without --resume: {output_checkpoint}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        _assert_resume_contract(
            output_dir,
            config=config,
            lineage=lineage,
            norm_provenance=norm_provenance,
            hybrid_provenance=hybrid_provenance,
        )
    model = build_model(
        config,
        device=args.device,
        task_encoder_model=args.task_encoder_model,
    )
    load_dir = output_dir if args.resume else Path(args.pretrained_dir).expanduser().resolve()
    load_mode = "resume" if args.resume else "b1k_23d_fresh"
    load_result = load_checkpoint_into_model(
        model,
        load_dir,
        mode=load_mode,
        device=args.device,
    )
    if bool(load_result.resumed) != bool(args.resume):
        raise ValueError(f"Checkpoint loader resumed={load_result.resumed!r} disagrees with --resume={args.resume!r}")

    optimizer = training_runtime.build_optimizer(
        model,
        mode=args.mode,
        weight_decay=float(training_cfg["weight_decay"]),
        learning_rate=float(training_cfg["max_lr"]),
        betas=tuple(float(value) for value in training_cfg["betas"]),
        device=args.device,
    )
    start_step = 0
    checkpoint_load_report = load_result.report.to_dict()
    if load_result.resumed:
        start_step = training_runtime.restore_optimizer_for_resume(optimizer, load_result.checkpoint)
    elif "optimizer" in load_result.checkpoint or "step" in load_result.checkpoint:
        raise ValueError("Fresh 8-D to 23-D transfer must not expose source optimizer or step")
    # The upstream loader intentionally keeps optimizer tensors on CPU. Once
    # restore_optimizer_for_resume has copied them into the target optimizer,
    # clear the source mapping before model training to avoid a duplicate copy.
    if isinstance(load_result.checkpoint, dict):
        load_result.checkpoint.clear()
    metrics_path = output_dir / "training_metrics.jsonl"
    _validate_training_metrics_resume(metrics_path, start_step=start_step)

    model = _apply_upstream_wrappers(model, config, torch=torch)
    loop = _loop_config(config, args)
    tools = upstream_tools or training_runtime.load_upstream_tools()
    transfer_payload = {
        "config_transfer": dataclasses.asdict(config_report),
        "checkpoint_load": checkpoint_load_report,
        "training_mode": args.mode,
        "task_encoder_model": str(Path(args.task_encoder_model).expanduser().resolve()),
        "resume": bool(args.resume),
        "dataset_lineage": lineage,
        "norm_stats_provenance": norm_provenance,
        "hybrid_action_provenance": hybrid_provenance,
    }
    _atomic_write_json(output_dir / "initialization_report.json", transfer_payload)
    result = training_runtime.train(
        model,
        dataloader,
        optimizer,
        loop=loop,
        upstream=tools,
        checkpoint_dir=output_dir,
        saved_config={
            **config,
            "weaver_lineage": lineage,
            "weaver_norm_provenance": norm_provenance,
            "weaver_hybrid_action_provenance": hybrid_provenance,
        },
        start_step=start_step,
    )
    metrics = _write_training_metrics(
        metrics_path,
        reports=result.reports,
        start_step=result.start_step,
        tiny_overfit_batches=args.tiny_overfit_batches,
    )
    summary = {
        "start_step": result.start_step,
        "completed_step": result.completed_step,
        "mode": args.mode,
        "split": args.split,
        "dataset_lineage": lineage,
        "norm_stats_provenance": norm_provenance,
        "hybrid_action_provenance": hybrid_provenance,
        "tiny_overfit_batches": args.tiny_overfit_batches,
        "tiny_overfit_steps": args.tiny_overfit_steps,
        "last_step": dataclasses.asdict(result.reports[-1]) if result.reports else None,
        "metrics": metrics,
        "telemetry": result.telemetry,
    }
    _atomic_write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return result, load_result


def _dataset_lineage(dataset: Any, *, split: str) -> dict[str, Any]:
    """Record the exact episode semantic IDs used by a training run."""

    manifests = getattr(dataset, "manifests", None)
    if not isinstance(manifests, Sequence):
        raise TypeError("BehaviorPrecomputed must expose manifests for training provenance")
    semantic = []
    episodes = []
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            raise TypeError("BehaviorPrecomputed manifests must contain mappings")
        semantic.append(str(manifest.get("semantic_sha256", "")))
        episodes.append(int(manifest["episode_index"]))
    if any(len(value) != 64 for value in semantic):
        raise ValueError("Every training manifest must contain a semantic SHA-256")
    return {
        "split": split,
        "episode_count": len(episodes),
        "episode_indices": episodes,
        "semantic_sha256": semantic,
    }


def _assert_resume_contract(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    lineage: Mapping[str, Any],
    norm_provenance: Mapping[str, Any],
    hybrid_provenance: Mapping[str, Any],
) -> None:
    """Reject resume when saved data/model/optimizer semantics changed."""

    config_path = output_dir / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"Resume checkpoint is missing its saved config: {config_path}")
    try:
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read resume config {config_path}: {exc}") from exc
    if not isinstance(saved, Mapping):
        raise ValueError(f"Resume config must contain a mapping: {config_path}")

    saved_lineage = saved.get("weaver_lineage")
    if _canonical(saved_lineage) != _canonical(lineage):
        raise ValueError("Resume dataset lineage differs from the saved checkpoint")
    saved_norm = saved.get("weaver_norm_provenance")
    if _canonical_provenance(saved_norm) != _canonical_provenance(norm_provenance):
        raise ValueError("Resume normalization provenance differs from the saved checkpoint")
    saved_hybrid = saved.get("weaver_hybrid_action_provenance")
    if _canonical_provenance(saved_hybrid) != _canonical_provenance(hybrid_provenance):
        raise ValueError("Resume hybrid action contract differs from the saved checkpoint")

    saved_config = {
        key: value
        for key, value in saved.items()
        if key not in {"weaver_lineage", "weaver_norm_provenance", "weaver_hybrid_action_provenance"}
    }
    if _canonical_config(saved_config) != _canonical_config(config):
        raise ValueError("Resume configuration differs from the saved checkpoint config")


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    return value


def _canonical_config(value: Mapping[str, Any]) -> Any:
    normalized = copy.deepcopy(dict(value))
    dataset = normalized.get("dataset")
    if isinstance(dataset, dict):
        # Absolute paths legitimately differ when a checkpoint is moved to the
        # GPU host; lineage and norm hashes above prove the underlying data.
        dataset.pop("path", None)
        dataset.pop("norm_stats_path", None)
    return _canonical(normalized)


def _canonical_provenance(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    normalized = dict(value)
    normalized.pop("path", None)
    normalized.pop("source_audit_path", None)
    nested = normalized.get("hybrid_action_contract")
    if isinstance(nested, Mapping):
        nested = dict(nested)
        nested.pop("source_audit_path", None)
        normalized["hybrid_action_contract"] = nested
    return _canonical(normalized)


def _loop_config(config: Mapping[str, Any], args: argparse.Namespace) -> training_runtime.LoopConfig:
    training = config["training"]
    return training_runtime.LoopConfig(
        mode=args.mode,
        device=args.device,
        max_steps=int(training["max_steps"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_lr=float(training["max_lr"]),
        min_lr=float(training["min_lr"]),
        warmup_steps=int(training["warmup_steps"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(float(value) for value in training["betas"]),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        checkpoint_every=int(config.get("ckpt_freq", 0)),
        tiny_overfit_batches=args.tiny_overfit_batches,
        tiny_overfit_steps=args.tiny_overfit_steps,
    )


def _apply_finetune_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    finetune = result.get("finetune_cfg", {})
    if finetune:
        if not isinstance(finetune, Mapping) or not isinstance(result.get("training"), dict):
            raise ValueError("finetune_cfg and training must both be mappings")
        result["training"].update(copy.deepcopy(dict(finetune)))
    return result


def _apply_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must use key=value syntax, got {expression!r}")
    key, raw_value = expression.split("=", 1)
    if not key or any(not part for part in key.split(".")):
        raise ValueError(f"Override has an invalid dotted key: {expression!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Override value is invalid YAML: {expression!r}") from exc
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if child is None:
            child = {}
            target[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Override path {key!r} crosses non-mapping field {part!r}")
        target = child
    target[parts[-1]] = value


def _validate_mode_config(config: Mapping[str, Any], *, mode: str) -> None:
    try:
        dataset = config["dataset"]
        training = config["training"]
    except KeyError as exc:
        raise ValueError(f"Resolved WEAVER config is missing {exc.args[0]!r}") from exc
    expected_reward_mode = "required" if mode == "full" else "disabled"
    if dataset.get("reward_mode") != expected_reward_mode:
        raise ValueError(
            f"{mode} mode requires dataset.reward_mode={expected_reward_mode!r}, got {dataset.get('reward_mode')!r}"
        )
    if bool(training.get("dynamics_only")) != (mode == "dynamics"):
        raise ValueError("training.dynamics_only must agree with the selected adapter mode")
    if (int(dataset.get("n_states", -1)), int(dataset.get("n_actions", -1))) != (23, 23):
        raise ValueError("BEHAVIOR WEAVER training requires 23-D state and action configuration")


def _apply_upstream_wrappers(model: Any, config: Mapping[str, Any], *, torch: Any) -> Any:
    if bool(config.get("use_activation_checkpointing", False)):
        try:
            from weaver.wm.nets import STAttentionBlock  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("Activation checkpointing requires frozen WEAVER modules") from exc
        for module in model.modules():
            if isinstance(module, STAttentionBlock):
                # This is the frozen upstream activation-checkpointing switch.
                module._use_checkpoint = True  # noqa: SLF001
    if bool(config.get("use_compile", False)):
        model = torch.compile(model)
    return model


def _seed_everything(seed: int, *, torch: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_training_metrics(
    path: Path,
    *,
    reports: Sequence[training_runtime.StepReport],
    start_step: int,
    tiny_overfit_batches: int,
) -> dict[str, Any]:
    """Atomically append one invocation's step reports to an auditable JSONL history."""

    existing = _read_training_metrics(path)
    _validate_training_metrics_resume(path, start_step=start_step, records=existing)
    appended = []
    for run_index, report in enumerate(reports):
        expected_step = start_step + run_index + 1
        if report.completed_step != expected_step:
            raise ValueError(
                f"Training report step {report.completed_step} is not contiguous with start step {start_step}"
            )
        appended.append(
            {
                **dataclasses.asdict(report),
                "run_start_step": start_step,
                "run_step": run_index + 1,
                "batch_slot": run_index % tiny_overfit_batches if tiny_overfit_batches > 0 else None,
            }
        )

    records = [*existing, *appended]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "records_appended": len(appended),
        "records_total": len(records),
        "first_recorded_step": records[0]["completed_step"] if records else None,
        "last_recorded_step": records[-1]["completed_step"] if records else None,
        "history_complete": bool(records) and records[0].get("completed_step") == 1,
    }


def _read_training_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid training metrics at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Training metrics must contain JSON objects: {path}:{line_number}")
                records.append(value)
    return records


def _validate_training_metrics_resume(
    path: Path,
    *,
    start_step: int,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    existing = _read_training_metrics(path) if records is None else records
    if existing:
        previous_step = existing[-1].get("completed_step")
        if previous_step != start_step:
            raise ValueError(
                f"Training metrics end at step {previous_step!r}, but resumed checkpoint starts at {start_step}"
            )


def _require_torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("train_weaver_b1k.py must run in the frozen WEAVER PyTorch environment") from exc
    return torch


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        from openpi.weaver import upstream_runtime  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("The OmegaSupreme upstream_runtime adapter is unavailable") from exc
    run(
        args,
        build_model=upstream_runtime.build_model,
        load_checkpoint_into_model=upstream_runtime.load_checkpoint_into_model,
    )


if __name__ == "__main__":
    main()
