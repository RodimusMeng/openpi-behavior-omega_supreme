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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--norm-stats-path")
    parser.add_argument("--mode", choices=("full", "dynamics"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--resume", action="store_true")
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
        max_trajectories=args.max_trajectories,
        latent_root=dataset_cfg.get("latent_root"),
    )
    if args.split != "train":
        raise ValueError("WEAVER production training is train-split only; held-out val/test are evaluation-only")
    lineage = _dataset_lineage(dataset, split=args.split)

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
    model = build_model(config, device=args.device)
    load_dir = output_dir if args.resume else Path(args.pretrained_dir).expanduser().resolve()
    load_mode = "resume" if args.resume else "b1k_23d_fresh"
    load_result = load_checkpoint_into_model(
        model,
        load_dir,
        mode=load_mode,
        device=args.device,
    )
    if bool(load_result.resumed) != bool(args.resume):
        raise ValueError(
            f"Checkpoint loader resumed={load_result.resumed!r} disagrees with --resume={args.resume!r}"
        )

    optimizer = training_runtime.build_optimizer(
        model,
        mode=args.mode,
        weight_decay=float(training_cfg["weight_decay"]),
        learning_rate=float(training_cfg["max_lr"]),
        betas=tuple(float(value) for value in training_cfg["betas"]),
        device=args.device,
    )
    start_step = 0
    if load_result.resumed:
        start_step = training_runtime.restore_optimizer_for_resume(optimizer, load_result.checkpoint)
    elif "optimizer" in load_result.checkpoint or "step" in load_result.checkpoint:
        raise ValueError("Fresh 8-D to 23-D transfer must not expose source optimizer or step")

    model = _apply_upstream_wrappers(model, config, torch=torch)
    loop = _loop_config(config, args)
    tools = upstream_tools or training_runtime.load_upstream_tools()
    transfer_payload = {
        "config_transfer": dataclasses.asdict(config_report),
        "checkpoint_load": load_result.report.to_dict(),
        "training_mode": args.mode,
        "resume": bool(args.resume),
        "dataset_lineage": lineage,
    }
    _atomic_write_json(output_dir / "initialization_report.json", transfer_payload)
    result = training_runtime.train(
        model,
        dataloader,
        optimizer,
        loop=loop,
        upstream=tools,
        checkpoint_dir=output_dir,
        saved_config={**config, "weaver_lineage": lineage},
        start_step=start_step,
    )
    summary = {
        "start_step": result.start_step,
        "completed_step": result.completed_step,
        "mode": args.mode,
        "split": args.split,
        "dataset_lineage": lineage,
        "tiny_overfit_batches": args.tiny_overfit_batches,
        "tiny_overfit_steps": args.tiny_overfit_steps,
        "last_step": dataclasses.asdict(result.reports[-1]) if result.reports else None,
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
            f"{mode} mode requires dataset.reward_mode={expected_reward_mode!r}, "
            f"got {dataset.get('reward_mode')!r}"
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
