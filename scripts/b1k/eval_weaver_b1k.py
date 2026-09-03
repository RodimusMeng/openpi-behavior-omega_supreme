"""Evaluate frozen-upstream WEAVER prediction on held-out BEHAVIOR episodes."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import dataclasses
from datetime import UTC
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml

from openpi.weaver import behavior_compliance
from openpi.weaver import prediction_eval
from openpi.weaver import reward_critic_eval
from openpi.weaver import upstream_config
from openpi.weaver.behavior_precomputed import BehaviorPrecomputed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-mode", choices=("identity", "resume", "b1k_23d_fresh"), required=True)
    parser.add_argument("--task-encoder-model", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--norm-stats-path")
    parser.add_argument("--hybrid-action-contract", required=True)
    parser.add_argument("--latent-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compliance-manifest")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument(
        "--horizon",
        type=int,
        action="append",
        default=[],
        help="Repeat for short/long-horizon evaluation; defaults to checkpoint eval_horizon/horizon",
    )
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reward-critic", action="store_true", help="Also evaluate frozen RM/critic on held-out labels"
    )
    parser.add_argument("--decode-rgb", action="store_true")
    parser.add_argument("--decode-chunk-size", type=int, default=16)
    parser.add_argument("--lpips", action="store_true", help="Compute real LPIPS; requires the optional lpips package")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nfe", type=int, help="Override model.val_steps (number of flow integration steps)")
    parser.add_argument(
        "--inference-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable override under the complete upstream inference config",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted override applied last to the complete checkpoint config",
    )
    return parser.parse_args(argv)


def load_evaluation_config(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], upstream_config.BehaviorConfigReport]:
    """Resolve checkpoint -> B1K embodiment -> explicit upstream overrides."""

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
        dynamics_only=not args.reward_critic,
    )
    for expression in args.inference_override:
        _apply_override(config, f"inference.{expression}")
    if args.nfe is not None:
        _positive_int(args.nfe, name="nfe")
        config["model"]["val_steps"] = args.nfe
    # The generic dotted override is deliberately last, matching the training
    # entrypoint and allowing access to every current/future upstream knob.
    for expression in args.override:
        _apply_override(config, expression)
    _validate_config(config, reward_critic=args.reward_critic)
    return config, report


def run(
    args: argparse.Namespace,
    *,
    build_model: Callable[..., Any],
    load_checkpoint_into_model: Callable[..., Any],
    dataset_factory: Callable[..., Any] = BehaviorPrecomputed,
    torch: Any | None = None,
) -> dict[str, Any]:
    """Build the exact model/data contracts and write JSONL plus summary JSON."""

    torch = torch or _require_torch()
    _validate_runtime_args(args)
    compliance = _load_compliance_manifest(args.compliance_manifest)
    config, config_report = load_evaluation_config(args)
    horizons = _resolve_horizons(args.horizon, config)
    _seed_everything(args.seed, torch=torch)

    output_dir = Path(args.output_dir).expanduser().resolve()
    jsonl_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    existing = [path for path in (jsonl_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite evaluation outputs without --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry = prediction_eval.DeviceTelemetry(torch, args.device)
    telemetry.start()
    dataset_cfg = config["dataset"]
    base_dataset = dataset_factory(
        dataset_cfg["path"],
        split=args.split,
        horizon=max(horizons),
        image_keys=tuple(dataset_cfg["img_keys"]),
        n_history=int(config["n_history"]),
        n_memory_frames=int(config["n_memory_frames"]),
        t_memory=int(config["t_memory"]),
        normalize=bool(dataset_cfg["normalize"]),
        norm_stats_path=dataset_cfg.get("norm_stats_path"),
        reward_mode="required" if args.reward_critic else "disabled",
        collapse_prob=0.0,
        episode_indices=None if args.episode_indices is None else tuple(args.episode_indices),
        max_trajectories=args.max_trajectories,
        latent_root=args.latent_root or dataset_cfg.get("latent_root"),
        hybrid_action_contract_path=args.hybrid_action_contract,
    )
    _assert_checkpoint_lineage_disjoint(
        base_dataset,
        checkpoint_dir=Path(args.checkpoint_dir).expanduser().resolve(),
        checkpoint_mode=args.checkpoint_mode,
    )
    _assert_checkpoint_norm_provenance(
        base_dataset,
        checkpoint_dir=Path(args.checkpoint_dir).expanduser().resolve(),
        checkpoint_mode=args.checkpoint_mode,
    )
    _assert_checkpoint_hybrid_provenance(
        base_dataset,
        checkpoint_dir=Path(args.checkpoint_dir).expanduser().resolve(),
        checkpoint_mode=args.checkpoint_mode,
    )
    held_out = prediction_eval.HeldOutWindowDataset(
        base_dataset,
        max_horizon=max(horizons),
        stride=args.window_stride,
        max_windows=args.max_windows,
    )
    worker_kwargs = {"persistent_workers": True, "prefetch_factor": 4} if args.num_workers > 0 else {}
    loader = torch.utils.data.DataLoader(
        held_out,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
        drop_last=False,
        **worker_kwargs,
    )

    model = build_model(config, device=args.device, task_encoder_model=args.task_encoder_model)
    load_result = load_checkpoint_into_model(
        model,
        Path(args.checkpoint_dir).expanduser().resolve(),
        mode=args.checkpoint_mode,
        device=args.device,
    )
    checkpoint_report = load_result.report.to_dict()
    # ``resume`` may expose a large source optimizer mapping. Evaluation never
    # restores it, so release that reference before held-out rollout.
    del load_result
    state_mean, state_std = _state_normalization(base_dataset)
    predictor = prediction_eval.UpstreamWeaverPredictor(
        model,
        torch=torch,
        device=args.device,
        n_history=int(config["n_history"]),
        state_mean=state_mean,
        state_std=state_std,
        use_ema=args.ema,
        decode_rgb=args.decode_rgb,
        decode_chunk_size=args.decode_chunk_size,
    )
    lpips_metric = _build_lpips_metric(torch, device=args.device, network=args.lpips_net) if args.lpips else None

    with prediction_eval.AtomicJsonlWriter(jsonl_path) as writer:
        metric_summary = prediction_eval.evaluate_prediction_batches(
            loader,
            predictor,
            horizons=horizons,
            record_sink=writer,
            lpips_metric=lpips_metric,
        )
    head_metrics = (
        _evaluate_reward_critic(loader, model, args=args, config=config, torch=torch) if args.reward_critic else None
    )
    cuda_telemetry = telemetry.finish()
    summary = {
        **metric_summary,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "split": args.split,
        "dataset_root": str(Path(dataset_cfg["path"]).expanduser().resolve()),
        "checkpoint_config": {
            "path": str(Path(args.checkpoint_config).expanduser().resolve()),
            "sha256": _file_sha256(Path(args.checkpoint_config).expanduser().resolve()),
            "resolved": config,
            "b1k_transfer": dataclasses.asdict(config_report),
            "overrides": list(args.override),
            "inference_overrides": list(args.inference_override),
        },
        "checkpoint_load": checkpoint_report,
        "evaluation": {
            "ema": bool(args.ema),
            "batch_size": args.batch_size,
            "window_stride": args.window_stride,
            "max_windows": args.max_windows,
            "max_trajectories": args.max_trajectories,
            "episode_indices": [int(manifest["episode_index"]) for manifest in base_dataset.manifests],
            "seed": args.seed,
            "val_steps": int(config["model"]["val_steps"]),
            "effective_nfe": _effective_nfe(config, horizon=max(horizons)),
            "inference": config.get("inference", {}),
            "decoded_rgb": bool(args.decode_rgb),
            "decoded_rgb_target": "decode(cached_ground_truth_latent)" if args.decode_rgb else None,
            "lpips": bool(args.lpips),
            "lpips_net": args.lpips_net if args.lpips else None,
        },
        "telemetry": cuda_telemetry,
        "reward_critic": head_metrics,
        "compliance_manifest": compliance,
    }
    prediction_eval.atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return summary


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


def _validate_config(config: Mapping[str, Any], *, reward_critic: bool) -> None:
    contract = upstream_config.architecture_contract(config)
    if (contract.n_states, contract.n_actions) != (23, 23):
        raise ValueError(f"B1K prediction evaluation requires 23-D state/action, got {contract}")
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("Resolved config has no dataset mapping")
    expected_reward_mode = "required" if reward_critic else "disabled"
    if dataset.get("reward_mode") != expected_reward_mode:
        raise ValueError(f"Evaluation requires dataset.reward_mode={expected_reward_mode!r}")
    if "model" not in config or not isinstance(config["model"], Mapping):
        raise ValueError("Resolved config has no model mapping")
    _positive_int(config["model"].get("val_steps"), name="model.val_steps")


def _validate_runtime_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "window_stride"):
        _positive_int(getattr(args, name), name=name)
    if isinstance(args.num_workers, bool) or not isinstance(args.num_workers, int) or args.num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    for name in ("max_windows", "max_trajectories"):
        value = getattr(args, name)
        if value is not None:
            _positive_int(value, name=name)
    if args.episode_indices is not None:
        if not args.episode_indices or any(value < 0 for value in args.episode_indices):
            raise ValueError("episode_indices must contain non-negative IDs")
        if len(set(args.episode_indices)) != len(args.episode_indices):
            raise ValueError("episode_indices must not contain duplicates")
        if args.max_trajectories is not None:
            raise ValueError("Use either --episode-indices or --max-trajectories, not both")
    if (
        isinstance(args.decode_chunk_size, bool)
        or not isinstance(args.decode_chunk_size, int)
        or args.decode_chunk_size < 0
    ):
        raise ValueError("decode_chunk_size must be a non-negative integer")
    if args.lpips and not args.decode_rgb:
        raise ValueError("--lpips requires --decode-rgb; LPIPS is never synthesized from latent/state metrics")


def _resolve_horizons(values: Sequence[int], config: Mapping[str, Any]) -> tuple[int, ...]:
    result = tuple(sorted(set(values))) if values else (int(config.get("eval_horizon", config["horizon"])),)
    for value in result:
        _positive_int(value, name="horizon")
    return result


def _state_normalization(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    stats = getattr(dataset, "norm_stats", None)
    if stats is None:
        return np.zeros(23, dtype=np.float32), np.ones(23, dtype=np.float32)
    if not isinstance(stats, Sequence) or len(stats) != 4:
        raise ValueError("BehaviorPrecomputed norm_stats must contain state/action mean/std")
    return np.asarray(stats[0], dtype=np.float32), np.asarray(stats[1], dtype=np.float32)


def _evaluate_reward_critic(
    loader: Any,
    model: Any,
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    evaluator = reward_critic_eval.FrozenHeadBatchEvaluator(
        model,
        torch=torch,
        device=args.device,
        n_history=int(config["n_history"]),
        use_ema=bool(args.ema),
    )
    horizon = max(_resolve_horizons(args.horizon, config))
    collected = {
        name: {key: [] for key in ("reward_pred", "reward_target", "value_pred", "value_target")}
        for name in ("ground_truth", "imagined")
    }
    for batch in loader:
        result = evaluator.evaluate(batch, horizon=horizon)
        for name, values in result.items():
            for key, value in values.items():
                collected[name][key].append(value)
    summary = {}
    for name, values in collected.items():
        if not values["reward_pred"]:
            raise ValueError("Held-out reward/critic loader yielded no batches")
        metrics = reward_critic_eval.evaluate_predictions(
            np.concatenate(values["reward_pred"], axis=0),
            np.concatenate(values["reward_target"], axis=0),
            value_pred=np.concatenate(values["value_pred"], axis=0),
            value_target=np.concatenate(values["value_target"], axis=0),
        )
        summary[name] = metrics.to_dict()
    summary["privileged_labels_are_targets_only"] = True
    summary["same_state_candidate_ranking"] = "separate_online_oracle_gate_required"
    return summary


def _build_lpips_metric(torch: Any, *, device: str, network: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    try:
        import lpips  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "--lpips requested, but the optional lpips package is not installed. "
            "Install lpips in the disposable WEAVER environment or omit --lpips."
        ) from exc
    metric = lpips.LPIPS(net=network).to(device).eval()

    def compute(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        if predicted.shape != target.shape or predicted.ndim != 5:
            raise ValueError(f"LPIPS inputs must share shape (B,H,C,Y,X), got {predicted.shape}/{target.shape}")
        batch, horizon = predicted.shape[:2]
        pred = torch.as_tensor(predicted, dtype=torch.float32, device=device).reshape(
            batch * horizon, *predicted.shape[2:]
        )
        truth = torch.as_tensor(target, dtype=torch.float32, device=device).reshape(batch * horizon, *target.shape[2:])
        with torch.no_grad():
            value = metric(pred * 2.0 - 1.0, truth * 2.0 - 1.0)
        return value.reshape(batch, horizon).detach().cpu().numpy()

    return compute


def _seed_everything(seed: int, *, torch: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_compliance_manifest(path_value: str | None) -> dict[str, Any] | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read compliance manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Compliance manifest must contain a JSON object")
    violations = behavior_compliance.validate_manifest(payload, submission=False)
    if violations:
        raise ValueError(f"Compliance manifest failed: {violations}")
    return {"path": str(path), "sha256": _file_sha256(path), "schema_version": payload.get("schema_version")}


def _effective_nfe(config: Mapping[str, Any], *, horizon: int) -> int:
    """Count frozen Diffusion-Forcing forward passes for this horizon."""

    steps = int(config["model"]["val_steps"])
    inference = config.get("inference", {})
    stagger = int(inference.get("pyramid_stagger_width", 1)) if isinstance(inference, Mapping) else 1
    if steps <= 0 or stagger < 0 or horizon <= 0:
        raise ValueError("val_steps/horizon must be positive and stagger width non-negative")
    # Frozen WEAVER _build_pyramid_schedule: height=S+w*(H-1)+1,
    # followed by height-1 model calls.
    return steps + stagger * (horizon - 1)


def _assert_checkpoint_lineage_disjoint(
    dataset: Any,
    *,
    checkpoint_dir: Path,
    checkpoint_mode: str,
) -> None:
    """Reject held-out episodes that were part of a B1K training checkpoint."""

    summary_path = checkpoint_dir / "training_summary.json"
    report_path = checkpoint_dir / "initialization_report.json"
    lineage: Mapping[str, Any] | None = None
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            candidate = payload.get("dataset_lineage")
            if isinstance(candidate, Mapping):
                lineage = candidate
    if lineage is None and report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            candidate = payload.get("dataset_lineage")
            if isinstance(candidate, Mapping):
                lineage = candidate
    if lineage is None:
        if checkpoint_mode != "b1k_23d_fresh":
            raise ValueError(
                "Trained/resumed checkpoint has no dataset lineage; refusing held-out evaluation without provenance"
            )
        return
    train_episodes = {int(value) for value in lineage.get("episode_indices", ())}
    train_semantic = {str(value) for value in lineage.get("semantic_sha256", ()) if str(value)}
    held_episodes: set[int] = set()
    held_semantic: set[str] = set()
    for manifest in getattr(dataset, "manifests", ()):
        held_episodes.add(int(manifest["episode_index"]))
        held_semantic.add(str(manifest.get("semantic_sha256", "")))
    episode_overlap = train_episodes & held_episodes
    semantic_overlap = train_semantic & held_semantic
    if episode_overlap or semantic_overlap:
        raise ValueError(
            "Held-out evaluation overlaps training lineage: "
            f"episode_indices={sorted(episode_overlap)}, semantic_sha256={sorted(semantic_overlap)}"
        )


def _assert_checkpoint_norm_provenance(
    dataset: Any,
    *,
    checkpoint_dir: Path,
    checkpoint_mode: str,
) -> None:
    """Prove held-out normalization is identical to the training artifact."""

    config_path = checkpoint_dir / "config.yaml"
    if not config_path.is_file():
        if checkpoint_mode != "b1k_23d_fresh":
            raise ValueError("Trained checkpoint has no saved config for normalization provenance")
        return
    try:
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read checkpoint config {config_path}: {exc}") from exc
    if not isinstance(saved, Mapping):
        raise ValueError(f"Checkpoint config must contain a mapping: {config_path}")
    expected = saved.get("weaver_norm_provenance")
    if expected is None:
        if checkpoint_mode != "b1k_23d_fresh":
            raise ValueError("Trained checkpoint has no normalization provenance")
        return
    actual = getattr(dataset, "norm_stats_provenance", None)
    if _canonical_norm_provenance(expected) != _canonical_norm_provenance(actual):
        raise ValueError("Held-out normalization artifact differs from checkpoint training provenance")


def _canonical_norm_provenance(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    normalized = {str(key): item for key, item in value.items() if key != "path"}
    return json.loads(json.dumps(normalized, sort_keys=True, allow_nan=False))


def _assert_checkpoint_hybrid_provenance(
    dataset: Any,
    *,
    checkpoint_dir: Path,
    checkpoint_mode: str,
) -> None:
    """Prove evaluation uses the hybrid action contract saved by training."""

    config_path = checkpoint_dir / "config.yaml"
    if not config_path.is_file():
        if checkpoint_mode != "b1k_23d_fresh":
            raise ValueError("Trained checkpoint has no saved config for hybrid action provenance")
        return
    try:
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read checkpoint config {config_path}: {exc}") from exc
    if not isinstance(saved, Mapping):
        raise ValueError(f"Checkpoint config must contain a mapping: {config_path}")
    expected = saved.get("weaver_hybrid_action_provenance")
    if expected is None:
        if checkpoint_mode != "b1k_23d_fresh":
            raise ValueError("Trained checkpoint has no hybrid action provenance")
        return
    actual = getattr(dataset, "hybrid_action_provenance", None)
    if _canonical_hybrid_provenance(expected) != _canonical_hybrid_provenance(actual):
        raise ValueError("Held-out hybrid action contract differs from checkpoint training provenance")


def _canonical_hybrid_provenance(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    normalized = {str(key): item for key, item in value.items() if key != "source_audit_path"}
    return json.loads(json.dumps(normalized, sort_keys=True, allow_nan=False))


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("eval_weaver_b1k.py must run in the frozen WEAVER PyTorch environment") from exc
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
