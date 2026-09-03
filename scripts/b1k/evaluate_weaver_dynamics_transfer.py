"""Deterministically compare fresh and trained 23-D WEAVER dynamics checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml

from openpi.weaver import training_runtime
from openpi.weaver import upstream_config
from openpi.weaver import upstream_runtime
from openpi.weaver.behavior_precomputed import BehaviorPrecomputed

SCHEMA_VERSION = "b1k_weaver_dynamics_transfer_eval_v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Saved 23-D training config.yaml")
    parser.add_argument("--released-checkpoint-dir", required=True)
    parser.add_argument("--trained-checkpoint-dir", required=True)
    parser.add_argument("--task-encoder-model", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--norm-stats-path", required=True)
    parser.add_argument("--hybrid-action-contract", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--evaluation-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--fresh-device", default="cuda:0")
    parser.add_argument("--trained-device", default="cuda:1")
    parser.add_argument("--max-trajectories", type=int, default=4)
    parser.add_argument("--window-fraction", type=float, default=0.5)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505, 606, 707, 808])
    parser.add_argument("--initialization-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    torch = _require_torch()
    config_path = Path(args.config).expanduser().resolve()
    config = _read_config(config_path)
    contract = upstream_config.architecture_contract(upstream_config.checkpoint_defaults(config))
    if (contract.n_states, contract.n_actions) != (23, 23):
        raise ValueError("Dynamics transfer evaluation requires a saved 23-D/23-D config")
    if not 0.0 <= args.window_fraction <= 1.0 or not math.isfinite(args.window_fraction):
        raise ValueError("--window-fraction must be finite and in [0,1]")
    if args.max_trajectories <= 0 or not args.noise_seeds or len(set(args.noise_seeds)) != len(args.noise_seeds):
        raise ValueError("Evaluation requires positive trajectories and unique noise seeds")
    if torch.device(args.fresh_device) == torch.device(args.trained_device):
        raise ValueError("Fresh and trained evaluation devices must differ")

    dataset_cfg = config["dataset"]
    dataset = BehaviorPrecomputed(
        args.dataset_dir,
        split=args.evaluation_split,
        horizon=int(config["horizon"]),
        image_keys=tuple(dataset_cfg["img_keys"]),
        n_history=int(config["n_history"]),
        n_memory_frames=int(config["n_memory_frames"]),
        t_memory=int(config["t_memory"]),
        normalize=True,
        norm_stats_path=args.norm_stats_path,
        reward_mode="disabled",
        collapse_prob=0.0,
        episode_indices=None if args.episode_indices is None else tuple(args.episode_indices),
        max_trajectories=args.max_trajectories if args.episode_indices is None else None,
        latent_root=dataset_cfg.get("latent_root"),
        hybrid_action_contract_path=args.hybrid_action_contract,
    )
    evaluation_protocol = _validate_data_contract(config, dataset)
    windows = _evaluation_windows(dataset, fraction=args.window_fraction)

    _seed_everything(args.initialization_seed, torch=torch)
    fresh = upstream_runtime.build_model(
        config,
        device=args.fresh_device,
        task_encoder_model=args.task_encoder_model,
    )
    fresh_load = upstream_runtime.load_checkpoint_into_model(
        fresh,
        args.released_checkpoint_dir,
        mode=upstream_runtime.B1K_23D_FRESH_MODE,
        device=args.fresh_device,
    )

    trained = upstream_runtime.build_model(
        config,
        device=args.trained_device,
        task_encoder_model=args.task_encoder_model,
    )
    trained_load = upstream_runtime.load_checkpoint_into_model(
        trained,
        args.trained_checkpoint_dir,
        mode=upstream_runtime.RESUME_MODE,
        device=args.trained_device,
    )
    if isinstance(trained_load.checkpoint, dict):
        trained_load.checkpoint.clear()

    fresh.eval()
    trained.eval()
    tools = training_runtime.load_upstream_tools()
    records = []
    with torch.inference_mode():
        for window in windows:
            collated = torch.utils.data.default_collate([window["batch"]])
            for noise_seed in args.noise_seeds:
                fresh_values = _evaluate_once(
                    fresh,
                    collated,
                    device=args.fresh_device,
                    noise_seed=noise_seed,
                    tools=tools,
                    torch=torch,
                )
                trained_values = _evaluate_once(
                    trained,
                    collated,
                    device=args.trained_device,
                    noise_seed=noise_seed,
                    tools=tools,
                    torch=torch,
                )
                records.append(
                    {
                        "batch_slot": window["batch_slot"],
                        "episode_index": window["episode_index"],
                        "window_t": window["window_t"],
                        "noise_seed": noise_seed,
                        "fresh": fresh_values,
                        "trained": trained_values,
                    }
                )

    metric_names = ("Total Loss", "flow/Total Loss", "flow/states", *[f"flow/{key}" for key in dataset.image_keys])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "dataset": {
            "root": str(Path(args.dataset_dir).expanduser().resolve()),
            "split": dataset.split,
            "episode_indices": [window["episode_index"] for window in windows],
            "window_t": [window["window_t"] for window in windows],
            "window_fraction": args.window_fraction,
            "norm_stats_path": str(Path(args.norm_stats_path).expanduser().resolve()),
        },
        "evaluation": {
            "protocol": evaluation_protocol,
            "model_mode": "eval",
            "weights": "raw_model",
            "autocast": "bfloat16",
            "initialization_seed": args.initialization_seed,
            "noise_seeds": list(args.noise_seeds),
            "records": len(records),
        },
        "fresh": {
            "device": args.fresh_device,
            "checkpoint_load": fresh_load.report.to_dict(),
        },
        "trained": {
            "device": args.trained_device,
            "checkpoint_load": trained_load.report.to_dict(),
        },
        "aggregate": _aggregate(records, metric_names=metric_names),
        "aggregate_by_batch_slot": _aggregate_by_batch_slot(records, metric_names=metric_names),
        "records": records,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    _atomic_write_json(output_path, payload)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "output_json": str(output_path),
                "dataset": payload["dataset"],
                "evaluation": payload["evaluation"],
                "fresh_checkpoint": {
                    "mode": fresh_load.report.mode,
                    "path": fresh_load.report.checkpoint_path,
                    "step": fresh_load.report.checkpoint_step,
                },
                "trained_checkpoint": {
                    "mode": trained_load.report.mode,
                    "path": trained_load.report.checkpoint_path,
                    "step": trained_load.report.checkpoint_step,
                },
                "aggregate": payload["aggregate"],
                "aggregate_by_batch_slot": payload["aggregate_by_batch_slot"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _evaluation_windows(dataset: BehaviorPrecomputed, *, fraction: float) -> list[dict[str, Any]]:
    windows = []
    for batch_slot, manifest in enumerate(dataset.manifests):
        min_t = dataset.n_history - 1
        max_t = int(manifest["sampled_count"]) - dataset.horizon - 1
        window_t = min_t + round((max_t - min_t) * fraction)
        windows.append(
            {
                "batch_slot": batch_slot,
                "episode_index": int(manifest["episode_index"]),
                "window_t": window_t,
                "batch": dataset.window_at(batch_slot, window_t, collapse=False),
            }
        )
    return windows


def _validate_data_contract(config: Mapping[str, Any], dataset: BehaviorPrecomputed) -> str:
    expected_lineage = {
        "split": dataset.split,
        "episode_count": len(dataset.manifests),
        "episode_indices": [int(manifest["episode_index"]) for manifest in dataset.manifests],
        "semantic_sha256": [str(manifest["semantic_sha256"]) for manifest in dataset.manifests],
    }
    training_lineage = config.get("weaver_lineage")
    if not isinstance(training_lineage, Mapping):
        raise ValueError("Trained checkpoint config has no WEAVER dataset lineage")
    if training_lineage == expected_lineage:
        protocol = "in_sample"
    else:
        training_ids = {int(value) for value in training_lineage.get("episode_indices", ())}
        evaluation_ids = {int(value) for value in expected_lineage["episode_indices"]}
        if overlap := sorted(training_ids & evaluation_ids):
            raise ValueError(f"Held-out evaluation overlaps training episode IDs: {overlap}")
        protocol = "held_out_episode"
    saved_norm = config.get("weaver_norm_provenance")
    actual_norm = dataset.norm_stats_provenance
    if not isinstance(saved_norm, Mapping) or not isinstance(actual_norm, Mapping):
        raise ValueError("Evaluation requires saved and actual normalization provenance")
    saved_without_path = {key: value for key, value in saved_norm.items() if key != "path"}
    actual_without_path = {key: value for key, value in actual_norm.items() if key != "path"}
    if saved_without_path != actual_without_path:
        raise ValueError("Evaluation normalization provenance differs from the trained checkpoint config")
    saved_hybrid = _canonical_provenance(config.get("weaver_hybrid_action_provenance"))
    actual_hybrid = _canonical_provenance(dataset.hybrid_action_provenance)
    if saved_hybrid is None or actual_hybrid is None or saved_hybrid != actual_hybrid:
        raise ValueError("Evaluation hybrid action contract differs from the trained checkpoint config")
    return protocol


def _canonical_provenance(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    normalized = dict(value)
    normalized.pop("path", None)
    normalized.pop("source_audit_path", None)
    return {
        str(key): _canonical_provenance(item) if isinstance(item, Mapping) else item
        for key, item in sorted(normalized.items(), key=lambda pair: str(pair[0]))
    }


def _evaluate_once(
    model: Any,
    batch: Mapping[str, Any],
    *,
    device: str,
    noise_seed: int,
    tools: training_runtime.UpstreamTools,
    torch: Any,
) -> dict[str, float]:
    _seed_everything(noise_seed, torch=torch)
    device_batch = tools.move_tensors_to_device(batch, device=device)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
        loss, values = training_runtime.compute_loss(model, device_batch, mode="dynamics")
    result = {name: float(value.detach().item()) for name, value in values.items() if value.numel() == 1}
    result["loss"] = float(loss.detach().item())
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"Non-finite deterministic evaluation result on {device}: {result}")
    return result


def _aggregate(records: Sequence[Mapping[str, Any]], *, metric_names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in metric_names:
        fresh = [float(record["fresh"][metric]) for record in records]
        trained = [float(record["trained"][metric]) for record in records]
        deltas = [target - source for source, target in zip(fresh, trained, strict=True)]
        fresh_mean = sum(fresh) / len(fresh)
        trained_mean = sum(trained) / len(trained)
        result[metric] = {
            "fresh_mean": fresh_mean,
            "trained_mean": trained_mean,
            "delta": trained_mean - fresh_mean,
            "relative_delta": (trained_mean - fresh_mean) / fresh_mean if fresh_mean != 0 else None,
            "trained_better_count": sum(delta < 0 for delta in deltas),
            "comparison_count": len(deltas),
        }
    return result


def _aggregate_by_batch_slot(
    records: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
) -> dict[str, Any]:
    slots = sorted({int(record["batch_slot"]) for record in records})
    return {
        str(slot): _aggregate(
            [record for record in records if int(record["batch_slot"]) == slot],
            metric_names=metric_names,
        )
        for slot in slots
    }


def _read_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read evaluation config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation config must contain a mapping: {path}")
    return value


def _seed_everything(seed: int, *, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite deterministic evaluation: {path}")
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
        raise RuntimeError("Deterministic dynamics evaluation requires the frozen WEAVER PyTorch environment") from exc
    return torch


if __name__ == "__main__":
    main()
