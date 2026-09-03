"""Audit π0.5 gripper boundaries and WEAVER stochastic candidate attribution.

The policy pool is sampled once per recorded observation.  WEAVER scoring then
reuses that immutable pool.  Each candidate is also scored alone under common
random seeds, so action effects are paired against identical rollout noise.
No simulator outcome is used or claimed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Literal

import numpy as np

from openpi.weaver import b1k_contract
from openpi.weaver import behavior_precomputed
from openpi.weaver import candidate_pool
from openpi.weaver import causal_action_adapter
from openpi.weaver import hybrid_action

SCHEMA_VERSION = "b1k_weaver_candidate_stability_v1"


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_root: str
    numeric_dir: str
    output_json: str
    policy_config: str
    policy_dir: str
    weaver_config: str
    weaver_checkpoint: str
    task_encoder_model: str
    hybrid_action_contract: str
    norm_stats: str
    tasks: tuple[str, ...] = ()
    num_tasks: int = 12
    exclude_tasks: tuple[str, ...] = ("turn_on_radio",)
    split: Literal["train", "val", "test"] = "train"
    seed: int = 20260903
    samples_per_episode: int = 3
    num_candidates: int = 4
    scored_observations: int = 1
    rollout_seeds: tuple[int, ...] = (101, 202, 303, 404, 505)
    robot: str = "b1k/R1Pro"
    control_hz: float = 30.0
    target_hz: float = 5.0
    imagination_horizon: int = 5
    num_flow_steps: int = 10
    action_horizon: int = 6
    criterion: Literal["reward", "advantage"] = "advantage"
    checkpoint_mode: Literal["resume", "identity", "b1k_23d_fresh"] = "resume"
    resize_mode: Literal["letterbox", "center-crop", "stretch"] = "stretch"
    device: str = "cuda:0"
    use_ema: bool = True
    gripper_tolerance: float = 0.10


def gripper_distance_report(values: np.ndarray, *, tolerance: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"gripper values must be finite (N,2), got {array.shape}")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("gripper tolerance must be finite and non-negative")
    distance = np.minimum(np.abs(array - 1.0), np.abs(array + 1.0))

    def summarize(column: np.ndarray) -> dict[str, Any]:
        return {
            "count": int(column.size),
            "p50": float(np.quantile(column, 0.50)),
            "p95": float(np.quantile(column, 0.95)),
            "p99": float(np.quantile(column, 0.99)),
            "max": float(column.max()),
            "above_0p05_count": int(np.sum(column > 0.05)),
            "above_0p05_fraction": float(np.mean(column > 0.05)),
            "above_0p10_count": int(np.sum(column > 0.10)),
            "above_0p10_fraction": float(np.mean(column > 0.10)),
            "above_contract_tolerance_count": int(np.sum(column > tolerance)),
            "above_contract_tolerance_fraction": float(np.mean(column > tolerance)),
        }

    return {
        "tolerance": tolerance,
        "left": summarize(distance[:, 0]),
        "right": summarize(distance[:, 1]),
        "combined": summarize(distance.reshape(-1)),
    }


def common_noise_score_report(score_matrix: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) <= 0 or not np.isfinite(scores).all():
        raise ValueError(f"score matrix must be finite (seeds,candidates), got {scores.shape}")
    action_variance = float(np.mean(np.var(scores, axis=1)))
    noise_variance = float(np.mean(np.var(scores, axis=0)))
    signal_to_noise = None if noise_variance == 0 else math.sqrt(action_variance / noise_variance)
    winners = np.argmax(scores, axis=1)
    counts = np.bincount(winners, minlength=scores.shape[1])
    margins = (
        np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
        if scores.shape[1] > 1
        else np.zeros(scores.shape[0])
    )
    return {
        "shape": list(scores.shape),
        "scores": scores.tolist(),
        "candidate_mean": np.mean(scores, axis=0).tolist(),
        "candidate_std_across_noise": np.std(scores, axis=0).tolist(),
        "action_variance_under_common_noise": action_variance,
        "rollout_noise_variance": noise_variance,
        "action_to_noise_rms_ratio": signal_to_noise,
        "winner_indices": winners.tolist(),
        "winner_counts": counts.tolist(),
        "winner_agreement_fraction": float(counts.max() / len(winners)),
        "score_margin_mean": float(np.mean(margins)),
        "score_margin_min": float(np.min(margins)),
    }


def _sampled_ordinals(length: int, count: int) -> tuple[int, ...]:
    if length <= 0 or count <= 0:
        raise ValueError("episode length and samples_per_episode must be positive")
    if count > length:
        raise ValueError(f"Cannot choose {count} unique ordinals from episode length {length}")
    points = np.linspace(0, length - 1, count + 2)[1:-1]
    ordinals = tuple(dict.fromkeys(round(value) for value in points))
    if len(ordinals) != count:
        ordinals = tuple(np.linspace(0, length - 1, count, dtype=np.int64).tolist())
    if len(set(ordinals)) != count:
        raise ValueError("Could not construct unique deterministic sampled ordinals")
    return ordinals


def _subset(pool: candidate_pool.CandidatePool, index: int) -> candidate_pool.CandidatePool:
    noises = None if pool.noises is None else np.array(pool.noises[index : index + 1], copy=True)
    return candidate_pool.CandidatePool(
        actions=np.array(pool.actions[index : index + 1], copy=True),
        observation_digest=pool.observation_digest,
        noises=noises,
    )


def _reproducibility_record(first: Any, second: Any) -> dict[str, Any]:
    rewards_equal = np.array_equal(first.rewards, second.rewards)
    values_equal = first.values is None and second.values is None
    if first.values is not None and second.values is not None:
        values_equal = np.array_equal(first.values, second.values)
    scores_equal = np.array_equal(first.scores, second.scores)
    first_hashes = {key: value["future_sha256"] for key, value in first.rollout_diagnostics.items()}
    second_hashes = {key: value["future_sha256"] for key, value in second.rollout_diagnostics.items()}
    return {
        "exact": bool(rewards_equal and values_equal and scores_equal and first_hashes == second_hashes),
        "rewards_equal": bool(rewards_equal),
        "values_equal": bool(values_equal),
        "scores_equal": bool(scores_equal),
        "rollout_sha256_equal": first_hashes == second_hashes,
        "rollout_sha256": first_hashes,
    }


def _score_stability(
    policy: Any, observation: dict[str, Any], pool: candidate_pool.CandidatePool, seeds: Sequence[int]
) -> dict[str, Any]:
    first_seed = int(seeds[0])
    _, first, _ = policy.score_candidate_pool(observation, pool, rollout_seed=first_seed)
    _, repeated, _ = policy.score_candidate_pool(observation, pool, rollout_seed=first_seed)
    common_noise = np.empty((len(seeds), pool.num_candidates), dtype=np.float64)
    rollout_hashes: list[list[dict[str, str]]] = []
    for seed_index, rollout_seed in enumerate(seeds):
        seed_hashes = []
        for candidate_index in range(pool.num_candidates):
            _, scored, _ = policy.score_candidate_pool(
                observation,
                _subset(pool, candidate_index),
                rollout_seed=int(rollout_seed),
            )
            common_noise[seed_index, candidate_index] = float(scored.scores[0])
            seed_hashes.append({key: value["future_sha256"] for key, value in scored.rollout_diagnostics.items()})
        rollout_hashes.append(seed_hashes)
    reversed_scores = np.empty(pool.num_candidates, dtype=np.float64)
    for candidate_index in reversed(range(pool.num_candidates)):
        _, scored, _ = policy.score_candidate_pool(
            observation,
            _subset(pool, candidate_index),
            rollout_seed=first_seed,
        )
        reversed_scores[candidate_index] = float(scored.scores[0])
    order_invariant = np.array_equal(reversed_scores, common_noise[0])
    return {
        "batched_same_seed_reproducibility": _reproducibility_record(first, repeated),
        "common_noise": common_noise_score_report(common_noise),
        "singleton_reverse_order_exact": bool(order_invariant),
        "singleton_reverse_order_scores": reversed_scores.tolist(),
        "singleton_rollout_sha256_by_seed_candidate": rollout_hashes,
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite candidate audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(args: Args) -> None:
    from scripts.b1k import replay_weaver_pipeline  # noqa: PLC0415
    from scripts.b1k import serve_weaver_b1k  # noqa: PLC0415

    if args.samples_per_episode <= 0 or args.num_candidates < 2 or args.scored_observations < 0:
        raise ValueError("samples_per_episode must be positive, num_candidates >= 2, and scored_observations >= 0")
    if not args.rollout_seeds or len(set(args.rollout_seeds)) != len(args.rollout_seeds):
        raise ValueError("rollout_seeds must be non-empty and unique")
    started = time.perf_counter()
    numeric_root = Path(args.numeric_dir).expanduser().resolve()
    manifests = causal_action_adapter.read_manifest_index(numeric_root, split=args.split)
    selected = replay_weaver_pipeline.select_task_manifests(
        manifests,
        tasks=args.tasks,
        num_tasks=args.num_tasks,
        exclude_tasks=args.exclude_tasks,
        seed=args.seed,
    )
    prompts = serve_weaver_b1k.load_task_prompt_catalog(numeric_root)
    if prompts is None:
        raise FileNotFoundError("Candidate audit requires the authoritative task prompt catalog")
    repo_ids = {str(manifest["dataset"]["repo_id"]) for _, manifest in selected}
    if len(repo_ids) != 1:
        raise ValueError("Selected candidate audit manifests mix dataset repo IDs")
    contract = b1k_contract.build_r1pro_contract()
    video_keys = [contract.robot.observations[key].dataset_key for key in sorted(contract.robot.observations)]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    dataset = LeRobotDataset(
        repo_id=repo_ids.pop(),
        root=Path(args.dataset_root).expanduser().resolve(),
        video_keys=video_keys,
        return_uint8=True,
        tolerance_s=1e-4,
        video_backend="pyav",
    )
    first_task, first_manifest = selected[0]
    runtime = replay_weaver_pipeline.runtime_args(
        replay_weaver_pipeline.Args(
            dataset_root=args.dataset_root,
            numeric_dir=args.numeric_dir,
            output_json=args.output_json,
            policy_config=args.policy_config,
            policy_dir=args.policy_dir,
            weaver_config=args.weaver_config,
            weaver_checkpoint=args.weaver_checkpoint,
            task_encoder_model=args.task_encoder_model,
            hybrid_action_contract=args.hybrid_action_contract,
            norm_stats=args.norm_stats,
            num_candidates=args.num_candidates,
            robot=args.robot,
            control_hz=args.control_hz,
            target_hz=args.target_hz,
            imagination_horizon=args.imagination_horizon,
            num_flow_steps=args.num_flow_steps,
            action_horizon=args.action_horizon,
            criterion=args.criterion,
            checkpoint_mode=args.checkpoint_mode,
            resize_mode=args.resize_mode,
            device=args.device,
            use_ema=args.use_ema,
        ),
        task=first_task,
        prompt=str(first_manifest["prompt"]),
    )
    policy = serve_weaver_b1k.build_online_weaver_policy(runtime)
    grip_indices = hybrid_action.gripper_indices()
    gripper_values: list[np.ndarray] = []
    records = []
    scored_count = 0
    for task, manifest in selected:
        numeric = behavior_precomputed.load_numeric_episode(numeric_root, manifest)
        prompt = prompts.get(task)
        if prompt != str(manifest["prompt"]):
            raise ValueError(f"Prompt catalog mismatch for {task!r}")
        for ordinal in _sampled_ordinals(numeric.length, args.samples_per_episode):
            absolute_index = int(numeric.source_absolute_indices[ordinal])
            observation, observation_summary = replay_weaver_pipeline.build_online_observation(
                dataset[absolute_index],
                contract=contract,
                expected_episode_index=int(manifest["episode_index"]),
                expected_absolute_index=absolute_index,
            )
            policy.set_task_prompt(prompt)
            pool = policy.sample_candidate_pool(observation)
            gripper_values.append(np.array(pool.actions[..., list(grip_indices)].reshape(-1, 2), copy=True))
            stability = None
            if scored_count < args.scored_observations:
                policy.prime_replan_context(observation)
                stability = _score_stability(policy, observation, pool, args.rollout_seeds)
                scored_count += 1
            records.append(
                {
                    "task": task,
                    "episode_index": int(manifest["episode_index"]),
                    "sampled_ordinal": ordinal,
                    "source_absolute_index": absolute_index,
                    "observation": observation_summary,
                    "pool_actions_sha256": pool.actions_sha256,
                    "pool_shape": list(pool.actions.shape),
                    "candidate_pairwise_rms": float(
                        np.sqrt(
                            np.mean(np.square(pool.actions[0].astype(np.float64) - pool.actions[1].astype(np.float64)))
                        )
                    ),
                    "stability": stability,
                }
            )
            print(
                f"task={task} episode={manifest['episode_index']} ordinal={ordinal} "
                f"pool={pool.actions_sha256[:12]} scored={stability is not None}",
                flush=True,
            )
    gripper = gripper_distance_report(np.concatenate(gripper_values), tolerance=args.gripper_tolerance)
    stability_records = [record["stability"] for record in records if record["stability"] is not None]
    technical_pass = all(
        record["batched_same_seed_reproducibility"]["exact"] and record["singleton_reverse_order_exact"]
        for record in stability_records
    )
    tolerance_pass = gripper["combined"]["above_contract_tolerance_count"] == 0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if technical_pass and tolerance_pass else "failed",
        "scope": "simulator-free stochastic attribution and policy gripper contract only",
        "warning": "Scores are WEAVER predictions, not simulator outcomes or success-rate evidence.",
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "numeric_root": str(numeric_root),
        "policy_dir": str(Path(args.policy_dir).expanduser().resolve()),
        "weaver_checkpoint": str(Path(args.weaver_checkpoint).expanduser().resolve()),
        "seed": args.seed,
        "rollout_seeds": list(args.rollout_seeds),
        "tasks": [task for task, _ in selected],
        "sampled_observations": len(records),
        "scored_observations": len(stability_records),
        "gripper": gripper,
        "technical_attribution_gate": technical_pass,
        "gripper_tolerance_gate": tolerance_pass,
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {"python": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    output = Path(args.output_json).expanduser().resolve()
    _atomic_write(output, payload)
    print(f"CANDIDATE STABILITY AUDIT {payload['status'].upper()}: {output}", flush=True)
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main(__import__("tyro").cli(Args))
