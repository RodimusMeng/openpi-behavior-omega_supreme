"""Run the real OpenPI -> hybrid adapter -> WEAVER selector without OmniGibson.

The replay source is one exact full-resolution LeRobot row per selected task.
No recorded future state or reward is exposed to the selector.  This is a
plumbing and provenance gate, not a counterfactual success-rate evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Literal

import numpy as np
import torch
import tyro

from openpi.weaver import b1k_contract
from openpi.weaver import b1k_dataset_adapter
from openpi.weaver import behavior_precomputed
from openpi.weaver import causal_action_adapter
from scripts.b1k import serve_weaver_b1k

SCHEMA_VERSION = "b1k_weaver_simulator_free_replay_v1"


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
    num_tasks: int = 3
    exclude_tasks: tuple[str, ...] = ("turn_on_radio",)
    split: Literal["train", "val", "test"] = "train"
    seed: int = 0
    sampled_ordinal: int | None = None
    robot: str = "b1k/R1Pro"
    num_candidates: int = 2
    control_hz: float = 30.0
    target_hz: float = 5.0
    imagination_horizon: int = 5
    num_flow_steps: int = 10
    action_horizon: int = 6
    criterion: Literal["reward", "advantage"] = "advantage"
    checkpoint_mode: Literal["resume", "identity", "b1k_23d_fresh"] = "resume"
    resize_mode: Literal["letterbox", "center-crop", "stretch"] = "stretch"
    device: str = "cuda:1"
    use_ema: bool = True
    require_distinct_candidates: bool = True
    require_distinct_rollouts: bool = True


def _normalized_task_name(value: Any) -> str:
    name = str(value).strip()
    if name.startswith("b1k/"):
        name = name.split("/", 1)[1].strip()
    if not name or "/" in name:
        raise ValueError(f"Invalid B1K task name: {value!r}")
    return name


def _manifest_task_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw = manifest.get("task_names")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError(f"Episode {manifest.get('episode_index')} has no task_names")
    names = tuple(_normalized_task_name(value) for value in raw)
    if len(names) != len(set(names)):
        raise ValueError(f"Episode {manifest.get('episode_index')} repeats a task name")
    return names


def select_task_manifests(
    manifests: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    num_tasks: int,
    exclude_tasks: Sequence[str],
    seed: int,
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Choose one episode per task with deterministic episode-level sampling."""

    if isinstance(num_tasks, bool) or not isinstance(num_tasks, int) or num_tasks <= 0:
        raise ValueError("num_tasks must be a positive integer")
    excluded = {_normalized_task_name(value) for value in exclude_tasks}
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for manifest in manifests:
        b1k_dataset_adapter.verify_episode_manifest(manifest)
        manifest_tasks = _manifest_task_names(manifest)
        if len(manifest_tasks) != 1:
            continue
        by_task.setdefault(manifest_tasks[0], []).append(manifest)
    available = sorted(set(by_task) - excluded)
    if tasks:
        selected_tasks = tuple(_normalized_task_name(value) for value in tasks)
        if len(selected_tasks) != len(set(selected_tasks)):
            raise ValueError("Explicit replay tasks must be unique")
        if rejected := sorted(set(selected_tasks) & excluded):
            raise ValueError(f"Explicit replay tasks are excluded: {rejected}")
        if missing := sorted(set(selected_tasks) - set(available)):
            raise ValueError(f"Replay tasks are absent from the selected split: {missing}")
    else:
        if len(available) < num_tasks:
            raise ValueError(f"Need {num_tasks} replay tasks after exclusions, found {len(available)}")
        selected_tasks = tuple(random.Random(seed).sample(available, num_tasks))
    rng = random.Random(seed + 1)
    result = []
    for task_name in selected_tasks:
        candidates = sorted(by_task[task_name], key=lambda item: int(item["episode_index"]))
        result.append((task_name, candidates[rng.randrange(len(candidates))]))
    return tuple(result)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _scalar_int(value: Any, *, name: str) -> int:
    array = _as_numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    return int(array.reshape(-1)[0])


def _as_hwc_uint8(value: Any, *, name: str) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must contain one RGB image, got {array.shape}")
    if array.shape[-1] == 3:
        pass
    elif array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(f"{name} has no three-channel RGB axis: {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"{name} must be exact uint8 pixels, got {array.dtype}")
    return np.ascontiguousarray(array)


def build_online_observation(
    row: Mapping[str, Any],
    *,
    contract: b1k_contract.B1KWeaverContract,
    expected_episode_index: int,
    expected_absolute_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Map one exact LeRobot row back to the official evaluator key space."""

    actual_episode = _scalar_int(row["episode_index"], name="episode_index")
    actual_index = _scalar_int(row["index"], name="index")
    if actual_episode != expected_episode_index or actual_index != expected_absolute_index:
        raise ValueError(
            "LeRobot replay row identity changed: "
            f"expected episode/index={expected_episode_index}/{expected_absolute_index}, "
            f"got {actual_episode}/{actual_index}"
        )
    raw_state = _as_numpy(row["observation.state"])
    if raw_state.ndim != 1 or not np.isfinite(raw_state).all():
        raise ValueError(f"Replay proprio must be one finite vector, got {raw_state.shape}")
    contract.extract_state(raw_state)
    observation: dict[str, np.ndarray] = {f"{contract.robot.name}::proprio": np.array(raw_state, copy=True)}
    image_shapes: dict[str, list[int]] = {}
    for camera_key in sorted(contract.robot.observations):
        camera = contract.robot.observations[camera_key]
        image = _as_hwc_uint8(row[camera.dataset_key], name=camera.dataset_key)
        observation[camera.obs_key] = image
        image_shapes[camera.obs_key] = list(image.shape)
    return observation, {
        "raw_state_shape": list(raw_state.shape),
        "raw_state_dtype": str(raw_state.dtype),
        "image_shapes": image_shapes,
        "observation_keys": sorted(observation),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _device_telemetry(device: str) -> dict[str, Any]:
    if not str(device).startswith("cuda"):
        return {"device": device}
    index = torch.device(device)
    torch.cuda.synchronize(index)
    return {
        "device": str(index),
        "gpu_name": torch.cuda.get_device_name(index),
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def runtime_args(args: Args, *, task: str, prompt: str) -> serve_weaver_b1k.Args:
    return serve_weaver_b1k.Args(
        robot=args.robot,
        task=f"b1k/{task}",
        prompt=prompt,
        policy_config=args.policy_config,
        policy_dir=args.policy_dir,
        weaver_config=args.weaver_config,
        weaver_checkpoint=args.weaver_checkpoint,
        task_encoder_model=args.task_encoder_model,
        hybrid_action_contract=args.hybrid_action_contract,
        norm_stats=args.norm_stats,
        dataset_dir=args.numeric_dir,
        num_candidates=args.num_candidates,
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
    )


def main(args: Args) -> None:
    script_started = time.perf_counter()
    if args.num_candidates < 2 and (args.require_distinct_candidates or args.require_distinct_rollouts):
        raise ValueError("Distinctness gates require at least two candidates")
    numeric_root = Path(args.numeric_dir).expanduser().resolve()
    manifests = causal_action_adapter.read_manifest_index(numeric_root, split=args.split)
    selected = select_task_manifests(
        manifests,
        tasks=args.tasks,
        num_tasks=args.num_tasks,
        exclude_tasks=args.exclude_tasks,
        seed=args.seed,
    )
    prompt_catalog = serve_weaver_b1k.load_task_prompt_catalog(numeric_root)
    if prompt_catalog is None:
        raise FileNotFoundError(f"Simulator-free replay requires the authoritative task prompt catalog: {numeric_root}")
    for task_name, manifest in selected:
        expected_prompt = prompt_catalog.get(task_name)
        if expected_prompt is None or expected_prompt != str(manifest["prompt"]):
            raise ValueError(
                f"Episode {manifest['episode_index']} prompt does not match the authoritative catalog for {task_name!r}"
            )
    print(
        json.dumps(
            {
                "selected_replays": [
                    {"task": task_name, "episode_index": int(manifest["episode_index"])}
                    for task_name, manifest in selected
                ]
            },
            indent=2,
        ),
        flush=True,
    )
    dataset_repo_ids = {str(manifest["dataset"]["repo_id"]) for _, manifest in selected}
    if len(dataset_repo_ids) != 1:
        raise ValueError(f"Selected replay episodes mix dataset repo IDs: {sorted(dataset_repo_ids)}")
    contract = b1k_contract.build_r1pro_contract()
    all_video_keys = [contract.robot.observations[key].dataset_key for key in sorted(contract.robot.observations)]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    dataset_started = time.perf_counter()
    dataset = LeRobotDataset(
        repo_id=dataset_repo_ids.pop(),
        root=Path(args.dataset_root).expanduser().resolve(),
        video_keys=all_video_keys,
        return_uint8=True,
        tolerance_s=1e-4,
        video_backend="pyav",
    )
    dataset_load_seconds = time.perf_counter() - dataset_started
    first_task, first_manifest = selected[0]
    server_args = runtime_args(args, task=first_task, prompt=str(first_manifest["prompt"]))
    runtime_started = time.perf_counter()
    policy = serve_weaver_b1k.build_online_weaver_policy(server_args)
    runtime_load_seconds = time.perf_counter() - runtime_started
    results: list[dict[str, Any]] = []
    for replay_index, (task_name, manifest) in enumerate(selected):
        numeric = behavior_precomputed.load_numeric_episode(numeric_root, manifest)
        ordinal = numeric.length // 2 if args.sampled_ordinal is None else args.sampled_ordinal
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 0 <= ordinal < numeric.length:
            raise ValueError(
                f"sampled_ordinal must be in [0,{numeric.length - 1}] for episode {manifest['episode_index']}"
            )
        absolute_index = int(numeric.source_absolute_indices[ordinal])
        observation, observation_summary = build_online_observation(
            dataset[absolute_index],
            contract=contract,
            expected_episode_index=int(manifest["episode_index"]),
            expected_absolute_index=absolute_index,
        )
        policy.set_task_prompt(str(manifest["prompt"]))
        torch.manual_seed(args.seed + replay_index)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        replay_started = time.perf_counter()
        action = _as_numpy(policy.act(observation)).astype(np.float32, copy=True)
        elapsed = time.perf_counter() - replay_started
        diagnostics = policy.last_replan
        if diagnostics is None:
            raise RuntimeError("Replay did not trigger a WEAVER replan")
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError(f"Replay must return one finite 23-D controller action, got {action.shape}")
        selected_first = np.asarray(diagnostics["selected_action_first"], dtype=np.float32)
        if not np.array_equal(action, selected_first):
            raise ValueError("Returned controller action is not the first action of the selected immutable candidate")
        if args.require_distinct_candidates and diagnostics["candidate_pairwise_rms_max"] <= 0:
            raise ValueError(f"Task {task_name!r} produced an identical candidate pool")
        rollout = diagnostics["rollout"]
        rollout_distinct = any(value["candidate_pairwise_rms_max"] > 0 for value in rollout.values())
        if args.require_distinct_rollouts and not rollout_distinct:
            raise ValueError(f"Task {task_name!r} produced identical imagined rollouts")
        result = {
            "task": task_name,
            "prompt": str(manifest["prompt"]),
            "split": args.split,
            "episode_index": int(manifest["episode_index"]),
            "sampled_ordinal": ordinal,
            "source_frame_index": int(numeric.source_frame_indices[ordinal]),
            "source_absolute_index": absolute_index,
            "source_timestamp_s": float(numeric.timestamps_s[ordinal]),
            "numeric_episode_semantic_sha256": str(manifest["semantic_sha256"]),
            "observation": observation_summary,
            "selected_action": action.tolist(),
            "selected_action_shape": list(action.shape),
            "selected_action_dtype": str(action.dtype),
            "replan": diagnostics,
            "rollout_distinct": rollout_distinct,
            "elapsed_seconds": elapsed,
            "torch_device": _device_telemetry(args.device),
        }
        results.append(result)
        print(
            f"[{replay_index + 1}/{len(selected)}] task={task_name} episode={manifest['episode_index']} "
            f"candidate={diagnostics['selected_candidate_id']} scores={diagnostics['scores']} "
            f"elapsed={elapsed:.3f}s",
            flush=True,
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "warning": (
            "Simulator-free plumbing gate only. RM/critic scores are not counterfactual outcomes or success-rate evidence."
        ),
        "seed": args.seed,
        "tasks": [task for task, _ in selected],
        "excluded_tasks": sorted({_normalized_task_name(value) for value in args.exclude_tasks}),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "numeric_root": str(numeric_root),
        "policy_config": args.policy_config,
        "policy_dir": str(Path(args.policy_dir).expanduser().resolve()),
        "weaver_checkpoint": str(Path(args.weaver_checkpoint).expanduser().resolve()),
        "hybrid_action_contract": str(Path(args.hybrid_action_contract).expanduser().resolve()),
        "norm_stats": str(Path(args.norm_stats).expanduser().resolve()),
        "resize_mode": args.resize_mode,
        "environment": {
            "python": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "xla_python_client_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        },
        "timing_seconds": {
            "dataset_load": dataset_load_seconds,
            "combined_runtime_load": runtime_load_seconds,
            "total": time.perf_counter() - script_started,
        },
        "records": results,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    _atomic_write_json(output_path, payload)
    print(f"SIMULATOR-FREE REPLAY PASSED: {output_path}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
