"""Parallel coordinator-writer WEAVER preparation.

This entry point reuses the serial episode and transaction contracts while
moving episode preparation into isolated worker processes. Workers write only
run-scoped staging NPZs; the coordinator alone commits final artifacts and the
manifest index.
"""

# This entry point deliberately composes the serial transaction helpers instead
# of duplicating their artifact contract.
# ruff: noqa: PLW0603, SLF001

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from typing import Any

import tyro

try:
    from scripts.b1k import prepare_weaver_dataset as _serial
except ModuleNotFoundError:
    import prepare_weaver_dataset as _serial
from openpi.weaver import b1k_dataset_adapter as _adapter
from openpi.weaver import reward_contract as _reward_contract
from openpi.weaver.b1k_contract import build_r1pro_contract


@dataclasses.dataclass(frozen=True)
class Args(_serial.Args):
    workers: int = 4
    max_in_flight: int | None = None
    manifest_flush_every: int = 256
    worker_threads: int = 1


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    split: str
    source_count: int
    sampled_count: int
    raw_state_shape: tuple[int, ...]
    policy_action_shape: tuple[int, ...]
    weaver_state_shape: tuple[int, ...]
    weaver_action_shape: tuple[int, ...]
    max_dt_error: float
    manifest: dict[str, Any]
    staging_path: str
    staging_size_bytes: int
    staging_sha256: str
    timings: dict[str, float]


_WORKER_STATE: dict[str, Any] | None = None


def _set_worker_threads(count: int) -> None:
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(count)


def _worker_initializer(
    worker_threads: int,
    config_name: str,
    dataset_root: str,
    args: Args,
    shared_state: dict[str, Any],
) -> None:
    global _WORKER_STATE
    started = time.perf_counter()
    _set_worker_threads(worker_threads)
    # Do not inherit/pickle Arrow or LeRobot handles. Each spawned worker opens
    # its own numeric view once and then reuses it for all assigned episodes.
    _, data_factory, dataset, _ = _serial._prepare_dataset(
        _serial.Args(config_name=config_name, dataset_root=dataset_root)
    )
    if str(data_factory.repo_id) != str(shared_state["dataset_repo_id"]):
        raise ValueError(
            "Worker dataset repo_id disagrees with coordinator preflight: "
            f"worker={data_factory.repo_id!r}, coordinator={shared_state['dataset_repo_id']!r}"
        )
    if not math.isclose(float(dataset.fps), float(shared_state["source_fps"]), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Worker dataset fps disagrees with coordinator preflight: "
            f"worker={dataset.fps!r}, coordinator={shared_state['source_fps']!r}"
        )
    columns = tuple(shared_state["columns"])
    numeric_dataset = _serial._numeric_dataset_view(dataset, columns)
    _WORKER_STATE = dict(shared_state)
    _WORKER_STATE.update(
        {
            "args": args,
            "dataset": numeric_dataset,
            "columns": columns,
            "contract": build_r1pro_contract(),
            "worker_initialization_seconds": time.perf_counter() - started,
            "report_worker_initialization": True,
        }
    )


def _prepare_episode(episode_index: int) -> EpisodeResult:
    if _WORKER_STATE is None:
        raise RuntimeError("Parallel WEAVER worker was not initialized")
    state = _WORKER_STATE
    args = state["args"]
    payload = _serial._prepare_episode_payload(
        dataset=state["dataset"],
        episode_index=episode_index,
        episode_row=state["all_episode_rows"][episode_index],
        columns=state["columns"],
        prompt=state["episode_prompts"][episode_index],
        task_names=state["episode_tasks"][episode_index],
        split=state["splits"][episode_index],
        dataset_repo_id=str(state["dataset_repo_id"]),
        dataset_metadata_sha256=state["metadata_sha256"],
        dataset_provenance=state["dataset_provenance"],
        source_fps=float(state["source_fps"]),
        target_hz=args.target_hz,
        min_sampled_frames=args.min_sampled_frames,
        contract=state["contract"],
    )
    manifest = payload["manifest"]
    arrays = payload["arrays"]
    timings = dict(payload["timings"])
    if state.pop("report_worker_initialization", False):
        timings["worker_initialization_seconds"] = float(state["worker_initialization_seconds"])
    split = state["splits"][episode_index]

    staging_path = ""
    staging_size_bytes = 0
    staging_sha256 = ""
    if args.dry_run:
        # Dry runs benchmark read/adaptation only, not compression or I/O.
        timings["npz_compress_seconds"] = 0.0
    else:
        started = time.perf_counter()
        staging_root = Path(state["staging_root"])
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_path_obj = staging_root / f"{os.getpid()}-{episode_index:06d}.npz"
        write_started = time.perf_counter()
        _serial._atomic_write_npz(staging_path_obj, arrays)
        timings["npz_compress_fsync_seconds"] = time.perf_counter() - started
        timings["staging_write_seconds"] = time.perf_counter() - write_started
        staging_path = str(staging_path_obj)
        staging_size_bytes = staging_path_obj.stat().st_size
        hash_started = time.perf_counter()
        staging_sha256 = _serial._file_sha256(staging_path_obj)
        timings["staging_sha256_seconds"] = time.perf_counter() - hash_started
        manifest["numeric_artifact"].update({"size_bytes": staging_size_bytes, "sha256": staging_sha256})
    return EpisodeResult(
        episode_index=episode_index, split=split, source_count=payload["source_end"] - payload["source_start"],
        sampled_count=payload["sampled_count"], raw_state_shape=payload["raw_state_shape"],
        policy_action_shape=payload["policy_action_shape"], weaver_state_shape=payload["weaver_state_shape"],
        weaver_action_shape=payload["weaver_action_shape"],
        max_dt_error=payload["max_dt_error"], manifest=manifest,
        staging_path=staging_path, staging_size_bytes=staging_size_bytes,
        staging_sha256=staging_sha256, timings=timings,
    )


def _commit_result(
    result: EpisodeResult,
    *, output_root: Path,
    existing_pairs: dict[int, tuple[dict[str, Any], Path, Path]],
) -> tuple[dict[str, Any], str, dict[str, float]]:
    timings = dict(result.timings)
    manifest = result.manifest
    relative_npz = Path(manifest["numeric_artifact"]["path"])
    npz_path = output_root / relative_npz
    json_path = npz_path.with_suffix(".json")
    existing = existing_pairs.get(result.episode_index)
    staging_path = Path(result.staging_path) if result.staging_path else None
    if existing is not None:
        started = time.perf_counter()
        if staging_path is None or not staging_path.is_file():
            raise _serial.ArtifactResumeError(f"Worker staging artifact is missing: {staging_path}")
        staging_arrays = _serial._load_npz_arrays(staging_path)
        persisted = _serial._validate_existing_pair(
            output_root, existing[1], existing[2], expected_manifest=manifest, expected_arrays=staging_arrays
        )
        timings["existing_validation_seconds"] = time.perf_counter() - started
        staging_path.unlink(missing_ok=True)
        return persisted, "reused", timings
    if npz_path.exists() or json_path.exists():
        started = time.perf_counter()
        if staging_path is None or not staging_path.is_file():
            raise _serial.ArtifactResumeError(f"Worker staging artifact is missing: {staging_path}")
        staging_arrays = _serial._load_npz_arrays(staging_path)
        persisted, status = _serial._resolve_existing_pair(
            output_root, json_path, npz_path, expected_manifest=manifest, expected_arrays=staging_arrays
        )
        timings["existing_validation_seconds"] = time.perf_counter() - started
        staging_path.unlink(missing_ok=True)
        existing_pairs[result.episode_index] = (persisted, json_path, npz_path)
        return persisted, status, timings

    if staging_path is None or not staging_path.is_file():
        raise _serial.ArtifactResumeError(f"Worker staging artifact is missing: {staging_path}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    commit_started = time.perf_counter()
    os.replace(staging_path, npz_path)
    timings["npz_commit_seconds"] = time.perf_counter() - commit_started
    try:
        started = time.perf_counter()
        _serial._atomic_write_json(json_path, manifest)
        timings["json_write_seconds"] = time.perf_counter() - started
    except BaseException:
        # Keep strict resume fail-closed if a normal exception happens after NPZ.
        _serial._discard_partial_write(npz_path)
        raise
    existing_pairs[result.episode_index] = (manifest, json_path, npz_path)
    return manifest, "written", timings


def _validate_args(args: Args) -> None:
    if args.workers <= 0:
        raise ValueError(f"workers must be positive, got {args.workers}")
    if args.worker_threads <= 0:
        raise ValueError(f"worker_threads must be positive, got {args.worker_threads}")
    if args.manifest_flush_every <= 0:
        raise ValueError(f"manifest_flush_every must be positive, got {args.manifest_flush_every}")
    if args.max_in_flight is not None and args.max_in_flight < args.workers:
        raise ValueError("max_in_flight must be at least workers")
    if os.name != "posix" or "spawn" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("Parallel WEAVER preparation currently requires Linux multiprocessing spawn")


def _build_run_state(args: Args) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    config, data_factory, dataset, dataset_root = _serial._prepare_dataset(args)
    contract = build_r1pro_contract()
    selected_features = _adapter.validate_selected_video_features(
        dataset.features, contract
    )
    required = {
        "index", "frame_index", "timestamp", "episode_index", "observation.state",
        contract.robot.action_key, "next.reward", "next.terminated", "next.truncated",
    }
    available = set(getattr(dataset.hf_dataset, "column_names", dataset.features))
    if missing := required - available:
        raise KeyError(f"B1K dataset is missing required WEAVER numeric features: {sorted(missing)}")
    metadata_sha256 = _serial._metadata_sha256(dataset_root)
    if args.expected_metadata_sha256 is not None and metadata_sha256 != args.expected_metadata_sha256:
        raise ValueError(
            "Dataset metadata SHA-256 does not match --expected-metadata-sha256: "
            f"expected={args.expected_metadata_sha256}, actual={metadata_sha256}"
        )
    if not math.isclose(float(dataset.fps), 30.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"B1K WEAVER source contract requires 30 Hz data, got fps={dataset.fps!r}")
    task_names_by_index, task_prompts = _serial._task_catalog(dataset.meta, dataset_root)
    dataset_provenance = _serial._dataset_preflight(
        dataset_root, dataset, task_names_by_index=task_names_by_index, task_prompts=task_prompts
    )
    all_episode_rows: dict[int, dict[str, Any]] = {}
    episode_tasks: dict[int, tuple[str, ...]] = {}
    episode_prompts: dict[int, str] = {}
    for position in range(len(dataset.meta.episodes)):
        row = dict(dataset.meta.episodes[position])
        index = int(_serial._scalar(row["episode_index"]))
        if index in all_episode_rows:
            raise ValueError(f"Duplicate episode_index in metadata: {index}")
        all_episode_rows[index] = row
        episode_tasks[index] = _serial._episode_tasks(row, task_names_by_index=task_names_by_index)
        if len(episode_tasks[index]) != 1:
            raise ValueError(f"Episode {index} must resolve to exactly one task")
        episode_prompts[index] = task_prompts[episode_tasks[index][0]]
    held_out_tasks = _serial._deterministic_held_out_tasks(
        task_prompts, explicit=args.held_out_tasks, fraction=args.held_out_task_fraction, seed=args.split_seed
    )
    splits = _adapter.deterministic_episode_split(
        episode_tasks, val_fraction=args.val_fraction, seed=args.split_seed, held_out_tasks=held_out_tasks
    )
    selected = sorted(all_episode_rows)
    if args.episode_indices is not None and args.sample_tasks is not None:
        raise ValueError("episode_indices and sample_tasks are mutually exclusive")
    if args.episode_indices is not None:
        requested = set(args.episode_indices)
        if missing := requested - set(selected):
            raise ValueError(f"Unknown episode indices: {sorted(missing)}")
        selected = [index for index in selected if index in requested]
    elif args.sample_tasks is not None:
        selected = list(
            _adapter.deterministic_task_episode_sample(
                episode_tasks, num_tasks=args.sample_tasks, episodes_per_task=args.episodes_per_task,
                seed=args.selection_seed,
            )
        )
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError(f"max_episodes must be positive, got {args.max_episodes}")
        selected = selected[: args.max_episodes]
    expected_views = {
        view.weaver_key: {
            "rgb": view.dataset_rgb_key,
            "depth": None,
        }
        for view in contract.camera_views
    }
    state = {
        "dataset_root": dataset_root, "dataset_repo_id": str(data_factory.repo_id),
        "source_fps": float(dataset.fps),
        "columns": (
            "index", "frame_index", "timestamp", "episode_index", "observation.state",
            contract.robot.action_key, "next.reward", "next.terminated", "next.truncated",
        ),
        "metadata_sha256": metadata_sha256, "dataset_provenance": dataset_provenance,
        "all_episode_rows": all_episode_rows, "episode_tasks": episode_tasks,
        "episode_prompts": episode_prompts, "splits": splits,
    }
    metadata = {
        "config": config.name, "task_prompts": task_prompts, "held_out_tasks": held_out_tasks,
        "selected_features": selected_features, "expected_views": expected_views,
    }
    return state, selected, metadata


def _worker_shared_state(state: dict[str, Any], *, staging_root: Path) -> dict[str, Any]:
    """Return the picklable, read-only contract sent to spawned workers."""

    return {
        "dataset_repo_id": state["dataset_repo_id"],
        "source_fps": state["source_fps"],
        "columns": state["columns"],
        "metadata_sha256": state["metadata_sha256"],
        "dataset_provenance": state["dataset_provenance"],
        "all_episode_rows": state["all_episode_rows"],
        "episode_tasks": state["episode_tasks"],
        "episode_prompts": state["episode_prompts"],
        "splits": state["splits"],
        "staging_root": str(staging_root),
    }


def _worker_initializer_args(
    args: Args,
    state: dict[str, Any],
    worker_state: dict[str, Any],
) -> tuple[int, str, str, Args, dict[str, Any]]:
    """Keep the process-pool initializer contract explicit and testable."""

    return (
        args.worker_threads,
        args.config_name,
        str(state["dataset_root"]),
        args,
        worker_state,
    )


def _main_unlocked(args: Args) -> None:
    _validate_args(args)
    run_started = time.perf_counter()
    state, selected_indices, metadata = _build_run_state(args)
    output_root = Path(args.output_dir).expanduser().resolve()
    staging_root = output_root / ".staging" / f"run-{os.getpid()}-{time.time_ns()}"
    if not args.dry_run:
        staging_root.mkdir(parents=True, exist_ok=False)
    worker_state = _worker_shared_state(state, staging_root=staging_root)
    print(json.dumps({
        "mode": "parallel-single-writer-v1", "config": metadata["config"],
        "dataset_root": str(state["dataset_root"]), "repo_id": state["dataset_repo_id"],
        "metadata_sha256": state["metadata_sha256"], "source_fps": state["source_fps"],
        "selected_episodes": len(selected_indices), "workers": args.workers,
        "max_in_flight": args.max_in_flight or args.workers * 2,
        "worker_threads": args.worker_threads, "manifest_flush_every": args.manifest_flush_every,
        "output_dir": str(output_root), "dry_run": args.dry_run,
    }, indent=2), flush=True)

    existing_pairs = {} if args.dry_run else _serial._discover_existing_pairs(output_root)
    for manifest, _, _ in existing_pairs.values():
        _serial._validate_existing_context(
            manifest, dataset_repo_id=str(state["dataset_repo_id"]),
            dataset_metadata_sha256=state["metadata_sha256"], target_hz=args.target_hz,
            expected_views=metadata["expected_views"], source_fps=state["source_fps"],
            all_episode_rows=state["all_episode_rows"], episode_tasks=state["episode_tasks"],
            episode_prompts=state["episode_prompts"], splits=state["splits"],
            dataset_provenance=state["dataset_provenance"],
        )
    manifest_index = {index: pair[0] for index, pair in existing_pairs.items()}
    selected_count = len(selected_indices)
    selected_indices, manifests = _serial._partition_selected_for_resume(selected_indices, existing_pairs)
    reused_existing_count = len(manifests)
    print(json.dumps({
        "resume_reused_episodes": reused_existing_count,
        "episodes_to_prepare": len(selected_indices),
    }), flush=True)
    rejected: list[dict[str, Any]] = []
    aggregate_timings: dict[str, float] = {}
    dirty_since_flush = 0
    processing_started = time.perf_counter()
    max_in_flight = args.max_in_flight or args.workers * 2

    def flush_manifest() -> None:
        nonlocal dirty_since_flush
        if args.dry_run or dirty_since_flush == 0:
            return
        started = time.perf_counter()
        _serial._atomic_write_manifest_index(output_root / "manifest.jsonl", manifest_index)
        aggregate_timings["manifest_checkpoint_seconds"] = aggregate_timings.get(
            "manifest_checkpoint_seconds", 0.0
        ) + (time.perf_counter() - started)
        dirty_since_flush = 0

    _set_worker_threads(args.worker_threads)
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_initializer,
        initargs=_worker_initializer_args(args, state, worker_state),
    )
    pending: dict[concurrent.futures.Future[EpisodeResult], int] = {}
    iterator = iter(selected_indices)
    completed = 0
    try:
        while len(pending) < max_in_flight:
            try:
                index = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(_prepare_episode, index)] = index
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    result = future.result()
                    if args.dry_run:
                        persisted, status, timings = result.manifest, "dry-run", result.timings
                    else:
                        persisted, status, timings = _commit_result(
                            result, output_root=output_root, existing_pairs=existing_pairs
                        )
                        manifest_index[index] = persisted
                        dirty_since_flush += 1
                    manifests.append(persisted)
                    completed += 1
                    for name, seconds in timings.items():
                        aggregate_timings[name] = aggregate_timings.get(name, 0.0) + seconds
                    if dirty_since_flush >= args.manifest_flush_every:
                        flush_manifest()
                    print(
                        f"[{completed}/{len(selected_indices)}] episode={index} split={result.split} "
                        f"artifact={status} source={result.source_count} sampled={result.sampled_count} "
                        f"raw_state={result.raw_state_shape} policy_action={result.policy_action_shape} "
                        f"weaver_state={result.weaver_state_shape} weaver_action={result.weaver_action_shape} "
                        f"max_dt_error={result.max_dt_error:.6f}s timings={json.dumps(timings, sort_keys=True)}",
                        flush=True,
                    )
                except _serial.ArtifactResumeError:
                    raise
                except Exception as exc:
                    failure = {"episode_index": index, "error_type": type(exc).__name__, "error": str(exc)}
                    rejected.append(failure)
                    print(json.dumps({"rejected": failure}, ensure_ascii=False), flush=True)
                    if args.fail_fast:
                        raise
                try:
                    next_index = next(iterator)
                except StopIteration:
                    continue
                pending[executor.submit(_prepare_episode, next_index)] = next_index
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        flush_manifest()
        if not args.dry_run:
            shutil.rmtree(staging_root, ignore_errors=True)

    wall_seconds = time.perf_counter() - run_started
    processing_seconds = time.perf_counter() - processing_started
    processed_source_rows = sum(int(manifest["source"]["full_rate_count"]) for manifest in manifests)
    newly_prepared = manifests[reused_existing_count:]
    newly_processed_source_rows = sum(int(manifest["source"]["full_rate_count"]) for manifest in newly_prepared)
    summary = {
        "schema_version": _adapter.SCHEMA_VERSION, "preparation_runtime": "parallel-single-writer-v1",
        "dataset_repo_id": state["dataset_repo_id"], "dataset_metadata_sha256": state["metadata_sha256"],
        "dataset_provenance": state["dataset_provenance"],
        "task_catalog": {
            "schema_version": "behavior_tasks_v1",
            "sha256": state["dataset_provenance"]["task_catalog_sha256"],
            "count": len(metadata["task_prompts"]),
        },
        "held_out_tasks": list(metadata["held_out_tasks"]), "target_hz": args.target_hz,
        "selected_video_features": list(metadata["selected_features"]), "processed": len(manifests),
        "processed_source_rows": processed_source_rows,
        "resume": {
            "selected": selected_count,
            "reused_existing": reused_existing_count,
            "newly_prepared": len(newly_prepared),
            "remaining_after_run": selected_count - len(manifests),
        },
        "rejected": rejected,
        "split_counts": {
            split: sum(manifest["split"] == split for manifest in manifests) for split in ("train", "val", "test")
        },
        "parallelism": {
            "workers": args.workers, "worker_threads": args.worker_threads,
            "max_in_flight": max_in_flight, "manifest_flush_every": args.manifest_flush_every,
        },
        "timings": {
            "wall_seconds": wall_seconds,
            "setup_seconds": processing_started - run_started,
            "processing_seconds": processing_seconds,
            "episodes_per_minute": 60.0 * len(newly_prepared) / wall_seconds if wall_seconds else 0.0,
            "source_rows_per_second": newly_processed_source_rows / wall_seconds if wall_seconds else 0.0,
            "stage_totals_seconds": aggregate_timings,
        },
        "latent_complete": False, "reward_export": dict(_reward_contract.REWARD_CONTRACT),
        "online_steering": "disabled_offline_state_delta_is_noncausal",
    }
    if not args.dry_run:
        _serial._atomic_write_json(
            output_root / "task_prompts.json",
            {
                "schema_version": "behavior_task_prompts_v1",
                "dataset_metadata_sha256": state["metadata_sha256"],
                "task_catalog_sha256": state["dataset_provenance"]["task_catalog_sha256"],
                "prompts": {name: metadata["task_prompts"][name] for name in sorted(metadata["task_prompts"])},
            },
        )
        _serial._atomic_write_json(output_root / "summary.json", summary)
        _serial._atomic_write_manifest_index(output_root / "manifest.jsonl", manifest_index)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main(args: Args) -> None:
    _validate_args(args)
    if args.dry_run:
        _main_unlocked(args)
        return
    output_root = Path(args.output_dir).expanduser().resolve()
    with _serial._output_run_lock(output_root):
        _main_unlocked(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
