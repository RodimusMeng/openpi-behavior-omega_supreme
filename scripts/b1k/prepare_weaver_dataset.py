"""Prepare episode-level numeric artifacts from the full B1K LeRobot dataset.

This is the first half of the WEAVER dataset path. It does not encode SD3
latents. B1K ``next.reward`` is a 30 Hz transition-aligned BehaviorTask
predicate-progress delta; it is summed over every half-open source-row interval
between two 5 Hz frames. Endpoint sampling is never used. Each output episode
contains the complete numeric training contract and enough provenance for later
caches to reject stale or incompatible artifacts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from openpi.training import config as _config
from openpi.weaver import b1k_dataset_adapter as _adapter
from openpi.weaver import reward_contract as _reward_contract
from openpi.weaver.b1k_contract import build_r1pro_contract

NUMERIC_ARRAY_KEYS = frozenset(
    {
        "index",
        "frame_index",
        "source_timestamps_s",
        "source_policy_actions",
        "source_rewards",
        "source_terminated",
        "source_truncated",
        "source_frame_indices",
        "source_absolute_indices",
        "timestamps_s",
        "timing_errors_s",
        "states",
        "state_delta_actions",
        "policy_actions_at_frame",
        "transition_valid",
        "rewards",
    }
)


class ArtifactResumeError(RuntimeError):
    """An existing artifact cannot be proved complete and current."""


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi05_b1k_full"
    dataset_root: str | None = None
    output_dir: str = "outputs/weaver/b1k-v3again"
    episode_indices: tuple[int, ...] | None = None
    sample_tasks: int | None = None
    episodes_per_task: int = 1
    selection_seed: int = 2026
    max_episodes: int | None = None
    target_hz: float = 5.0
    min_sampled_frames: int = 10
    val_fraction: float = 0.1
    split_seed: int = 42
    held_out_tasks: tuple[str, ...] = ()
    include_depth: bool = False
    dry_run: bool = False
    fail_fast: bool = True


def _scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar metadata value, got shape {array.shape}")
    return array.reshape(-1)[0].item()


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, list) and value and hasattr(value[0], "detach"):
        return np.stack([item.detach().cpu().numpy() for item in value])
    array = np.asarray(value)
    if array.dtype == object and isinstance(value, list):
        try:
            array = np.stack([np.asarray(item) for item in value])
        except ValueError as exc:
            raise ValueError(f"Could not stack dataset column {name!r}") from exc
    return array


def _normalize_task_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        tasks = (value,)
    elif value is None:
        tasks = ()
    else:
        tasks = tuple(str(task) for task in value)
    return tuple(task.strip() for task in tasks if task.strip())


def _episode_tasks(meta: Any, episode: dict[str, Any]) -> tuple[str, ...]:
    tasks = _normalize_task_list(episode.get("tasks"))
    if tasks:
        return tasks

    if "task_index" not in episode:
        raise ValueError(f"Episode {episode.get('episode_index')} has neither tasks nor task_index metadata")
    task_indices = np.asarray(episode["task_index"]).reshape(-1)
    resolved: list[str] = []
    for task_index in task_indices:
        matches = meta.tasks[meta.tasks["task_index"] == int(task_index)]
        if len(matches) != 1:
            raise ValueError(f"Could not resolve task_index={task_index} for episode {episode.get('episode_index')}")
        resolved.append(str(matches.index[0]))
    return _normalize_task_list(resolved)


def _metadata_sha256(dataset_root: Path) -> str:
    meta_root = dataset_root / "meta"
    required = meta_root / "info.json"
    if not required.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {required}")
    files = sorted(
        path
        for path in meta_root.rglob("*")
        if path.is_file() and (path.name in {"info.json", "tasks.parquet"} or "episodes" in path.parts)
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(dataset_root).as_posix()
        digest.update(relative.encode())
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_manifest_index(path: Path, manifests: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for episode_index in sorted(manifests):
            stream.write(json.dumps(manifests[episode_index], sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactResumeError(f"Could not read existing episode manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactResumeError(f"Existing episode manifest must contain an object: {path}")
    return value


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactResumeError(f"Existing numeric artifact is corrupt or unsafe: {path}: {exc}") from exc


def _validate_existing_pair(
    output_root: Path,
    json_path: Path,
    npz_path: Path,
    *,
    expected_manifest: dict[str, Any] | None = None,
    expected_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if not json_path.is_file() or not npz_path.is_file():
        raise ArtifactResumeError(f"Orphan episode artifact; JSON and NPZ must both exist: {json_path} / {npz_path}")
    manifest = _read_json_object(json_path)
    try:
        _adapter.verify_episode_manifest(manifest)
    except (TypeError, ValueError) as exc:
        raise ArtifactResumeError(f"Existing episode manifest is stale or invalid: {json_path}: {exc}") from exc
    if manifest.get("schema_version") != _adapter.SCHEMA_VERSION:
        raise ArtifactResumeError(
            f"Existing episode manifest uses schema {manifest.get('schema_version')!r}, "
            f"expected {_adapter.SCHEMA_VERSION!r}: {json_path}"
        )

    artifact = manifest.get("numeric_artifact")
    if not isinstance(artifact, dict) or artifact.get("complete") is not True:
        raise ArtifactResumeError(f"Existing numeric artifact is not marked complete: {json_path}")
    try:
        declared_path = (output_root / Path(str(artifact["path"]))).resolve()
    except KeyError as exc:
        raise ArtifactResumeError(f"Existing numeric artifact has no path: {json_path}") from exc
    if not declared_path.is_relative_to(output_root.resolve()) or declared_path != npz_path.resolve():
        raise ArtifactResumeError(f"Existing numeric artifact path does not match its JSON pair: {json_path}")
    if artifact.get("size_bytes") != npz_path.stat().st_size or artifact.get("sha256") != _file_sha256(npz_path):
        raise ArtifactResumeError(f"Existing numeric artifact bytes do not match its manifest: {npz_path}")

    arrays = _load_npz_arrays(npz_path)
    declared_keys = artifact.get("keys")
    if set(arrays) != NUMERIC_ARRAY_KEYS or set(declared_keys or ()) != NUMERIC_ARRAY_KEYS:
        raise ArtifactResumeError(
            f"Existing numeric artifact keys are stale: expected {sorted(NUMERIC_ARRAY_KEYS)}, "
            f"manifest={sorted(declared_keys or ())}, npz={sorted(arrays)}"
        )
    if expected_manifest is not None:
        if manifest.get("semantic_sha256") != expected_manifest.get("semantic_sha256"):
            raise ArtifactResumeError(f"Existing episode semantics do not match current source/config: {json_path}")
        for field in ("source", "task_names"):
            if manifest.get(field) != expected_manifest.get(field):
                raise ArtifactResumeError(f"Existing episode {field} is stale: {json_path}")
        expected_artifact = expected_manifest["numeric_artifact"]
        if artifact.get("path") != expected_artifact.get("path"):
            raise ArtifactResumeError(f"Existing numeric artifact path is stale: {json_path}")
    if expected_arrays is not None:
        if set(expected_arrays) != NUMERIC_ARRAY_KEYS:
            raise AssertionError("Internal numeric array contract is incomplete")
        for key in sorted(NUMERIC_ARRAY_KEYS):
            if not np.array_equal(arrays[key], expected_arrays[key]):
                raise ArtifactResumeError(f"Existing numeric array {key!r} is stale: {npz_path}")
    return manifest


def _discover_existing_pairs(output_root: Path) -> dict[int, tuple[dict[str, Any], Path, Path]]:
    episode_root = output_root / "episodes"
    if not episode_root.exists():
        return {}
    json_by_stem = {path.relative_to(episode_root).with_suffix(""): path for path in episode_root.rglob("*.json")}
    npz_by_stem = {path.relative_to(episode_root).with_suffix(""): path for path in episode_root.rglob("*.npz")}
    orphan_json = sorted(str(stem) for stem in json_by_stem.keys() - npz_by_stem.keys())
    orphan_npz = sorted(str(stem) for stem in npz_by_stem.keys() - json_by_stem.keys())
    if orphan_json or orphan_npz:
        raise ArtifactResumeError(
            f"Orphan episode artifacts under {episode_root}: JSON-only={orphan_json}, NPZ-only={orphan_npz}"
        )

    discovered: dict[int, tuple[dict[str, Any], Path, Path]] = {}
    for stem in sorted(json_by_stem, key=str):
        json_path = json_by_stem[stem]
        npz_path = npz_by_stem[stem]
        manifest = _validate_existing_pair(output_root, json_path, npz_path)
        episode_index = manifest.get("episode_index")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise ArtifactResumeError(f"Existing episode_index must be an integer: {json_path}")
        if episode_index in discovered:
            raise ArtifactResumeError(f"Duplicate existing episode_index {episode_index}: {json_path}")
        discovered[episode_index] = (manifest, json_path, npz_path)
    return discovered


def _validate_existing_context(
    manifest: dict[str, Any],
    *,
    dataset_repo_id: str,
    dataset_metadata_sha256: str,
    target_hz: float,
    expected_views: dict[str, dict[str, str | None]],
    source_fps: float,
    all_episode_rows: dict[int, dict[str, Any]],
    episode_tasks: dict[int, tuple[str, ...]],
    splits: dict[int, str],
) -> None:
    """Reject every existing pair that is not part of the current dataset run."""

    episode_index = manifest.get("episode_index")
    if episode_index not in all_episode_rows:
        raise ArtifactResumeError(f"Existing episode {episode_index!r} no longer exists in current metadata")
    expected_dataset = {"repo_id": dataset_repo_id, "metadata_sha256": dataset_metadata_sha256}
    current_contract = {
        "dataset": expected_dataset,
        "target_hz": target_hz,
        "views": expected_views,
        "state": {"representation": "r1pro_compact_state_v1", "dim": 23},
        "training_action": _adapter.TRAINING_ACTION_CONTRACT,
        "source_policy_action": _adapter.SOURCE_POLICY_ACTION_CONTRACT,
        "online_steering": _adapter.ONLINE_STEERING_CONTRACT,
        "reward": _reward_contract.REWARD_CONTRACT,
        "reward_available": True,
        "prompt": episode_tasks[episode_index][0],
        "split": splits[episode_index],
    }
    for field, expected in current_contract.items():
        if manifest.get(field) != expected:
            raise ArtifactResumeError(
                f"Existing episode {episode_index} has stale {field}: "
                f"declared={manifest.get(field)!r}, current={expected!r}"
            )
    if manifest.get("task_names") != list(episode_tasks[episode_index]):
        raise ArtifactResumeError(f"Existing episode {episode_index} has stale task_names")

    episode_row = all_episode_rows[episode_index]
    source_start = int(_scalar(episode_row["dataset_from_index"]))
    source_end = int(_scalar(episode_row["dataset_to_index"]))
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ArtifactResumeError(f"Existing episode {episode_index} has no source object")
    expected_source_fields = {
        "dataset_from_index": source_start,
        "dataset_to_index": source_end,
        "source_fps": source_fps,
        "full_rate_count": source_end - source_start,
        "full_rate_columns": {
            "index": "index",
            "frame_index": "frame_index",
            "timestamp": "source_timestamps_s",
            "policy_action": "source_policy_actions",
            "next.reward": "source_rewards",
            "next.terminated": "source_terminated",
            "next.truncated": "source_truncated",
        },
        "policy_action_aggregation_5hz": "undefined",
    }
    for field, expected in expected_source_fields.items():
        if source.get(field) != expected:
            raise ArtifactResumeError(
                f"Existing episode {episode_index} has stale source.{field}: "
                f"declared={source.get(field)!r}, current={expected!r}"
            )


def _authoritative_index_column(value: Any, *, name: str, expected: np.ndarray) -> np.ndarray:
    array = _to_numpy(value, name=name).reshape(-1)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Authoritative LeRobot {name} must be an integer vector, got {array.dtype}")
    if not np.array_equal(array.astype(np.int64, copy=False), expected.astype(np.int64, copy=False)):
        raise ValueError(
            f"Authoritative LeRobot {name} does not match episode metadata bounds: "
            f"expected [{int(expected[0])}, ..., {int(expected[-1])}], "
            f"got [{int(array[0]) if array.size else 'empty'}, ..., {int(array[-1]) if array.size else 'empty'}]"
        )
    return array


def _episode_rows(dataset: Any, episode: dict[str, Any], columns: tuple[str, ...]) -> tuple[dict[str, Any], int, int]:
    start = int(_scalar(episode["dataset_from_index"]))
    end = int(_scalar(episode["dataset_to_index"]))
    if end <= start:
        raise ValueError(f"Episode {episode.get('episode_index')} has invalid row bounds [{start}, {end})")
    rows = dataset.hf_dataset.select_columns(list(columns))[start:end]
    return rows, start, end


def _prepare_dataset(args: Args):
    config = _config.get_config(args.config_name)
    data_factory = config.data
    if not isinstance(data_factory, _config.LeRobotB1KDataConfig):
        raise TypeError(f"{args.config_name} does not use LeRobotB1KDataConfig")
    if data_factory.robot_config_name != "b1k/R1Pro":
        raise ValueError(f"WEAVER adapter currently supports b1k/R1Pro, got {data_factory.robot_config_name}")

    base_config = data_factory.base_config
    dataset_root = Path(args.dataset_root or base_config.dataset_root or "").expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"B1K dataset root does not exist: {dataset_root}")
    dataset_kwargs = dict(base_config.dataset_kwargs)
    dataset_kwargs.pop("video_keys", None)
    dataset = base_config.data_cls(
        repo_id=data_factory.repo_id,
        root=dataset_root,
        video_keys=[],
        **dataset_kwargs,
    )
    return config, data_factory, dataset, dataset_root


def main(args: Args) -> None:
    config, data_factory, dataset, dataset_root = _prepare_dataset(args)
    contract = build_r1pro_contract()
    selected_features = _adapter.validate_selected_video_features(
        dataset.features,
        contract,
        include_depth=args.include_depth,
    )
    required_numeric_features = {
        "index",
        "frame_index",
        "timestamp",
        "episode_index",
        "observation.state",
        contract.robot.action_key,
        "next.reward",
        "next.terminated",
        "next.truncated",
    }
    available_columns = set(getattr(dataset.hf_dataset, "column_names", dataset.features))
    missing_numeric_features = required_numeric_features - available_columns
    if missing_numeric_features:
        raise KeyError(f"B1K dataset is missing required WEAVER numeric features: {sorted(missing_numeric_features)}")
    metadata_sha256 = _metadata_sha256(dataset_root)
    if not math.isclose(float(dataset.fps), 30.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"B1K WEAVER source contract requires 30 Hz data, got fps={dataset.fps!r}")

    all_episode_rows: dict[int, dict[str, Any]] = {}
    episode_tasks: dict[int, tuple[str, ...]] = {}
    for position in range(len(dataset.meta.episodes)):
        row = dict(dataset.meta.episodes[position])
        episode_index = int(_scalar(row["episode_index"]))
        if episode_index in all_episode_rows:
            raise ValueError(f"Duplicate episode_index in metadata: {episode_index}")
        all_episode_rows[episode_index] = row
        episode_tasks[episode_index] = _episode_tasks(dataset.meta, row)

    splits = _adapter.deterministic_episode_split(
        episode_tasks,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
        held_out_tasks=args.held_out_tasks,
    )
    selected_indices = sorted(all_episode_rows)
    if args.episode_indices is not None and args.sample_tasks is not None:
        raise ValueError("episode_indices and sample_tasks are mutually exclusive")
    if args.episode_indices is not None:
        requested = set(args.episode_indices)
        missing = requested - set(selected_indices)
        if missing:
            raise ValueError(f"Unknown episode indices: {sorted(missing)}")
        selected_indices = [index for index in selected_indices if index in requested]
    elif args.sample_tasks is not None:
        selected_indices = list(
            _adapter.deterministic_task_episode_sample(
                episode_tasks,
                num_tasks=args.sample_tasks,
                episodes_per_task=args.episodes_per_task,
                seed=args.selection_seed,
            )
        )
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError(f"max_episodes must be positive, got {args.max_episodes}")
        selected_indices = selected_indices[: args.max_episodes]

    output_root = Path(args.output_dir).expanduser().resolve()
    expected_views = {
        view.weaver_key: {
            "rgb": view.dataset_rgb_key,
            "depth": (
                contract.robot.observations[view.robot_camera_key].depth_dataset_key if args.include_depth else None
            ),
        }
        for view in contract.camera_views
    }
    columns = (
        "index",
        "frame_index",
        "timestamp",
        "episode_index",
        "observation.state",
        contract.robot.action_key,
        "next.reward",
        "next.terminated",
        "next.truncated",
    )

    print(
        json.dumps(
            {
                "config": config.name,
                "dataset_root": str(dataset_root),
                "repo_id": data_factory.repo_id,
                "metadata_sha256": metadata_sha256,
                "source_fps": dataset.fps,
                "total_episodes": dataset.num_episodes,
                "selected_episodes": len(selected_indices),
                "selected_tasks": sorted({episode_tasks[index][0] for index in selected_indices}),
                "selected_video_features": selected_features,
                "target_hz": args.target_hz,
                "output_dir": str(output_root),
                "dry_run": args.dry_run,
            },
            indent=2,
        ),
        flush=True,
    )

    existing_pairs = {} if args.dry_run else _discover_existing_pairs(output_root)
    for existing_manifest, _, _ in existing_pairs.values():
        _validate_existing_context(
            existing_manifest,
            dataset_repo_id=str(data_factory.repo_id),
            dataset_metadata_sha256=metadata_sha256,
            target_hz=args.target_hz,
            expected_views=expected_views,
            source_fps=float(dataset.fps),
            all_episode_rows=all_episode_rows,
            episode_tasks=episode_tasks,
            splits=splits,
        )
    manifest_index = {episode_index: pair[0] for episode_index, pair in existing_pairs.items()}
    manifests: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for ordinal, episode_index in enumerate(selected_indices, start=1):
        episode_row = all_episode_rows[episode_index]
        try:
            rows, source_start, source_end = _episode_rows(dataset, episode_row, columns)
            row_episode_indices = _to_numpy(rows["episode_index"], name="episode_index").reshape(-1)
            if row_episode_indices.size != source_end - source_start or not np.all(
                row_episode_indices == episode_index
            ):
                raise ValueError(
                    f"Episode row range [{source_start},{source_end}) contains other episode indices: "
                    f"{np.unique(row_episode_indices).tolist()}"
                )
            timestamps = _to_numpy(rows["timestamp"], name="timestamp").reshape(-1)
            raw_states = _to_numpy(rows["observation.state"], name="observation.state")
            policy_actions = _to_numpy(rows[contract.robot.action_key], name=contract.robot.action_key)
            source_rewards = _to_numpy(rows["next.reward"], name="next.reward")
            source_terminated = _to_numpy(rows["next.terminated"], name="next.terminated")
            source_truncated = _to_numpy(rows["next.truncated"], name="next.truncated")
            source_length = source_end - source_start
            source_indices = _authoritative_index_column(
                rows["index"],
                name="index",
                expected=np.arange(source_start, source_end, dtype=np.int64),
            )
            source_frame_indices = _authoritative_index_column(
                rows["frame_index"],
                name="frame_index",
                expected=np.arange(source_length, dtype=np.int64),
            )
            adapted = _adapter.adapt_episode(
                episode_index=episode_index,
                prompt=episode_tasks[episode_index][0],
                timestamps_s=timestamps,
                raw_states=raw_states,
                policy_actions=policy_actions,
                rewards=source_rewards,
                terminated=source_terminated,
                truncated=source_truncated,
                target_hz=args.target_hz,
                min_sampled_frames=args.min_sampled_frames,
                contract=contract,
            )
            split = splits[episode_index]
            manifest = _adapter.episode_manifest(
                adapted,
                split=split,
                dataset_repo_id=str(data_factory.repo_id),
                dataset_metadata_sha256=metadata_sha256,
                include_depth=args.include_depth,
                contract=contract,
            )
            manifest.update(
                {
                    "source": {
                        "dataset_from_index": source_start,
                        "dataset_to_index": source_end,
                        "source_fps": dataset.fps,
                        "full_rate_count": source_length,
                        "full_rate_columns": {
                            "index": "index",
                            "frame_index": "frame_index",
                            "timestamp": "source_timestamps_s",
                            "policy_action": "source_policy_actions",
                            "next.reward": "source_rewards",
                            "next.terminated": "source_terminated",
                            "next.truncated": "source_truncated",
                        },
                        "policy_action_aggregation_5hz": "undefined",
                        "raw_state_dim": int(raw_states.shape[-1]),
                    },
                    "task_names": list(episode_tasks[episode_index]),
                    "numeric_artifact": {"complete": True},
                    "latent_artifact": {"complete": False},
                }
            )

            relative_npz = Path("episodes") / split / f"{episode_index:06d}.npz"
            relative_json = relative_npz.with_suffix(".json")
            sampled_offsets = adapted.source_frame_indices
            sampled_frame_indices = source_frame_indices[sampled_offsets]
            if not np.array_equal(sampled_frame_indices, sampled_offsets):
                raise ValueError("Authoritative frame_index does not match the adapter's episode-local offsets")
            arrays = {
                "index": source_indices,
                "frame_index": source_frame_indices,
                "source_timestamps_s": timestamps,
                "source_policy_actions": policy_actions,
                "source_rewards": source_rewards,
                "source_terminated": source_terminated,
                "source_truncated": source_truncated,
                "source_frame_indices": sampled_frame_indices,
                "source_absolute_indices": source_indices[sampled_offsets],
                "timestamps_s": adapted.timestamps_s,
                "timing_errors_s": adapted.timing_errors_s,
                "states": adapted.states,
                "state_delta_actions": adapted.state_delta_actions,
                "policy_actions_at_frame": adapted.policy_actions_at_frame,
                "transition_valid": adapted.transition_valid,
                "rewards": adapted.rewards,
            }
            manifest["numeric_artifact"].update(
                {
                    "path": relative_npz.as_posix(),
                    "keys": sorted(arrays),
                }
            )
            write_status = "dry-run"
            if not args.dry_run:
                npz_path = output_root / relative_npz
                json_path = output_root / relative_json
                existing = existing_pairs.get(episode_index)
                if existing is not None:
                    _, existing_json_path, existing_npz_path = existing
                    persisted_manifest = _validate_existing_pair(
                        output_root,
                        existing_json_path,
                        existing_npz_path,
                        expected_manifest=manifest,
                        expected_arrays=arrays,
                    )
                    write_status = "reused"
                else:
                    if npz_path.exists() or json_path.exists():
                        raise ArtifactResumeError(f"Orphan or unindexed episode artifact: {npz_path} / {json_path}")
                    _atomic_write_npz(npz_path, arrays)
                    manifest["numeric_artifact"].update(
                        {
                            "size_bytes": npz_path.stat().st_size,
                            "sha256": _file_sha256(npz_path),
                        }
                    )
                    _atomic_write_json(json_path, manifest)
                    persisted_manifest = manifest
                    existing_pairs[episode_index] = (persisted_manifest, json_path, npz_path)
                    write_status = "written"
                manifest_index[episode_index] = persisted_manifest
                _atomic_write_manifest_index(output_root / "manifest.jsonl", manifest_index)
            else:
                persisted_manifest = manifest
            manifests.append(persisted_manifest)
            print(
                f"[{ordinal}/{len(selected_indices)}] episode={episode_index} split={split} "
                f"artifact={write_status} "
                f"source={source_end - source_start} sampled={len(adapted.timestamps_s)} "
                f"raw_state={raw_states.shape} policy_action={policy_actions.shape} "
                f"weaver_state={adapted.states.shape} weaver_action={adapted.state_delta_actions.shape} "
                f"max_dt_error={adapted.timing_errors_s.max(initial=0.0):.6f}s",
                flush=True,
            )
        except ArtifactResumeError:
            raise
        except Exception as exc:
            failure = {"episode_index": episode_index, "error_type": type(exc).__name__, "error": str(exc)}
            rejected.append(failure)
            print(json.dumps({"rejected": failure}, ensure_ascii=False), flush=True)
            if args.fail_fast:
                raise

    summary = {
        "schema_version": _adapter.SCHEMA_VERSION,
        "dataset_repo_id": data_factory.repo_id,
        "dataset_metadata_sha256": metadata_sha256,
        "target_hz": args.target_hz,
        "selected_video_features": list(selected_features),
        "processed": len(manifests),
        "rejected": rejected,
        "split_counts": {
            split: sum(manifest["split"] == split for manifest in manifests) for split in ("train", "val", "test")
        },
        "latent_complete": False,
        "reward_export": dict(_reward_contract.REWARD_CONTRACT),
        "online_steering": "disabled_offline_state_delta_is_noncausal",
    }
    if not args.dry_run:
        _atomic_write_json(output_root / "summary.json", summary)
        _atomic_write_manifest_index(output_root / "manifest.jsonl", manifest_index)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
