"""Export exact B1K RGB rows to an environment-neutral uint8 frame cache.

Run this stage in the OmegaSupreme environment, where the pinned LeRobot
reader is already available.  The output is linked byte-for-byte to the numeric
episode artifact and can be consumed by the official WEAVER environment
without installing LeRobot there.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal

import av
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

from openpi.weaver import b1k_dataset_adapter as _episode_adapter
from openpi.weaver import behavior_precomputed
from openpi.weaver import frame_cache


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_root: str
    numeric_dir: str
    exporter_git_sha: str
    lerobot_git_sha: str
    output_dir: str | None = None
    split: Literal["train", "val", "test"] = "train"
    episode_indices: tuple[int, ...] | None = None
    max_episodes: int | None = None
    verify_cache_only: bool = False


@dataclasses.dataclass(frozen=True)
class _WorkItem:
    manifest: dict[str, Any]
    numeric: behavior_precomputed.NumericEpisode
    npz_path: Path
    manifest_path: Path


def _scalar(value: Any, *, name: str) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    return array.reshape(-1)[0].item()


def _as_hwc_uint8(value: Any, *, key: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{key} must contain one RGB frame, got shape {array.shape}")
    if array.shape[-1] == 3:
        pass
    elif array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(f"{key} has no three-channel RGB axis: {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(
            f"{key} must be decoded as exact uint8 pixels, got {array.dtype}; "
            "ensure LeRobotDataset(return_uint8=True) is honored"
        )
    return np.ascontiguousarray(array)


def _read_numeric_manifests(root: Path, args: Args) -> list[dict[str, Any]]:
    path = root / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing numeric manifest index: {path}")
    manifests: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                manifest = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            _episode_adapter.verify_episode_manifest(manifest)
            if manifest["split"] == args.split:
                manifests.append(manifest)
    manifests.sort(key=lambda value: int(value["episode_index"]))
    if args.episode_indices is not None:
        requested = set(args.episode_indices)
        available = {int(manifest["episode_index"]) for manifest in manifests}
        if missing := requested - available:
            raise ValueError(f"Requested episodes are absent from split {args.split}: {sorted(missing)}")
        manifests = [manifest for manifest in manifests if int(manifest["episode_index"]) in requested]
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        manifests = manifests[: args.max_episodes]
    if not manifests:
        raise ValueError(f"No numeric episodes selected for split {args.split}")
    return manifests


def _view_contract(manifests: list[dict[str, Any]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    image_keys = behavior_precomputed.DEFAULT_IMAGE_KEYS
    selected_features = tuple(manifests[0]["views"][key]["rgb"] for key in image_keys)
    for manifest in manifests:
        current = tuple(manifest["views"][key]["rgb"] for key in image_keys)
        if current != selected_features:
            raise ValueError("Selected numeric manifests do not share one ordered RGB camera mapping")
    return image_keys, selected_features


def _open_visual_dataset(
    dataset_root: Path,
    *,
    dataset_repo_id: str,
    selected_features: tuple[str, ...],
    decoder: frame_cache.DecoderSpec,
):
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"B1K dataset root does not exist: {dataset_root}")
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=dataset_root,
        video_keys=list(selected_features),
        return_uint8=decoder.return_uint8,
        tolerance_s=decoder.tolerance_s,
        video_backend=decoder.video_backend,
    )
    actual = _episode_adapter.validate_selected_video_features(dataset.features)
    if actual != selected_features:
        raise ValueError(f"LeRobot selected feature order changed: expected {selected_features}, got {actual}")
    return dataset


def _decode_episode(
    dataset: Any,
    work: _WorkItem,
    *,
    image_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    manifest = work.manifest
    episode_index = int(manifest["episode_index"])
    dataset_keys = {image_key: manifest["views"][image_key]["rgb"] for image_key in image_keys}
    buckets: dict[str, list[np.ndarray]] = {image_key: [] for image_key in image_keys}
    for expected_index in work.numeric.source_absolute_indices.tolist():
        row = dataset[int(expected_index)]
        actual_index = int(_scalar(row["index"], name="index"))
        if actual_index != expected_index:
            raise ValueError(f"LeRobot row mismatch: requested {expected_index}, got {actual_index}")
        actual_episode = int(_scalar(row["episode_index"], name="episode_index"))
        if actual_episode != episode_index:
            raise ValueError(
                f"LeRobot episode mismatch at absolute row {expected_index}: "
                f"expected {episode_index}, got {actual_episode}"
            )
        for image_key, dataset_key in dataset_keys.items():
            buckets[image_key].append(_as_hwc_uint8(row[dataset_key], key=dataset_key))
    try:
        return {image_key: np.stack(values, axis=0) for image_key, values in buckets.items()}
    except ValueError as exc:
        shapes = {key: [value.shape for value in values] for key, values in buckets.items()}
        raise ValueError(f"Episode {episode_index} decoded inconsistent frame shapes: {shapes}") from exc


def main(args: Args) -> None:
    if not args.exporter_git_sha.strip():
        raise ValueError("exporter_git_sha must be a committed OmegaSupreme revision")
    decoder = frame_cache.DecoderSpec(
        lerobot_git_sha=args.lerobot_git_sha,
        pyav_version=av.__version__,
    )
    numeric_root = Path(args.numeric_dir).expanduser().resolve()
    frame_root = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir is not None else numeric_root / "frames"
    )
    manifests = _read_numeric_manifests(numeric_root, args)
    image_keys, selected_features = _view_contract(manifests)
    dataset_repo_ids = {str(manifest["dataset"]["repo_id"]) for manifest in manifests}
    if len(dataset_repo_ids) != 1:
        raise ValueError(f"Selected manifests contain multiple dataset repo IDs: {sorted(dataset_repo_ids)}")
    dataset_repo_id = dataset_repo_ids.pop()

    pending: list[_WorkItem] = []
    for ordinal, manifest in enumerate(manifests, start=1):
        numeric = behavior_precomputed.load_numeric_episode(numeric_root, manifest)
        episode_index = int(manifest["episode_index"])
        npz_path = frame_root / args.split / f"{episode_index:06d}.npz"
        manifest_path = npz_path.with_suffix(".json")
        if npz_path.exists() or manifest_path.exists():
            frame_cache.load_episode_frames(
                frame_root,
                manifest,
                source_absolute_indices=numeric.source_absolute_indices,
                view_order=image_keys,
                expected_exporter_git_sha=args.exporter_git_sha,
                expected_decoder=decoder,
            )
            print(
                f"[{ordinal}/{len(manifests)}] episode={episode_index} frames={numeric.length} cached=True",
                flush=True,
            )
        else:
            pending.append(
                _WorkItem(
                    manifest=manifest,
                    numeric=numeric,
                    npz_path=npz_path,
                    manifest_path=manifest_path,
                )
            )

    print(
        json.dumps(
            {
                "numeric_root": str(numeric_root),
                "frame_root": str(frame_root),
                "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
                "dataset_repo_id": dataset_repo_id,
                "episodes": len(manifests),
                "cache_hits": len(manifests) - len(pending),
                "cache_misses": len(pending),
                "view_order": image_keys,
                "selected_video_features": selected_features,
                "pixel_format": frame_cache.PIXEL_FORMAT,
                "decoder": dataclasses.asdict(decoder),
                "verify_cache_only": args.verify_cache_only,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.verify_cache_only:
        if pending:
            missing = [int(work.manifest["episode_index"]) for work in pending]
            raise frame_cache.FrameCacheError(f"Frame cache misses remain for episodes: {missing}")
        return
    if not pending:
        return

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset = _open_visual_dataset(
        dataset_root,
        dataset_repo_id=dataset_repo_id,
        selected_features=selected_features,
        decoder=decoder,
    )
    for ordinal, work in enumerate(pending, start=1):
        frames = _decode_episode(dataset, work, image_keys=image_keys)
        semantics = frame_cache.frame_semantics(
            work.manifest,
            source_absolute_indices=work.numeric.source_absolute_indices,
            view_order=image_keys,
            decoder=decoder,
            exporter_git_sha=args.exporter_git_sha,
        )
        frame_cache.write_frame_artifact(
            work.npz_path,
            work.manifest_path,
            semantics=semantics,
            source_absolute_indices=work.numeric.source_absolute_indices,
            frames=frames,
        )
        verified = frame_cache.load_episode_frames(
            frame_root,
            work.manifest,
            source_absolute_indices=work.numeric.source_absolute_indices,
            view_order=image_keys,
            expected_exporter_git_sha=args.exporter_git_sha,
            expected_decoder=decoder,
        )
        episode_index = int(work.manifest["episode_index"])
        shapes = {view: list(value.shape) for view, value in verified.frames.items()}
        print(
            f"[{ordinal}/{len(pending)}] episode={episode_index} frames={verified.length} cached=False shapes={shapes}",
            flush=True,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
