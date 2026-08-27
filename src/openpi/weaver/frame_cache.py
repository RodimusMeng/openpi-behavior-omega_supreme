"""Crash-safe, byte-verifiable uint8 RGB frame artifacts for B1K WEAVER.

This module deliberately depends only on the Python standard library and
NumPy.  The Omega/LeRobot environment writes these artifacts; the official
WEAVER environment can then consume them without importing LeRobot or its
dependency graph.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from openpi.weaver.b1k_dataset_adapter import SCHEMA_VERSION as EPISODE_SCHEMA_VERSION
from openpi.weaver.b1k_dataset_adapter import array_sha256
from openpi.weaver.b1k_dataset_adapter import semantic_sha256
from openpi.weaver.b1k_dataset_adapter import verify_episode_manifest

FRAME_SCHEMA_VERSION = "b1k_weaver_frame_v1"
PIXEL_FORMAT = {
    "dtype": "uint8",
    "layout": "THWC",
    "channel_order": "RGB",
    "channels": 3,
}
FRAME_SEMANTIC_FIELDS = (
    "schema_version",
    "numeric_episode_schema_version",
    "numeric_episode_semantic_sha256",
    "numeric_artifact_sha256",
    "dataset",
    "episode_index",
    "split",
    "sampled_count",
    "source_absolute_indices_sha256",
    "views",
    "view_order",
    "pixel_format",
    "decoder",
    "exporter_git_sha",
)


class FrameCacheError(RuntimeError):
    """Raised when a frame artifact is absent, stale, corrupt, or unsafe."""


@dataclasses.dataclass(frozen=True)
class FrameEpisode:
    source_absolute_indices: np.ndarray
    frames: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any] | None = None

    @property
    def length(self) -> int:
        return int(self.source_absolute_indices.shape[0])


@dataclasses.dataclass(frozen=True)
class DecoderSpec:
    lerobot_git_sha: str
    pyav_version: str
    video_backend: str = "pyav"
    return_uint8: bool = True
    tolerance_s: float = 5e-4

    def __post_init__(self) -> None:
        if len(self.lerobot_git_sha) != 40:
            raise ValueError("lerobot_git_sha must be an exact 40-character Git commit")
        bytes.fromhex(self.lerobot_git_sha)
        if not self.pyav_version.strip():
            raise ValueError("pyav_version must be recorded")
        if self.video_backend != "pyav":
            raise ValueError(f"First frame bridge requires video_backend='pyav', got {self.video_backend!r}")
        if self.return_uint8 is not True:
            raise ValueError("Frame bridge requires return_uint8=True")
        if self.tolerance_s != 5e-4:
            raise ValueError(f"Frame bridge requires tolerance_s=5e-4, got {self.tolerance_s!r}")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of the exact bytes at ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _view_contract(numeric_episode_manifest: Mapping[str, Any], view_order: tuple[str, ...]) -> dict[str, Any]:
    if not view_order or len(view_order) != len(set(view_order)):
        raise ValueError(f"view_order must be non-empty and unique, got {view_order}")
    available_views = numeric_episode_manifest.get("views", {})
    if not isinstance(available_views, Mapping):
        raise ValueError("numeric episode manifest has no views mapping")
    views: dict[str, Any] = {}
    for view in view_order:
        value = available_views.get(view)
        if not isinstance(value, Mapping) or not isinstance(value.get("rgb"), str) or not value["rgb"]:
            raise ValueError(f"numeric episode view {view!r} has no RGB dataset key")
        views[view] = {
            "dataset_rgb_key": value["rgb"],
            "array_key": f"{view}_rgb",
        }
    return views


def frame_semantics(
    numeric_episode_manifest: Mapping[str, Any],
    *,
    source_absolute_indices: np.ndarray,
    view_order: tuple[str, ...],
    decoder: DecoderSpec,
    exporter_git_sha: str,
) -> dict[str, Any]:
    """Build the complete semantic fingerprint for one decoded episode."""

    verify_episode_manifest(numeric_episode_manifest)
    if numeric_episode_manifest.get("schema_version") != EPISODE_SCHEMA_VERSION:
        raise ValueError(
            "Numeric episode schema mismatch: "
            f"expected {EPISODE_SCHEMA_VERSION!r}, got {numeric_episode_manifest.get('schema_version')!r}"
        )
    numeric_semantic_sha = numeric_episode_manifest.get("semantic_sha256")
    if not isinstance(numeric_semantic_sha, str) or len(numeric_semantic_sha) != 64:
        raise ValueError("numeric episode manifest has no valid semantic_sha256")
    bytes.fromhex(numeric_semantic_sha)
    numeric_artifact = numeric_episode_manifest.get("numeric_artifact")
    if not isinstance(numeric_artifact, Mapping):
        raise ValueError("numeric episode manifest has no numeric_artifact")
    numeric_artifact_sha = numeric_artifact.get("sha256")
    if not isinstance(numeric_artifact_sha, str) or len(numeric_artifact_sha) != 64:
        raise ValueError("numeric episode manifest has no valid numeric artifact SHA-256")
    bytes.fromhex(numeric_artifact_sha)
    if not exporter_git_sha.strip():
        raise ValueError("exporter_git_sha must be recorded")

    indices = np.asarray(source_absolute_indices)
    sampled_count = int(numeric_episode_manifest.get("sampled_count", -1))
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"source_absolute_indices must be an integer vector, got {indices.shape}/{indices.dtype}")
    if indices.shape[0] != sampled_count or np.any(np.diff(indices) <= 0):
        raise ValueError(
            "source_absolute_indices must be strictly increasing and match "
            f"sampled_count={sampled_count}, got {indices.shape}"
        )

    semantics = {
        "schema_version": FRAME_SCHEMA_VERSION,
        "numeric_episode_schema_version": EPISODE_SCHEMA_VERSION,
        "numeric_episode_semantic_sha256": numeric_semantic_sha,
        "numeric_artifact_sha256": numeric_artifact_sha,
        "dataset": numeric_episode_manifest["dataset"],
        "episode_index": numeric_episode_manifest["episode_index"],
        "split": numeric_episode_manifest["split"],
        "sampled_count": sampled_count,
        "source_absolute_indices_sha256": array_sha256(indices),
        "views": _view_contract(numeric_episode_manifest, view_order),
        "view_order": list(view_order),
        "pixel_format": dict(PIXEL_FORMAT),
        "decoder": dataclasses.asdict(decoder),
        "exporter_git_sha": exporter_git_sha,
    }
    semantics["semantic_sha256"] = semantic_sha256(semantics)
    return semantics


def verify_frame_semantics(manifest: Mapping[str, Any]) -> str:
    """Verify and return a frame artifact's semantic fingerprint."""

    try:
        semantic = {key: manifest[key] for key in FRAME_SEMANTIC_FIELDS}
    except KeyError as exc:
        raise ValueError(f"Frame manifest is missing semantic field {exc.args[0]!r}") from exc
    if semantic["schema_version"] != FRAME_SCHEMA_VERSION:
        raise ValueError(f"Unsupported frame schema: {semantic['schema_version']!r}")
    if semantic["pixel_format"] != PIXEL_FORMAT:
        raise ValueError(f"Unsupported frame pixel format: {semantic['pixel_format']!r}")
    decoder = semantic["decoder"]
    if not isinstance(decoder, Mapping):
        raise ValueError("Frame decoder contract must be an object")
    try:
        DecoderSpec(**decoder)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid frame decoder contract: {decoder!r}") from exc
    if not isinstance(semantic["exporter_git_sha"], str) or not semantic["exporter_git_sha"].strip():
        raise ValueError("Frame exporter_git_sha must be non-empty")
    expected = manifest.get("semantic_sha256")
    actual = semantic_sha256(semantic)
    if expected != actual:
        raise ValueError(f"Frame semantic SHA-256 mismatch: declared={expected!r}, actual={actual}")
    return actual


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    semantics: Mapping[str, Any],
) -> FrameEpisode:
    view_order = tuple(semantics["view_order"])
    expected_keys = {"source_absolute_indices", *(f"{view}_rgb" for view in view_order)}
    if set(arrays) != expected_keys:
        raise ValueError(f"Frame keys must be exactly {sorted(expected_keys)}, got {sorted(arrays)}")

    indices = np.asarray(arrays["source_absolute_indices"])
    sampled_count = int(semantics["sampled_count"])
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"source_absolute_indices must be an integer vector, got {indices.shape}/{indices.dtype}")
    if indices.shape != (sampled_count,) or np.any(np.diff(indices) <= 0):
        raise ValueError(
            f"source_absolute_indices must have shape ({sampled_count},) and increase strictly, got {indices.shape}"
        )
    if array_sha256(indices) != semantics["source_absolute_indices_sha256"]:
        raise ValueError("source_absolute_indices SHA-256 does not match frame semantics")
    immutable_indices = np.ascontiguousarray(indices).copy()
    immutable_indices.setflags(write=False)

    frames: dict[str, np.ndarray] = {}
    for view in view_order:
        key = f"{view}_rgb"
        frame_array = np.asarray(arrays[key])
        if frame_array.dtype != np.uint8 or frame_array.ndim != 4:
            raise ValueError(f"{key} must be rank-4 uint8 THWC, got {frame_array.shape}/{frame_array.dtype}")
        if (
            frame_array.shape[0] != sampled_count
            or frame_array.shape[1] <= 0
            or frame_array.shape[2] <= 0
            or frame_array.shape[-1] != 3
        ):
            raise ValueError(f"{key} must have shape ({sampled_count}, H, W, 3), got {frame_array.shape}")
        immutable = np.ascontiguousarray(frame_array).copy()
        immutable.setflags(write=False)
        frames[view] = immutable
    return FrameEpisode(source_absolute_indices=immutable_indices, frames=frames)


def write_frame_artifact(
    npz_path: str | Path,
    manifest_path: str | Path,
    *,
    semantics: Mapping[str, Any],
    source_absolute_indices: np.ndarray,
    frames: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Atomically commit one frame NPZ, followed by its complete manifest."""

    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    if npz_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite frame cache: {npz_path} / {manifest_path}")
    if semantics.get("schema_version") != FRAME_SCHEMA_VERSION:
        raise ValueError(f"Unexpected frame semantics schema: {semantics.get('schema_version')!r}")
    verify_frame_semantics(semantics)

    arrays = {"source_absolute_indices": source_absolute_indices}
    arrays.update({f"{view}_rgb": frames[view] for view in semantics["view_order"]})
    validated = _validate_arrays(arrays, semantics=semantics)
    committed_arrays = {"source_absolute_indices": validated.source_absolute_indices}
    committed_arrays.update({f"{view}_rgb": validated.frames[view] for view in semantics["view_order"]})

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_npz = npz_path.with_name(f".{npz_path.name}.tmp-{os.getpid()}")
    temp_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    try:
        with temp_npz.open("wb") as stream:
            np.savez_compressed(stream, **committed_arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_npz, npz_path)

        artifact = {
            **semantics,
            "complete": True,
            "artifact": {
                "path": npz_path.name,
                "size_bytes": npz_path.stat().st_size,
                "sha256": file_sha256(npz_path),
                "arrays": {
                    key: {"shape": list(value.shape), "dtype": value.dtype.str}
                    for key, value in committed_arrays.items()
                },
            },
        }
        artifact = json.loads(json.dumps(artifact, allow_nan=False, ensure_ascii=False))
        with temp_manifest.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(artifact, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_manifest, manifest_path)
        return artifact
    finally:
        temp_npz.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrameCacheError(f"frame manifest is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise FrameCacheError("frame manifest must contain a JSON object")
    return value


def load_frame_artifact(
    npz_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_semantic_sha256: str,
) -> FrameEpisode:
    """Load an artifact only after verifying semantics, bytes, keys and shapes."""

    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FrameCacheError("frame npz and complete manifest must both exist")
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema_version") != FRAME_SCHEMA_VERSION or manifest.get("complete") is not True:
        raise FrameCacheError("frame manifest is incomplete or has the wrong schema")
    try:
        verify_frame_semantics(manifest)
    except ValueError as exc:
        raise FrameCacheError(str(exc)) from exc
    if manifest.get("semantic_sha256") != expected_semantic_sha256:
        raise FrameCacheError("frame semantic fingerprint changed")

    artifact = manifest.get("artifact", {})
    if artifact.get("path") != npz_path.name:
        raise FrameCacheError("frame manifest points to a different artifact filename")
    if artifact.get("size_bytes") != npz_path.stat().st_size or artifact.get("sha256") != file_sha256(npz_path):
        raise FrameCacheError("frame npz size or SHA-256 does not match its manifest")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise FrameCacheError(f"frame npz is corrupt or unsafe: {exc}") from exc
    try:
        validated = _validate_arrays(arrays, semantics=manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise FrameCacheError(f"frame arrays violate the manifest contract: {exc}") from exc

    declared = artifact.get("arrays", {})
    actual_arrays = {"source_absolute_indices": validated.source_absolute_indices}
    actual_arrays.update({f"{view}_rgb": validated.frames[view] for view in manifest["view_order"]})
    actual = {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in actual_arrays.items()}
    if declared != actual:
        raise FrameCacheError("frame array shape/dtype metadata does not match the npz")
    return dataclasses.replace(validated, manifest=manifest)


def load_episode_frames(
    frame_root: str | Path,
    numeric_episode_manifest: Mapping[str, Any],
    *,
    source_absolute_indices: np.ndarray,
    view_order: tuple[str, ...],
    expected_exporter_git_sha: str | None = None,
    expected_decoder: DecoderSpec | None = None,
) -> FrameEpisode:
    """Load one frame artifact and prove exact linkage to current numeric data."""

    verify_episode_manifest(numeric_episode_manifest)
    frame_root = Path(frame_root).resolve()
    split = str(numeric_episode_manifest["split"])
    episode_index = int(numeric_episode_manifest["episode_index"])
    npz_path = frame_root / split / f"{episode_index:06d}.npz"
    manifest_path = npz_path.with_suffix(".json")
    manifest = _read_manifest(manifest_path)

    if expected_exporter_git_sha is not None and manifest.get("exporter_git_sha") != expected_exporter_git_sha:
        raise FrameCacheError(
            "frame artifact exporter revision changed: "
            f"expected {expected_exporter_git_sha!r}, got {manifest.get('exporter_git_sha')!r}"
        )
    if expected_decoder is not None and manifest.get("decoder") != dataclasses.asdict(expected_decoder):
        raise FrameCacheError("frame artifact decoder contract changed")

    expected_views = _view_contract(numeric_episode_manifest, view_order)
    numeric_artifact = numeric_episode_manifest.get("numeric_artifact", {})
    expected_linkage = {
        "numeric_episode_schema_version": numeric_episode_manifest.get("schema_version"),
        "numeric_episode_semantic_sha256": numeric_episode_manifest.get("semantic_sha256"),
        "numeric_artifact_sha256": numeric_artifact.get("sha256"),
        "dataset": numeric_episode_manifest.get("dataset"),
        "episode_index": episode_index,
        "split": split,
        "sampled_count": int(numeric_episode_manifest.get("sampled_count", -1)),
        "source_absolute_indices_sha256": array_sha256(np.asarray(source_absolute_indices)),
        "views": expected_views,
        "view_order": list(view_order),
        "pixel_format": PIXEL_FORMAT,
    }
    mismatched = [key for key, value in expected_linkage.items() if manifest.get(key) != value]
    if mismatched:
        raise FrameCacheError(
            f"frame artifact does not match current numeric episode {episode_index}: {sorted(mismatched)}"
        )
    artifact = load_frame_artifact(
        npz_path,
        manifest_path,
        expected_semantic_sha256=str(manifest.get("semantic_sha256")),
    )
    if not np.array_equal(artifact.source_absolute_indices, np.asarray(source_absolute_indices)):
        raise FrameCacheError("frame artifact absolute indices differ from the numeric episode")
    return artifact
