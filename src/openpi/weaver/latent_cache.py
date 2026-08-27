"""Auditable, crash-safe latent-cache artifacts for B1K WEAVER episodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from openpi.weaver import frame_cache as _frame_cache
from openpi.weaver.b1k_dataset_adapter import SCHEMA_VERSION as EPISODE_SCHEMA_VERSION
from openpi.weaver.b1k_dataset_adapter import semantic_sha256

LATENT_SCHEMA_VERSION = "b1k_weaver_latent_v2"
LATENT_SEMANTIC_FIELDS = (
    "schema_version",
    "numeric_episode_schema_version",
    "numeric_episode_semantic_sha256",
    "dataset",
    "episode_index",
    "split",
    "target_hz",
    "sampled_count",
    "source_frame_indices_sha256",
    "timestamps_sha256",
    "views",
    "view_order",
    "frame_schema_version",
    "frame_semantic_sha256",
    "frame_artifact_sha256",
    "encoder",
    "adapter_git_sha",
)


class CacheMissError(RuntimeError):
    """Raised when an artifact is absent, stale, corrupt, or incomplete."""


@dataclasses.dataclass(frozen=True)
class EncoderSpec:
    model_id: str
    revision: str
    image_size: tuple[int, int]
    spatial_size: int
    compute_dtype: str
    storage_dtype: str
    vae_snapshot_sha256: str
    checkpoint_config_sha256: str
    resize_mode: str = "letterbox"
    interpolation: str = "bilinear-align-corners-false"
    modality: str = "rgb"

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.revision.strip():
            raise ValueError("Encoder model_id and exact revision must be non-empty")
        if len(self.image_size) != 2 or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in self.image_size
        ):
            raise ValueError(f"image_size must contain two positive integers, got {self.image_size}")
        if isinstance(self.spatial_size, bool) or not isinstance(self.spatial_size, int) or self.spatial_size <= 0:
            raise ValueError(f"spatial_size must be a positive integer, got {self.spatial_size!r}")
        if self.compute_dtype != "float32":
            raise ValueError(f"First SD3 transfer requires float32 compute, got {self.compute_dtype!r}")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError(f"storage_dtype must be float16 or float32, got {self.storage_dtype!r}")
        for name, value in (
            ("vae_snapshot_sha256", self.vae_snapshot_sha256),
            ("checkpoint_config_sha256", self.checkpoint_config_sha256),
        ):
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
            bytes.fromhex(value)
        if self.resize_mode not in {"letterbox", "center-crop", "stretch"}:
            raise ValueError(f"Unsupported resize_mode: {self.resize_mode!r}")
        if self.interpolation != "bilinear-align-corners-false":
            raise ValueError(f"Unsupported interpolation contract: {self.interpolation!r}")
        if self.modality != "rgb":
            raise ValueError(f"Unsupported encoder modality: {self.modality!r}")


@dataclasses.dataclass(frozen=True)
class CacheStatus:
    hit: bool
    reason: str


def verify_latent_semantics(manifest: Mapping[str, Any]) -> str:
    """Verify and return the latent artifact's semantic fingerprint."""

    try:
        semantic = {key: manifest[key] for key in LATENT_SEMANTIC_FIELDS}
    except KeyError as exc:
        raise ValueError(f"Latent manifest is missing semantic field {exc.args[0]!r}") from exc
    expected = manifest.get("semantic_sha256")
    actual = semantic_sha256(semantic)
    if expected != actual:
        raise ValueError(f"Latent semantic SHA-256 mismatch: declared={expected!r}, actual={actual}")
    return actual


def latent_semantics(
    numeric_episode_manifest: Mapping[str, Any],
    *,
    frame_manifest: Mapping[str, Any],
    encoder: EncoderSpec,
    view_order: tuple[str, ...],
    adapter_git_sha: str,
) -> dict[str, Any]:
    """Build every semantic input that can invalidate encoded latents."""

    if numeric_episode_manifest.get("schema_version") != EPISODE_SCHEMA_VERSION:
        raise ValueError(
            "Numeric episode schema mismatch: "
            f"expected {EPISODE_SCHEMA_VERSION!r}, got {numeric_episode_manifest.get('schema_version')!r}"
        )
    numeric_semantic_sha = numeric_episode_manifest.get("semantic_sha256")
    if not isinstance(numeric_semantic_sha, str) or len(numeric_semantic_sha) != 64:
        raise ValueError("numeric episode manifest has no valid semantic_sha256")
    bytes.fromhex(numeric_semantic_sha)
    if not view_order or len(view_order) != len(set(view_order)):
        raise ValueError(f"view_order must be non-empty and unique, got {view_order}")
    available_views = numeric_episode_manifest.get("views", {})
    missing_views = set(view_order) - set(available_views)
    if missing_views:
        raise ValueError(f"view_order contains views absent from the episode manifest: {sorted(missing_views)}")
    if not adapter_git_sha.strip():
        raise ValueError("adapter_git_sha must be recorded")
    try:
        frame_semantic_sha = _frame_cache.verify_frame_semantics(frame_manifest)
    except ValueError as exc:
        raise ValueError(f"frame manifest semantics are invalid: {exc}") from exc
    if frame_manifest.get("complete") is not True:
        raise ValueError("frame manifest must be complete before latents are addressed")
    if frame_manifest.get("numeric_episode_semantic_sha256") != numeric_semantic_sha:
        raise ValueError("frame manifest is linked to different numeric episode semantics")
    if frame_manifest.get("episode_index") != numeric_episode_manifest.get("episode_index"):
        raise ValueError("frame and numeric episode indices differ")
    if frame_manifest.get("split") != numeric_episode_manifest.get("split"):
        raise ValueError("frame and numeric splits differ")
    if frame_manifest.get("sampled_count") != numeric_episode_manifest.get("sampled_count"):
        raise ValueError("frame and numeric sampled counts differ")
    if tuple(frame_manifest.get("view_order", ())) != view_order:
        raise ValueError("frame manifest view order differs from latent view order")
    numeric_artifact = numeric_episode_manifest.get("numeric_artifact", {})
    if frame_manifest.get("numeric_artifact_sha256") != numeric_artifact.get("sha256"):
        raise ValueError("frame manifest is linked to different numeric artifact bytes")
    frame_artifact = frame_manifest.get("artifact")
    if not isinstance(frame_artifact, Mapping):
        raise ValueError("frame manifest has no artifact object")
    frame_artifact_sha = frame_artifact.get("sha256")
    if not isinstance(frame_artifact_sha, str) or len(frame_artifact_sha) != 64:
        raise ValueError("frame manifest has no valid artifact SHA-256")
    bytes.fromhex(frame_artifact_sha)

    semantics = {
        "schema_version": LATENT_SCHEMA_VERSION,
        "numeric_episode_schema_version": EPISODE_SCHEMA_VERSION,
        "numeric_episode_semantic_sha256": numeric_semantic_sha,
        "dataset": numeric_episode_manifest["dataset"],
        "episode_index": numeric_episode_manifest["episode_index"],
        "split": numeric_episode_manifest["split"],
        "target_hz": numeric_episode_manifest["target_hz"],
        "sampled_count": numeric_episode_manifest["sampled_count"],
        "source_frame_indices_sha256": numeric_episode_manifest["source_frame_indices_sha256"],
        "timestamps_sha256": numeric_episode_manifest["timestamps_sha256"],
        "views": {view: available_views[view] for view in view_order},
        "view_order": list(view_order),
        "frame_schema_version": _frame_cache.FRAME_SCHEMA_VERSION,
        "frame_semantic_sha256": frame_semantic_sha,
        "frame_artifact_sha256": frame_artifact_sha,
        "encoder": dataclasses.asdict(encoder),
        "adapter_git_sha": adapter_git_sha,
    }
    semantics["semantic_sha256"] = semantic_sha256(semantics)
    return semantics


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: str | Path) -> str:
    """Hash every filename and exact file byte under a local model directory.

    Paths are part of the digest, so moving bytes to a different model-relative
    filename also invalidates the encoder contract. Symlinks are rejected to
    keep the digest independent of files outside the pinned snapshot.
    """

    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Snapshot directory does not exist: {root}")
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Snapshot directory contains no files: {root}")
    digest = hashlib.sha256()
    for file_path in files:
        if file_path.is_symlink():
            raise ValueError(f"Snapshot hashing refuses symlinks: {file_path}")
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(file_path.stat().st_size.to_bytes(8, "little"))
        with file_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_latents(
    latents: Mapping[str, np.ndarray],
    *,
    view_order: tuple[str, ...],
    sampled_count: int,
) -> dict[str, np.ndarray]:
    expected_keys = tuple(f"{view}_features" for view in view_order)
    if set(latents) != set(expected_keys):
        raise ValueError(f"Latent keys must be exactly {sorted(expected_keys)}, got {sorted(latents)}")
    validated: dict[str, np.ndarray] = {}
    reference_shape: tuple[int, ...] | None = None
    # Preserve the semantic view order in the NPZ instead of alphabetizing it.
    for key in expected_keys:
        array = np.asarray(latents[key])
        if array.ndim < 2 or array.shape[0] != sampled_count:
            raise ValueError(f"{key} must start with sampled time dimension {sampled_count}, got {array.shape}")
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"{key} must be floating point, got {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains non-finite values")
        if reference_shape is None:
            reference_shape = array.shape
        elif array.shape != reference_shape:
            raise ValueError(f"Every view must use one latent shape, got {reference_shape} and {array.shape}")
        immutable = np.array(array, copy=True)
        immutable.setflags(write=False)
        validated[key] = immutable
    return validated


def write_latent_artifact(
    npz_path: str | Path,
    manifest_path: str | Path,
    *,
    semantics: Mapping[str, Any],
    latents: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Atomically write data first and a complete manifest last."""

    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    if npz_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite latent cache: {npz_path} / {manifest_path}")
    if semantics.get("schema_version") != LATENT_SCHEMA_VERSION:
        raise ValueError(f"Unexpected latent semantics schema: {semantics.get('schema_version')!r}")
    expected_semantic_sha = semantics.get("semantic_sha256")
    if not isinstance(expected_semantic_sha, str) or len(expected_semantic_sha) != 64:
        raise ValueError("Latent semantics must contain semantic_sha256")
    bytes.fromhex(expected_semantic_sha)
    verify_latent_semantics(semantics)

    view_order = tuple(semantics["view_order"])
    validated = _validate_latents(latents, view_order=view_order, sampled_count=int(semantics["sampled_count"]))
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_npz = npz_path.with_name(f".{npz_path.name}.tmp-{os.getpid()}")
    temp_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    try:
        with temp_npz.open("wb") as stream:
            np.savez_compressed(stream, **validated)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_npz, npz_path)

        artifact = {
            **semantics,
            "complete": True,
            "artifact": {
                "path": npz_path.name,
                "size_bytes": npz_path.stat().st_size,
                "sha256": _file_sha256(npz_path),
                "arrays": {
                    key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in validated.items()
                },
            },
        }
        # Return the same canonical JSON value that is committed to disk.  In
        # particular, JSON represents tuple-valued encoder dimensions as lists.
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


def load_latent_artifact(
    npz_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_semantic_sha256: str,
) -> dict[str, np.ndarray]:
    """Load a cache hit only after checking semantics, bytes, keys and shapes."""

    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    if not npz_path.is_file() or not manifest_path.is_file():
        raise CacheMissError("latent npz and complete manifest must both exist")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheMissError(f"latent manifest is unreadable: {exc}") from exc
    if manifest.get("schema_version") != LATENT_SCHEMA_VERSION or manifest.get("complete") is not True:
        raise CacheMissError("latent manifest is incomplete or has the wrong schema")
    try:
        verify_latent_semantics(manifest)
    except ValueError as exc:
        raise CacheMissError(str(exc)) from exc
    if manifest.get("semantic_sha256") != expected_semantic_sha256:
        raise CacheMissError("latent semantic fingerprint changed")
    artifact = manifest.get("artifact", {})
    if artifact.get("size_bytes") != npz_path.stat().st_size or artifact.get("sha256") != _file_sha256(npz_path):
        raise CacheMissError("latent npz size or SHA-256 does not match its manifest")
    if artifact.get("path") != npz_path.name:
        raise CacheMissError("latent manifest points to a different artifact filename")

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise CacheMissError(f"latent npz is corrupt or unsafe: {exc}") from exc
    try:
        validated = _validate_latents(
            arrays,
            view_order=tuple(manifest["view_order"]),
            sampled_count=int(manifest["sampled_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheMissError(f"latent arrays violate the manifest contract: {exc}") from exc

    declared = artifact.get("arrays", {})
    actual = {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in validated.items()}
    if declared != actual:
        raise CacheMissError("latent array shape/dtype metadata does not match the npz")
    return validated


def cache_status(
    npz_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_semantic_sha256: str,
) -> CacheStatus:
    try:
        load_latent_artifact(
            npz_path,
            manifest_path,
            expected_semantic_sha256=expected_semantic_sha256,
        )
    except CacheMissError as exc:
        return CacheStatus(hit=False, reason=str(exc))
    return CacheStatus(hit=True, reason="verified semantic and byte-level cache hit")


def load_or_encode(
    npz_path: str | Path,
    manifest_path: str | Path,
    *,
    semantics: Mapping[str, Any],
    encode: Callable[[], Mapping[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], bool]:
    """Return verified cached latents or encode exactly once on a clean miss."""

    semantic_digest = str(semantics["semantic_sha256"])
    try:
        return (
            load_latent_artifact(
                npz_path,
                manifest_path,
                expected_semantic_sha256=semantic_digest,
            ),
            True,
        )
    except CacheMissError as exc:
        npz_path = Path(npz_path)
        manifest_path = Path(manifest_path)
        if npz_path.exists() or manifest_path.exists():
            raise CacheMissError(
                "A stale/corrupt cache exists; move it aside before re-encoding so provenance is not overwritten"
            ) from exc
        latents = dict(encode())
        write_latent_artifact(npz_path, manifest_path, semantics=semantics, latents=latents)
        return (
            load_latent_artifact(
                npz_path,
                manifest_path,
                expected_semantic_sha256=semantic_digest,
            ),
            False,
        )
