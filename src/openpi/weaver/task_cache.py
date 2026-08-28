"""Auditable released-WEAVER task-encoder features for the B1K task catalog."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from openpi.weaver.b1k_dataset_adapter import semantic_sha256

TASK_PROMPT_SCHEMA_VERSION = "behavior_task_prompts_v1"
TASK_FEATURE_SCHEMA_VERSION = "b1k_weaver_task_features_v1"
TASK_FEATURE_SEMANTIC_FIELDS = (
    "schema_version",
    "task_prompt_schema_version",
    "dataset_metadata_sha256",
    "task_catalog_sha256",
    "tasks",
    "encoder",
    "adapter_git_sha",
)


class TaskCacheMissError(RuntimeError):
    """Raised when task features are absent, stale, corrupt, or incomplete."""


@dataclasses.dataclass(frozen=True)
class TaskEncoderSpec:
    """Immutable identity of the released task-encoding path."""

    model_id: str
    revision: str
    feature_dim: int
    snapshot_sha256: str
    upstream_git_sha: str
    pooling: str = "pooler_output"
    max_length: int = 77
    compute_dtype: str = "float32"
    storage_dtype: str = "float32"

    def __post_init__(self) -> None:
        for name, value in (
            ("model_id", self.model_id),
            ("revision", self.revision),
            ("upstream_git_sha", self.upstream_git_sha),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if isinstance(self.feature_dim, bool) or not isinstance(self.feature_dim, int) or self.feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if len(self.snapshot_sha256) != 64:
            raise ValueError("snapshot_sha256 must be a SHA-256 digest")
        bytes.fromhex(self.snapshot_sha256)
        if self.pooling != "pooler_output" or self.max_length != 77:
            raise ValueError("First transfer must preserve released WEAVER pooler_output/max_length=77")
        if self.compute_dtype != "float32" or self.storage_dtype != "float32":
            raise ValueError("Released task features must use float32 compute and storage")


@dataclasses.dataclass(frozen=True)
class LoadedTaskFeatures:
    by_task_name: Mapping[str, np.ndarray]
    by_prompt: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_task_prompt_catalog(path: str | Path) -> dict[str, Any]:
    """Read the numeric pipeline's authoritative task catalog."""

    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read task prompt catalog {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != TASK_PROMPT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported task prompt catalog: {path}")
    for field in ("dataset_metadata_sha256", "task_catalog_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Task prompt catalog has no valid {field}")
        bytes.fromhex(digest)
    prompts = value.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("Task prompt catalog must contain a non-empty prompts object")
    if any(not isinstance(name, str) or not name.strip() for name in prompts):
        raise ValueError("Every task name must be a non-empty string")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts.values()):
        raise ValueError("Every task prompt must be a non-empty string")
    value["prompts"] = {name: prompts[name] for name in sorted(prompts)}
    return value


def task_feature_semantics(
    prompt_catalog: Mapping[str, Any],
    *,
    encoder: TaskEncoderSpec,
    adapter_git_sha: str,
) -> dict[str, Any]:
    """Build the semantic identity of a complete task-feature matrix."""

    if prompt_catalog.get("schema_version") != TASK_PROMPT_SCHEMA_VERSION:
        raise ValueError("Task feature semantics require behavior_task_prompts_v1")
    prompts = prompt_catalog.get("prompts")
    if not isinstance(prompts, Mapping) or not prompts:
        raise ValueError("Task feature semantics require a non-empty prompt mapping")
    if not adapter_git_sha.strip():
        raise ValueError("adapter_git_sha must be recorded")
    tasks = []
    for name in sorted(prompts):
        prompt = prompts[name]
        if not isinstance(name, str) or not name.strip() or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Task names and prompts must be non-empty strings")
        tasks.append(
            {
                "name": name,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    semantics = {
        "schema_version": TASK_FEATURE_SCHEMA_VERSION,
        "task_prompt_schema_version": TASK_PROMPT_SCHEMA_VERSION,
        "dataset_metadata_sha256": prompt_catalog["dataset_metadata_sha256"],
        "task_catalog_sha256": prompt_catalog["task_catalog_sha256"],
        "tasks": tasks,
        "encoder": dataclasses.asdict(encoder),
        "adapter_git_sha": adapter_git_sha,
    }
    semantics["semantic_sha256"] = semantic_sha256(semantics)
    return semantics


def verify_task_feature_semantics(manifest: Mapping[str, Any]) -> str:
    try:
        semantics = {field: manifest[field] for field in TASK_FEATURE_SEMANTIC_FIELDS}
    except KeyError as exc:
        raise ValueError(f"Task-feature manifest is missing semantic field {exc.args[0]!r}") from exc
    actual = semantic_sha256(semantics)
    if manifest.get("semantic_sha256") != actual:
        raise ValueError("Task-feature semantic SHA-256 mismatch")
    return actual


def _feature_matrix(
    features: Mapping[str, Any],
    *,
    tasks: list[Mapping[str, Any]],
    feature_dim: int,
) -> np.ndarray:
    names = tuple(str(task["name"]) for task in tasks)
    if set(features) != set(names):
        raise ValueError(f"Task feature names differ: expected={sorted(names)}, got={sorted(features)}")
    rows = []
    for name in names:
        row = np.asarray(features[name])
        if row.shape != (feature_dim,) or not np.issubdtype(row.dtype, np.floating):
            raise ValueError(f"Task feature {name!r} must be floating ({feature_dim},), got {row.shape}/{row.dtype}")
        if not np.isfinite(row).all():
            raise ValueError(f"Task feature {name!r} contains non-finite values")
        rows.append(np.asarray(row, dtype=np.float32))
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)


def write_task_feature_artifact(
    root: str | Path,
    *,
    semantics: Mapping[str, Any],
    features: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically commit one catalog-wide feature matrix and manifest."""

    root = Path(root)
    npz_path = root / "features.npz"
    manifest_path = root / "manifest.json"
    if npz_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite task-feature cache: {root}")
    if semantics.get("schema_version") != TASK_FEATURE_SCHEMA_VERSION:
        raise ValueError("Wrong task-feature schema")
    verify_task_feature_semantics(semantics)
    encoder = semantics.get("encoder")
    if not isinstance(encoder, Mapping):
        raise ValueError("Task-feature semantics have no encoder object")
    matrix = _feature_matrix(
        features,
        tasks=list(semantics["tasks"]),
        feature_dim=int(encoder["feature_dim"]),
    )
    root.mkdir(parents=True, exist_ok=True)
    temp_npz = root / f".features.npz.tmp-{os.getpid()}"
    temp_manifest = root / f".manifest.json.tmp-{os.getpid()}"
    try:
        with temp_npz.open("wb") as stream:
            np.savez_compressed(stream, features=matrix)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_npz, npz_path)
        manifest = {
            **semantics,
            "complete": True,
            "artifact": {
                "path": npz_path.name,
                "size_bytes": npz_path.stat().st_size,
                "sha256": _file_sha256(npz_path),
                "array": {"shape": list(matrix.shape), "dtype": matrix.dtype.str},
            },
        }
        manifest = json.loads(json.dumps(manifest, allow_nan=False, ensure_ascii=False))
        with temp_manifest.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_manifest, manifest_path)
        return manifest
    finally:
        temp_npz.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)


def load_task_feature_artifact(
    root: str | Path,
    *,
    prompt_catalog: Mapping[str, Any],
) -> LoadedTaskFeatures:
    """Load features only after catalog, semantic, byte, and shape checks."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    npz_path = root / "features.npz"
    if not manifest_path.is_file() or not npz_path.is_file():
        raise TaskCacheMissError("task-feature manifest and npz must both exist")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskCacheMissError(f"Task-feature manifest is unreadable: {exc}") from exc
    if manifest.get("schema_version") != TASK_FEATURE_SCHEMA_VERSION or manifest.get("complete") is not True:
        raise TaskCacheMissError("Task-feature manifest is incomplete or has the wrong schema")
    try:
        verify_task_feature_semantics(manifest)
    except ValueError as exc:
        raise TaskCacheMissError(str(exc)) from exc
    try:
        expected = task_feature_semantics(
            prompt_catalog,
            encoder=TaskEncoderSpec(**manifest["encoder"]),
            adapter_git_sha=str(manifest["adapter_git_sha"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskCacheMissError(f"Task-feature encoder/catalog semantics are invalid: {exc}") from exc
    if manifest.get("semantic_sha256") != expected["semantic_sha256"]:
        raise TaskCacheMissError("Task-feature cache belongs to a different prompt catalog")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise TaskCacheMissError("Task-feature manifest has no artifact object")
    if artifact.get("path") != npz_path.name:
        raise TaskCacheMissError("Task-feature manifest points to a different filename")
    if artifact.get("size_bytes") != npz_path.stat().st_size or artifact.get("sha256") != _file_sha256(npz_path):
        raise TaskCacheMissError("Task-feature NPZ bytes do not match the manifest")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if archive.files != ["features"]:
                raise ValueError(f"unexpected arrays {archive.files}")
            matrix = np.array(archive["features"], copy=True)
    except (OSError, ValueError, KeyError) as exc:
        raise TaskCacheMissError(f"Task-feature NPZ is corrupt or unsafe: {exc}") from exc
    tasks = list(manifest["tasks"])
    encoder = manifest["encoder"]
    expected_shape = (len(tasks), int(encoder["feature_dim"]))
    if matrix.shape != expected_shape or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise TaskCacheMissError(
            f"Task-feature matrix must be finite float32 {expected_shape}, got {matrix.shape}/{matrix.dtype}"
        )
    declared = artifact.get("array")
    if declared != {"shape": list(matrix.shape), "dtype": matrix.dtype.str}:
        raise TaskCacheMissError("Task-feature array metadata does not match the NPZ")
    by_name: dict[str, np.ndarray] = {}
    by_prompt: dict[str, np.ndarray] = {}
    for task, row in zip(tasks, matrix, strict=True):
        immutable = np.ascontiguousarray(row, dtype=np.float32)
        immutable.setflags(write=False)
        by_name[str(task["name"])] = immutable
        prompt = str(task["prompt"])
        if prompt in by_prompt and not np.array_equal(by_prompt[prompt], immutable):
            raise TaskCacheMissError("Identical prompts received different task features")
        by_prompt[prompt] = immutable
    return LoadedTaskFeatures(by_task_name=by_name, by_prompt=by_prompt, manifest=manifest)


def task_feature_provenance(root: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return artifact identity plus its diagnostic local path."""

    root = Path(root).resolve()
    artifact = manifest.get("artifact", {})
    return {
        "path": str(root),
        "schema_version": manifest.get("schema_version"),
        "semantic_sha256": manifest.get("semantic_sha256"),
        "artifact_sha256": artifact.get("sha256"),
        "encoder": manifest.get("encoder"),
    }
