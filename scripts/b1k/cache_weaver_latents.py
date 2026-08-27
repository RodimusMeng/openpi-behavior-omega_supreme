"""Encode verified B1K frame artifacts with the released WEAVER SD3 VAE.

Run this second stage inside the frozen official WEAVER Python 3.11
environment. It intentionally imports neither LeRobot nor Tyro: decoded
uint8 RGB is supplied by ``export_weaver_frames.py`` from the OmegaSupreme
environment.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as functional
import yaml

from openpi.weaver import b1k_dataset_adapter as _episode_adapter
from openpi.weaver import behavior_precomputed
from openpi.weaver import frame_cache
from openpi.weaver import latent_cache


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_config: str
    encoder_snapshot: str
    encoder_revision: str
    adapter_git_sha: str
    numeric_dir: str = "outputs/weaver/b1k-v3again"
    frame_dir: str | None = None
    split: Literal["train", "val", "test"] = "train"
    episode_indices: tuple[int, ...] | None = None
    max_episodes: int | None = None
    batch_size: int = 4
    device: str = "cuda"
    resize_mode: Literal["letterbox", "center-crop", "stretch"] = "letterbox"
    storage_dtype: Literal["float16", "float32"] = "float16"
    dry_run: bool = False
    verify_cache_only: bool = False


@dataclasses.dataclass(frozen=True)
class _WorkItem:
    ordinal: int
    manifest: dict[str, Any]
    numeric: behavior_precomputed.NumericEpisode
    semantics: dict[str, Any]
    npz_path: Path
    manifest_path: Path


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-config", required=True)
    parser.add_argument("--encoder-snapshot", required=True)
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--adapter-git-sha", required=True)
    parser.add_argument("--numeric-dir", default="outputs/weaver/b1k-v3again")
    parser.add_argument("--frame-dir")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--episode-indices", nargs="+", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resize-mode",
        choices=("letterbox", "center-crop", "stretch"),
        default="letterbox",
    )
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-cache-only", action="store_true")
    parsed = parser.parse_args(argv)
    episode_indices = None if parsed.episode_indices is None else tuple(parsed.episode_indices)
    return Args(
        checkpoint_config=parsed.checkpoint_config,
        encoder_snapshot=parsed.encoder_snapshot,
        encoder_revision=parsed.encoder_revision,
        adapter_git_sha=parsed.adapter_git_sha,
        numeric_dir=parsed.numeric_dir,
        frame_dir=parsed.frame_dir,
        split=parsed.split,
        episode_indices=episode_indices,
        max_episodes=parsed.max_episodes,
        batch_size=parsed.batch_size,
        device=parsed.device,
        resize_mode=parsed.resize_mode,
        storage_dtype=parsed.storage_dtype,
        dry_run=parsed.dry_run,
        verify_cache_only=parsed.verify_cache_only,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_contract(path: Path) -> tuple[tuple[str, ...], tuple[int, int], str, int]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint config must be a YAML object: {path}")
    cfg = value.get("defaults", value)
    try:
        image_keys = tuple(cfg["dataset"]["img_keys"])
        image_size = tuple(int(item) for item in cfg["dataset"]["image_size"])
        model_id = str(cfg["im_encoder"]["name"])
        spatial_size = int(cfg["im_encoder"]["spatial_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint config lacks dataset/im_encoder contract: {path}") from exc
    if len(image_size) != 2:
        raise ValueError(f"Checkpoint image_size must have two dimensions, got {image_size}")
    if image_keys != behavior_precomputed.DEFAULT_IMAGE_KEYS:
        raise ValueError(
            "First B1K transfer requires the released two-view order "
            f"{behavior_precomputed.DEFAULT_IMAGE_KEYS}, got {image_keys}"
        )
    return image_keys, image_size, model_id, spatial_size


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


def _resize_batch(batch: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    target_h, target_w = size
    source_h, source_w = batch.shape[-2:]
    if mode == "stretch":
        return functional.interpolate(batch, size=size, mode="bilinear", align_corners=False)
    scale = (
        min(target_h / source_h, target_w / source_w)
        if mode == "letterbox"
        else max(target_h / source_h, target_w / source_w)
    )
    raw_h = max(1, round(source_h * scale))
    raw_w = max(1, round(source_w * scale))
    if mode == "letterbox":
        resized_h, resized_w = min(target_h, raw_h), min(target_w, raw_w)
    else:
        resized_h, resized_w = max(target_h, raw_h), max(target_w, raw_w)
    resized = functional.interpolate(
        batch,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    if mode == "letterbox":
        pad_h = target_h - resized_h
        pad_w = target_w - resized_w
        return functional.pad(
            resized,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            value=0.0,
        )
    top = (resized_h - target_h) // 2
    left = (resized_w - target_w) // 2
    cropped = resized[:, :, top : top + target_h, left : left + target_w]
    if cropped.shape[-2:] != (target_h, target_w):
        raise RuntimeError(f"center-crop produced wrong shape: {tuple(cropped.shape)}")
    return cropped


def _encode_episode(
    frame_root: Path,
    numeric: behavior_precomputed.NumericEpisode,
    manifest: dict[str, Any],
    *,
    image_keys: tuple[str, ...],
    encoder_factory: Callable[[], torch.nn.Module],
    image_size: tuple[int, int],
    resize_mode: str,
    batch_size: int,
    device: torch.device,
    storage_dtype: str,
) -> dict[str, np.ndarray]:
    episode_frames = frame_cache.load_episode_frames(
        frame_root,
        manifest,
        source_absolute_indices=numeric.source_absolute_indices,
        view_order=image_keys,
    )
    encoder = encoder_factory()
    chunks: dict[str, list[np.ndarray]] = {f"{image_key}_features": [] for image_key in image_keys}
    with torch.inference_mode():
        for start in range(0, numeric.length, batch_size):
            stop = min(numeric.length, start + batch_size)
            for image_key in image_keys:
                uint8_frames = np.array(episode_frames.frames[image_key][start:stop], copy=True)
                frames = torch.from_numpy(uint8_frames).permute(0, 3, 1, 2).to(dtype=torch.float32)
                frames = frames.div_(255.0)
                frames = _resize_batch(frames, image_size, resize_mode).to(device)
                features = encoder(frames).detach().cpu().numpy()
                if features.ndim != 3 or features.shape[0] != stop - start:
                    raise ValueError(f"{image_key} encoder output must be (B,P,D), got {features.shape}")
                chunks[f"{image_key}_features"].append(features)
    dtype = np.float16 if storage_dtype == "float16" else np.float32
    arrays = {key: np.concatenate(values, axis=0).astype(dtype, copy=False) for key, values in chunks.items()}
    reference_shape = next(iter(arrays.values())).shape
    if any(value.shape != reference_shape for value in arrays.values()):
        shapes = [value.shape for value in arrays.values()]
        raise ValueError(f"Encoded views do not share one latent shape: {shapes}")
    return arrays


def _device_telemetry_start(device: torch.device) -> tuple[dict[str, Any], float]:
    telemetry: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is false: {device}")
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        telemetry.update({"gpu_name": properties.name, "gpu_total_bytes": int(properties.total_memory)})
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    return telemetry, time.perf_counter()


def _device_telemetry_finish(
    telemetry: dict[str, Any],
    started_at: float,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        telemetry.update(
            {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    telemetry["elapsed_seconds"] = time.perf_counter() - started_at
    return telemetry


def main(args: Args) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not args.adapter_git_sha.strip():
        raise ValueError("adapter_git_sha must be a committed OmegaSupreme revision")
    numeric_root = Path(args.numeric_dir).expanduser().resolve()
    frame_root = Path(args.frame_dir).expanduser().resolve() if args.frame_dir is not None else numeric_root / "frames"
    checkpoint_config = Path(args.checkpoint_config).expanduser().resolve()
    encoder_snapshot = Path(args.encoder_snapshot).expanduser().resolve()
    vae_snapshot = encoder_snapshot / "vae"
    if not checkpoint_config.is_file():
        raise FileNotFoundError(f"Missing downloaded checkpoint config: {checkpoint_config}")
    if not vae_snapshot.is_dir():
        raise FileNotFoundError(f"Pinned SD3 snapshot has no vae/ directory: {vae_snapshot}")
    if not (vae_snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Pinned SD3 VAE snapshot has no config.json: {vae_snapshot}")

    image_keys, image_size, model_id, spatial_size = _checkpoint_contract(checkpoint_config)
    manifests = _read_numeric_manifests(numeric_root, args)
    selected_features = tuple(manifests[0]["views"][image_key]["rgb"] for image_key in image_keys)
    if any(
        tuple(manifest["views"][image_key]["rgb"] for image_key in image_keys) != selected_features
        for manifest in manifests
    ):
        raise ValueError("Selected numeric manifests do not share one camera mapping")

    encoder_spec = latent_cache.EncoderSpec(
        model_id=model_id,
        revision=args.encoder_revision,
        image_size=image_size,
        spatial_size=spatial_size,
        compute_dtype="float32",
        storage_dtype=args.storage_dtype,
        vae_snapshot_sha256=latent_cache.directory_sha256(vae_snapshot),
        checkpoint_config_sha256=_sha256(checkpoint_config),
        resize_mode=args.resize_mode,
    )
    work_items: list[_WorkItem] = []
    for ordinal, manifest in enumerate(manifests, start=1):
        numeric = behavior_precomputed.load_numeric_episode(numeric_root, manifest)
        verified_frames = frame_cache.load_episode_frames(
            frame_root,
            manifest,
            source_absolute_indices=numeric.source_absolute_indices,
            view_order=image_keys,
        )
        if verified_frames.manifest is None:
            raise RuntimeError("Verified frame artifact did not return its manifest")
        semantics = latent_cache.latent_semantics(
            manifest,
            frame_manifest=verified_frames.manifest,
            encoder=encoder_spec,
            view_order=image_keys,
            adapter_git_sha=args.adapter_git_sha,
        )
        episode_index = int(manifest["episode_index"])
        npz_path = numeric_root / "latents" / args.split / f"{episode_index:06d}.npz"
        work_items.append(
            _WorkItem(
                ordinal=ordinal,
                manifest=manifest,
                numeric=numeric,
                semantics=semantics,
                npz_path=npz_path,
                manifest_path=npz_path.with_suffix(".json"),
            )
        )

    print(
        json.dumps(
            {
                "numeric_root": str(numeric_root),
                "frame_root": str(frame_root),
                "episodes": len(work_items),
                "split": args.split,
                "view_order": image_keys,
                "selected_video_features": selected_features,
                "encoder": dataclasses.asdict(encoder_spec),
                "encoder_snapshot": str(encoder_snapshot),
                "dry_run": args.dry_run,
                "verify_cache_only": args.verify_cache_only,
            },
            indent=2,
        ),
        flush=True,
    )

    pending: list[_WorkItem] = []
    for work in work_items:
        try:
            arrays = latent_cache.load_latent_artifact(
                work.npz_path,
                work.manifest_path,
                expected_semantic_sha256=work.semantics["semantic_sha256"],
            )
        except latent_cache.CacheMissError as exc:
            if work.npz_path.exists() or work.manifest_path.exists():
                raise latent_cache.CacheMissError(
                    f"Episode {work.manifest['episode_index']} has a stale/corrupt latent cache; "
                    "move it aside instead of overwriting it"
                ) from exc
            pending.append(work)
        else:
            print(
                f"[{work.ordinal}/{len(work_items)}] episode={work.manifest['episode_index']} "
                f"frames={work.numeric.length} shape={next(iter(arrays.values())).shape} cached=True",
                flush=True,
            )

    if args.verify_cache_only:
        if pending:
            missing = [int(work.manifest["episode_index"]) for work in pending]
            raise latent_cache.CacheMissError(f"Verified latent cache misses remain for episodes: {missing}")
        print("Every selected latent artifact is a verified semantic and byte-level cache hit.", flush=True)
        return

    if args.dry_run:
        for work in work_items:
            print(
                f"frame-cache episode={work.manifest['episode_index']} frames={work.numeric.length} verified=True",
                flush=True,
            )
        print("Dry run verified numeric and frame artifacts; SD3 was not imported or constructed.", flush=True)
        return
    if not pending:
        print("No cache misses; SD3 was not imported or constructed.", flush=True)
        return

    device = torch.device(args.device)
    telemetry, started_at = _device_telemetry_start(device)
    encoder: torch.nn.Module | None = None

    def get_encoder() -> torch.nn.Module:
        nonlocal encoder
        if encoder is None:
            try:
                from weaver.wm.encoders import SD3Encoder  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "Official WEAVER is not importable. Add the frozen WEAVER checkout to PYTHONPATH "
                    "and run this script in its Python 3.11 environment."
                ) from exc
            encoder = SD3Encoder(
                model_name=str(encoder_snapshot),
                image_size=image_size,
                spatial_size=spatial_size,
                device=str(device),
            ).eval()
            floating_dtypes = {parameter.dtype for parameter in encoder.parameters() if parameter.is_floating_point()}
            if floating_dtypes != {torch.float32}:
                dtypes = sorted(map(str, floating_dtypes))
                raise ValueError(f"SD3 VAE compute dtype must be float32, got {dtypes}")
        return encoder

    encoded_episodes = 0
    encoded_frames = 0
    try:
        for work in pending:
            arrays, cached = latent_cache.load_or_encode(
                work.npz_path,
                work.manifest_path,
                semantics=work.semantics,
                encode=lambda work=work: _encode_episode(
                    frame_root,
                    work.numeric,
                    work.manifest,
                    image_keys=image_keys,
                    encoder_factory=get_encoder,
                    image_size=image_size,
                    resize_mode=args.resize_mode,
                    batch_size=args.batch_size,
                    device=device,
                    storage_dtype=args.storage_dtype,
                ),
            )
            if not cached:
                encoded_episodes += 1
                encoded_frames += work.numeric.length
            print(
                f"[{work.ordinal}/{len(work_items)}] episode={work.manifest['episode_index']} "
                f"frames={work.numeric.length} shape={next(iter(arrays.values())).shape} cached={cached}",
                flush=True,
            )
    finally:
        telemetry.update(
            {
                "encoder_constructed": encoder is not None,
                "encoded_episodes": encoded_episodes,
                "encoded_frames": encoded_frames,
                "batch_size": args.batch_size,
                "compute_dtype": "float32",
                "storage_dtype": args.storage_dtype,
            }
        )
        print(
            json.dumps(
                {"telemetry": _device_telemetry_finish(telemetry, started_at, device=device)},
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main(_parse_args())
