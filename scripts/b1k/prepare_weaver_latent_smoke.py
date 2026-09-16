from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch

from weaver.wm.encoders import SD3Encoder


VIEW_MAP = {
    # BEHAVIOR source feature -> released-WEAVER semantic slot.
    "exterior_1_left": "observation.rgb.zed_link_camera_0",
    "wrist_left": "observation.rgb.left_realsense_link_camera_0",
}


def find_numeric_artifacts(
    numeric_root: Path,
    episode_index: int,
) -> tuple[Path, Path, dict]:
    matches = list(
        (numeric_root / "episodes").glob(f"*/{episode_index:06d}.json")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one numeric JSON for episode {episode_index}, "
            f"found {len(matches)}: {matches}"
        )

    json_path = matches[0]
    npz_path = json_path.with_suffix(".npz")

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    metadata = json.loads(json_path.read_text())
    return json_path, npz_path, metadata


def find_episode_row(
    dataset_root: Path,
    episode_index: int,
) -> dict:
    meta_root = dataset_root / "meta" / "episodes"

    for parquet_path in sorted(meta_root.rglob("*.parquet")):
        table = pq.read_table(parquet_path)

        if "episode_index" not in table.column_names:
            continue

        for row in table.to_pylist():
            if int(row["episode_index"]) == episode_index:
                return row

    raise RuntimeError(
        f"Episode {episode_index} not found under {meta_root}"
    )


def resolve_video(
    dataset_root: Path,
    row: dict,
    feature: str,
) -> tuple[Path, float, float]:
    prefix = f"videos/{feature}"

    chunk = int(row[f"{prefix}/chunk_index"])
    file_index = int(row[f"{prefix}/file_index"])
    from_timestamp = float(row[f"{prefix}/from_timestamp"])
    to_timestamp = float(row[f"{prefix}/to_timestamp"])

    path = (
        dataset_root
        / "videos"
        / feature
        / f"chunk-{chunk:03d}"
        / f"file-{file_index:03d}.mp4"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return path, from_timestamp, to_timestamp


def compute_file_frame_indices(
    source_frame_indices: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    from_timestamp: float,
    to_timestamp: float,
    source_fps: float,
) -> np.ndarray:
    """
    Convert episode-local source frame IDs to packed-MP4 frame IDs.

    We calculate the mapping two independent ways:
      1. packed episode start frame + episode-local source frame index
      2. packed absolute timestamp * FPS

    Agreement is an important alignment invariant.
    """
    source_frame_indices = np.asarray(source_frame_indices, dtype=np.int64)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)

    if len(source_frame_indices) != len(timestamps_s):
        raise ValueError("source_frame_indices / timestamps_s length mismatch")

    base_frame = int(round(from_timestamp * source_fps))

    by_index = base_frame + source_frame_indices

    absolute_timestamps = from_timestamp + timestamps_s
    by_time = np.rint(absolute_timestamps * source_fps).astype(np.int64)

    error = np.abs(by_index - by_time)
    max_error = int(error.max(initial=0))

    if max_error > 1:
        raise RuntimeError(
            "Packed-video frame mapping disagreement: "
            f"max_error={max_error} frames"
        )

    if absolute_timestamps[-1] >= to_timestamp + 1.0 / source_fps:
        raise RuntimeError(
            f"Last sampled timestamp {absolute_timestamps[-1]:.6f} "
            f"falls outside episode video interval ending at "
            f"{to_timestamp:.6f}"
        )

    return by_index


def decode_selected_frames(
    video_path: Path,
    frame_indices: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Smoke-test implementation: sequentially decode the packed video and
    retain exactly the requested packed-file frame indices.

    This is intentionally simple and alignment-safe. We can optimize
    seeking/sharding later if full-dataset latent generation is warranted.
    """
    wanted = [int(x) for x in frame_indices]
    wanted_set = set(wanted)

    if not wanted:
        raise ValueError("No requested video frames")

    max_index = max(wanted)
    found: dict[int, np.ndarray] = {}

    height, width = image_size

    for frame_idx, frame in enumerate(
        iio.imiter(video_path, plugin="FFMPEG")
    ):
        if frame_idx in wanted_set:
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise RuntimeError(
                    f"Unexpected RGB frame shape {frame.shape}"
                )

            # Exact upstream WEAVER preprocessing semantics.
            resized = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            found[frame_idx] = resized

        if frame_idx >= max_index:
            break

    missing = [x for x in wanted if x not in found]
    if missing:
        raise RuntimeError(
            f"Failed to decode {len(missing)} requested frames from "
            f"{video_path}; first missing={missing[:10]}"
        )

    return np.stack([found[x] for x in wanted], axis=0)


@torch.no_grad()
def encode_camera_frames(
    encoder: torch.nn.Module,
    frames: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    # Exact semantics of upstream datasets/preprocess_droid.py.
    frame_tensor = torch.from_numpy(
        frames.astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2)

    encoded = []

    for start in range(0, len(frame_tensor), batch_size):
        batch = frame_tensor[start : start + batch_size].to(device)
        features = encoder(batch)
        encoded.append(features.cpu().numpy())

    return np.concatenate(encoded, axis=0)


@torch.no_grad()
def write_reconstruction_sheet(
    encoder: SD3Encoder,
    frames: np.ndarray,
    features: np.ndarray,
    output_path: Path,
    *,
    device: str,
) -> None:
    """
    Save several:
        original resized RGB | SD3 reconstruction
    rows for visual inspection.
    """
    count = min(6, len(frames))

    sample_indices = np.linspace(
        0,
        len(frames) - 1,
        num=count,
        dtype=np.int64,
    )

    latent = torch.from_numpy(
        features[sample_indices].astype(np.float32)
    ).to(device)

    decoded = encoder.decode(latent)

    # Different diffusers paths may expose decoded images in [-1,1]
    # or already-clamped form. Handle both for debug visualization.
    if float(decoded.min()) < -0.05:
        decoded = decoded / 2.0 + 0.5

    decoded = (
        decoded.clamp(0, 1)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )

    decoded = np.rint(decoded * 255).astype(np.uint8)

    rows = []
    for local_i, frame_i in enumerate(sample_indices):
        original = frames[frame_i]
        recon = decoded[local_i]

        if recon.shape[:2] != original.shape[:2]:
            recon = cv2.resize(
                recon,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        rows.append(np.concatenate([original, recon], axis=1))

    sheet = np.concatenate(rows, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # cv2.imwrite expects BGR.
    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--numeric-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument("--episode-index", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    image_size = (192, 320)
    source_fps = 30.0
    precompute_spatial_size = 4

    json_path, npz_path, numeric_meta = find_numeric_artifacts(
        args.numeric_root,
        args.episode_index,
    )

    episode_row = find_episode_row(
        args.dataset_root,
        args.episode_index,
    )

    split = numeric_meta["split"]
    prompt = numeric_meta["prompt"]

    with np.load(npz_path) as numeric:
        source_frame_indices = numeric[
            "source_frame_indices"
        ].astype(np.int64)

        timestamps_s = numeric["timestamps_s"].astype(np.float64)

        states = numeric["states"]
        actions = numeric["state_delta_actions"]
        rewards = numeric["rewards"]
        transition_valid = numeric["transition_valid"]

        T = len(source_frame_indices)

        expected_lengths = {
            "timestamps": len(timestamps_s),
            "states": len(states),
            "actions": len(actions),
            "rewards": len(rewards),
            "transition_valid": len(transition_valid),
        }

        for name, length in expected_lengths.items():
            if length != T:
                raise RuntimeError(
                    f"{name} length={length}, expected T={T}"
                )

        if states.shape[1:] != (23,):
            raise RuntimeError(
                f"Expected states [T,23], got {states.shape}"
            )

        if actions.shape[1:] != (23,):
            raise RuntimeError(
                f"Expected state_delta_actions [T,23], "
                f"got {actions.shape}"
            )

    print("===== episode =====")
    print("episode_index:", args.episode_index)
    print("split:", split)
    print("prompt:", prompt)
    print("T:", T)

    print("\n===== loading SD3 VAE =====")

    encoder = SD3Encoder(
        model_name="stabilityai/stable-diffusion-3-medium-diffusers",
        image_size=image_size,
        spatial_size=precompute_spatial_size,
        device=args.device,
    ).eval()

    latent_arrays: dict[str, np.ndarray] = {}
    frame_index_arrays: dict[str, np.ndarray] = {}
    provenance_views: dict[str, dict] = {}

    debug_root = (
        args.output_root
        / "debug"
        / split
        / f"{args.episode_index:06d}"
    )

    for weaver_key, b1k_feature in VIEW_MAP.items():
        print(f"\n===== {weaver_key} =====")

        video_path, from_ts, to_ts = resolve_video(
            args.dataset_root,
            episode_row,
            b1k_feature,
        )

        print("source feature:", b1k_feature)
        print("video:", video_path)
        print("interval:", from_ts, "->", to_ts)

        packed_indices = compute_file_frame_indices(
            source_frame_indices,
            timestamps_s,
            from_timestamp=from_ts,
            to_timestamp=to_ts,
            source_fps=source_fps,
        )

        print(
            "packed frame range:",
            int(packed_indices[0]),
            "->",
            int(packed_indices[-1]),
        )

        frames = decode_selected_frames(
            video_path,
            packed_indices,
            image_size,
        )

        print("decoded:", frames.shape, frames.dtype)

        features = encode_camera_frames(
            encoder,
            frames,
            batch_size=args.batch_size,
            device=args.device,
        )

        print(
            "encoded:",
            features.shape,
            features.dtype,
            "min/max=",
            float(features.min()),
            float(features.max()),
        )

        expected_shape = (T, 60, 256)
        if features.shape != expected_shape:
            raise RuntimeError(
                f"{weaver_key}: expected latent shape "
                f"{expected_shape}, got {features.shape}"
            )

        if not np.isfinite(features).all():
            raise RuntimeError(
                f"{weaver_key}: non-finite latent values"
            )

        # Save storage-friendly FP16 only after validating FP32 output.
        features_fp16 = features.astype(np.float16)

        latent_arrays[weaver_key] = features_fp16
        frame_index_arrays[
            f"{weaver_key}_packed_frame_indices"
        ] = packed_indices

        provenance_views[weaver_key] = {
            "b1k_feature": b1k_feature,
            "video_path": str(video_path),
            "from_timestamp": from_ts,
            "to_timestamp": to_ts,
            "latent_shape": list(features_fp16.shape),
            "latent_dtype": str(features_fp16.dtype),
        }

        write_reconstruction_sheet(
            encoder,
            frames,
            features,
            debug_root / f"{weaver_key}_reconstruction.jpg",
            device=args.device,
        )

    latent_path = (
        args.output_root
        / "latents"
        / split
        / f"{args.episode_index:06d}.npz"
    )
    latent_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        latent_path,
        **latent_arrays,
        **frame_index_arrays,
        source_frame_indices=source_frame_indices,
        timestamps_s=timestamps_s.astype(np.float32),
    )

    annotation = {
        "schema_version": "b1k_weaver_latent_smoke_v1",
        "episode_index": args.episode_index,
        "split": split,
        "task_names": numeric_meta["task_names"],
        "prompt": prompt,
        "length": T,
        "numeric_json_path": str(json_path),
        "numeric_npz_path": str(npz_path),
        "latent_path": str(latent_path),
        "image_size": list(image_size),
        "source_fps": source_fps,
        "target_hz": numeric_meta["target_hz"],
        "precompute_spatial_size": precompute_spatial_size,
        "views": provenance_views,
    }

    annotation_path = (
        args.output_root
        / "annotations"
        / split
        / f"{args.episode_index:06d}.json"
    )
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        json.dumps(annotation, indent=2, ensure_ascii=False)
    )

    print("\n===== DONE =====")
    print("latent:", latent_path)
    print("annotation:", annotation_path)
    print("debug:", debug_root)


if __name__ == "__main__":
    main()
