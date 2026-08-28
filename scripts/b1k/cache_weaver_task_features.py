"""Encode the official B1K prompt catalog with released WEAVER's task encoder.

Run this in the pinned upstream WEAVER environment.  The output is one small
catalog-wide artifact, not 20,000 duplicated per-episode embeddings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.weaver import latent_cache
from openpi.weaver import task_cache


@dataclasses.dataclass(frozen=True)
class Args:
    numeric_dir: str
    encoder_snapshot: str
    encoder_revision: str
    upstream_git_sha: str
    adapter_git_sha: str
    output_dir: str | None = None
    batch_size: int = 32
    device: str = "cuda"
    dry_run: bool = False
    verify_cache_only: bool = False


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-dir", required=True)
    parser.add_argument("--encoder-snapshot", required=True)
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--upstream-git-sha", required=True)
    parser.add_argument("--adapter-git-sha", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-cache-only", action="store_true")
    values = parser.parse_args(argv)
    return Args(**vars(values))


def _resolved_model_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def _assert_loaded_snapshot(task_encoder: Any, expected: Path) -> None:
    """Prove that both upstream CLIP components loaded the hashed local snapshot."""

    text_encoder = getattr(task_encoder, "text_encoder", None)
    tokenizer = getattr(task_encoder, "tokenizer", None)
    model_path = _resolved_model_path(getattr(getattr(text_encoder, "config", None), "_name_or_path", None))
    tokenizer_path = _resolved_model_path(getattr(tokenizer, "name_or_path", None))
    expected = expected.resolve()
    if model_path != expected or tokenizer_path != expected:
        raise ValueError(
            "Released task encoder did not load model and tokenizer from the pinned local snapshot: "
            f"expected={str(expected)!r}, model={str(model_path)!r}, tokenizer={str(tokenizer_path)!r}"
        )


def _encode_catalog(task_encoder: Any, prompts: dict[str, str], *, batch_size: int) -> dict[str, np.ndarray]:
    names = tuple(sorted(prompts))
    result: dict[str, np.ndarray] = {}
    for start in range(0, len(names), batch_size):
        selected = names[start : start + batch_size]
        encoded = task_encoder([prompts[name] for name in selected]).detach().cpu().numpy()
        if encoded.ndim != 2 or encoded.shape[0] != len(selected):
            raise ValueError(f"Task encoder output must be (B,D), got {encoded.shape}")
        for name, row in zip(selected, encoded, strict=True):
            result[name] = np.asarray(row, dtype=np.float32)
    return result


def main(args: Args) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for name, value in (
        ("encoder_revision", args.encoder_revision),
        ("upstream_git_sha", args.upstream_git_sha),
        ("adapter_git_sha", args.adapter_git_sha),
    ):
        if not value.strip():
            raise ValueError(f"{name} must be non-empty")

    numeric_root = Path(args.numeric_dir).expanduser().resolve()
    snapshot = Path(args.encoder_snapshot).expanduser().resolve()
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else numeric_root / "task_features"
    )
    catalog = task_cache.read_task_prompt_catalog(numeric_root / "task_prompts.json")
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Pinned task-encoder snapshot is incomplete: {snapshot}")

    summary = {
        "numeric_root": str(numeric_root),
        "output_root": str(output_root),
        "tasks": len(catalog["prompts"]),
        "encoder_snapshot": str(snapshot),
        "encoder_revision": args.encoder_revision,
        "dry_run": args.dry_run,
        "verify_cache_only": args.verify_cache_only,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if args.dry_run:
        print("Dry run verified the task catalog and pinned encoder snapshot; encoder was not constructed.")
        return

    cache_paths = (output_root / "features.npz", output_root / "manifest.json")
    if any(path.exists() for path in cache_paths):
        loaded = task_cache.load_task_feature_artifact(output_root, prompt_catalog=catalog)
        encoder = loaded.manifest["encoder"]
        expected = {
            "revision": args.encoder_revision,
            "upstream_git_sha": args.upstream_git_sha,
            "snapshot_sha256": latent_cache.directory_sha256(snapshot),
        }
        mismatched = {key: (encoder.get(key), value) for key, value in expected.items() if encoder.get(key) != value}
        if mismatched:
            raise task_cache.TaskCacheMissError(f"Task-feature encoder provenance changed: {mismatched}")
        print(
            json.dumps(
                {
                    "cached": True,
                    "shape": [len(loaded.by_task_name), int(encoder["feature_dim"])],
                    "semantic_sha256": loaded.manifest["semantic_sha256"],
                },
                indent=2,
            ),
            flush=True,
        )
        return
    if args.verify_cache_only:
        raise task_cache.TaskCacheMissError(f"Task-feature cache is missing: {output_root}")

    try:
        from weaver.wm.encoders import ClipEncoder  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Official WEAVER is not importable; run in its pinned Python environment with both checkouts on PYTHONPATH"
        ) from exc
    task_encoder = ClipEncoder(model_name=str(snapshot), device=args.device).to(args.device).eval()
    _assert_loaded_snapshot(task_encoder, snapshot)
    feature_dim = int(task_encoder.feature_dim)
    if feature_dim != 768:
        raise ValueError(f"Released WEAVER task feature_dim must be 768, got {feature_dim}")
    features = _encode_catalog(task_encoder, dict(catalog["prompts"]), batch_size=args.batch_size)
    encoder_spec = task_cache.TaskEncoderSpec(
        model_id="openai/clip-vit-large-patch14",
        revision=args.encoder_revision,
        feature_dim=feature_dim,
        snapshot_sha256=latent_cache.directory_sha256(snapshot),
        upstream_git_sha=args.upstream_git_sha,
    )
    semantics = task_cache.task_feature_semantics(
        catalog,
        encoder=encoder_spec,
        adapter_git_sha=args.adapter_git_sha,
    )
    task_cache.write_task_feature_artifact(output_root, semantics=semantics, features=features)
    loaded = task_cache.load_task_feature_artifact(output_root, prompt_catalog=catalog)
    print(
        json.dumps(
            {
                "cached": False,
                "shape": [len(loaded.by_task_name), feature_dim],
                "semantic_sha256": loaded.manifest["semantic_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main(_parse_args())
