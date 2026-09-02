"""Compute train-only WEAVER normalization statistics from numeric artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

from openpi.weaver import b1k_dataset_adapter as _episode_adapter
from openpi.weaver import behavior_precomputed
from openpi.weaver import hybrid_action
from openpi.weaver import norm_stats


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: str = "outputs/weaver/b1k-v3again"
    output_path: str | None = None
    max_episodes: int | None = None
    min_std: float = 1e-6
    clamp_small_std: bool = False
    hybrid_action_contract: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite normalization statistics: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(args: Args) -> None:
    if args.max_episodes is not None:
        raise ValueError(
            "--max-episodes is intentionally unsupported for production normalization; "
            "statistics must cover the complete train split"
        )
    root = Path(args.dataset_dir).expanduser().resolve()
    index_path = root / "manifest.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing numeric episode manifest: {index_path}")
    manifests = []
    with index_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                manifest = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {index_path}:{line_number}: {exc}") from exc
            _episode_adapter.verify_episode_manifest(manifest)
            if manifest["split"] == "train":
                manifests.append(manifest)
    manifests.sort(key=lambda value: int(value["episode_index"]))
    if not manifests:
        raise ValueError("No train manifests found")

    repo_ids = {manifest["dataset"]["repo_id"] for manifest in manifests}
    metadata_hashes = {manifest["dataset"]["metadata_sha256"] for manifest in manifests}
    target_rates = {float(manifest["target_hz"]) for manifest in manifests}
    if len(repo_ids) != 1 or len(metadata_hashes) != 1 or len(target_rates) != 1:
        raise ValueError("Train manifests mix dataset identities or target rates")

    def episodes():
        for ordinal, manifest in enumerate(manifests, start=1):
            episode = behavior_precomputed.load_numeric_episode(root, manifest)
            print(
                f"[{ordinal}/{len(manifests)}] episode={manifest['episode_index']} "
                f"states={episode.length} valid_actions={int(episode.transition_valid.sum())}",
                flush=True,
            )
            yield episode

    hybrid_contract = (
        hybrid_action.load_audited_contract(args.hybrid_action_contract)
        if args.hybrid_action_contract is not None
        else None
    )
    result = norm_stats.build_norm_stats(
        episodes(),
        dataset_repo_id=repo_ids.pop(),
        dataset_metadata_sha256=metadata_hashes.pop(),
        target_hz=target_rates.pop(),
        train_manifest_sha256=_sha256(index_path),
        min_std=args.min_std,
        clamp_small_std=args.clamp_small_std,
        hybrid_contract=hybrid_contract,
    )
    output_path = (
        Path(args.output_path).expanduser().resolve() if args.output_path is not None else root / "norm_stats.json"
    )
    _atomic_write_json(output_path, result)
    print(json.dumps({"output_path": str(output_path), **result}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
