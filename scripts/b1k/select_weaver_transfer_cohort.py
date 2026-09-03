"""Select a deterministic, task-matched 40/10 WEAVER transfer cohort.

The default design chooses ten tasks, four train episodes per task, and one
validation episode for the same task.  Selection is episode-disjoint and the
result records immutable numeric semantic hashes for every episode.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

from openpi.weaver import b1k_dataset_adapter
from openpi.weaver import causal_action_adapter

SCHEMA_VERSION = "b1k_weaver_transfer_cohort_v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--train-per-task", type=int, default=4)
    parser.add_argument("--held-out-per-task", type=int, default=1)
    parser.add_argument("--held-out-split", choices=("val", "test"), default="val")
    parser.add_argument("--exclude-tasks", nargs="*", default=["turn_on_radio"])
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args(argv)


def _task_name(manifest: Mapping[str, Any]) -> str | None:
    names = manifest.get("task_names")
    if not isinstance(names, Sequence) or isinstance(names, str | bytes) or len(names) != 1:
        return None
    value = str(names[0]).strip()
    return value or None


def _group_by_task(manifests: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        b1k_dataset_adapter.verify_episode_manifest(manifest)
        task = _task_name(manifest)
        if task is not None:
            grouped[task].append(manifest)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["episode_index"]))
    return dict(grouped)


def select_cohort(
    train_manifests: Sequence[Mapping[str, Any]],
    held_out_manifests: Sequence[Mapping[str, Any]],
    *,
    num_tasks: int,
    train_per_task: int,
    held_out_per_task: int,
    exclude_tasks: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    counts = (num_tasks, train_per_task, held_out_per_task)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts):
        raise ValueError("num_tasks and per-task episode counts must be positive integers")
    train_by_task = _group_by_task(train_manifests)
    held_by_task = _group_by_task(held_out_manifests)
    excluded = {str(value).strip() for value in exclude_tasks}
    eligible = sorted(
        task
        for task in set(train_by_task) & set(held_by_task)
        if task not in excluded
        and len(train_by_task[task]) >= train_per_task
        and len(held_by_task[task]) >= held_out_per_task
    )
    if len(eligible) < num_tasks:
        raise ValueError(
            f"Need {num_tasks} eligible task-matched cohorts, found {len(eligible)}; "
            f"requirements=train:{train_per_task}, held_out:{held_out_per_task}"
        )
    rng = random.Random(seed)
    tasks = sorted(rng.sample(eligible, num_tasks))
    records = []
    for task in tasks:
        train = sorted(rng.sample(train_by_task[task], train_per_task), key=lambda item: int(item["episode_index"]))
        held = sorted(rng.sample(held_by_task[task], held_out_per_task), key=lambda item: int(item["episode_index"]))
        records.append(
            {
                "task": task,
                "train": [_episode_identity(item) for item in train],
                "held_out": [_episode_identity(item) for item in held],
            }
        )
    train_ids = [item["episode_index"] for record in records for item in record["train"]]
    held_ids = [item["episode_index"] for record in records for item in record["held_out"]]
    if set(train_ids) & set(held_ids):
        raise ValueError("Selected train and held-out episode IDs overlap")
    return {
        "tasks": tasks,
        "records": records,
        "train_episode_indices": train_ids,
        "held_out_episode_indices": held_ids,
    }


def _episode_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    semantic = str(manifest.get("semantic_sha256", ""))
    if len(semantic) != 64:
        raise ValueError(f"Episode {manifest.get('episode_index')} has no semantic SHA-256")
    return {"episode_index": int(manifest["episode_index"]), "semantic_sha256": semantic}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cohort selection: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    root = Path(args.dataset_dir).expanduser().resolve()
    selection = select_cohort(
        causal_action_adapter.read_manifest_index(root, split="train"),
        causal_action_adapter.read_manifest_index(root, split=args.held_out_split),
        num_tasks=args.num_tasks,
        train_per_task=args.train_per_task,
        held_out_per_task=args.held_out_per_task,
        exclude_tasks=args.exclude_tasks,
        seed=args.seed,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(root),
        "manifest_jsonl_sha256": _sha256(root / "manifest.jsonl"),
        "seed": args.seed,
        "train_split": "train",
        "held_out_split": args.held_out_split,
        "num_tasks": args.num_tasks,
        "train_per_task": args.train_per_task,
        "held_out_per_task": args.held_out_per_task,
        "excluded_tasks": sorted(set(args.exclude_tasks)),
        **selection,
    }
    output = Path(args.output_json).expanduser().resolve()
    _atomic_write(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"COHORT PASSED: {output}", flush=True)


if __name__ == "__main__":
    main()
