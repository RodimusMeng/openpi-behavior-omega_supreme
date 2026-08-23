"""Load one deterministic BehaviorPrecomputed window and audit its batch contract."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
from typing import Any

import numpy as np
import tyro

from openpi.weaver.behavior_precomputed import BehaviorPrecomputed


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: str
    split: str = "train"
    position: int = 0
    horizon: int = 8
    n_history: int = 2
    n_memory_frames: int = 0
    t_memory: int = 1
    normalize: bool = False


def _describe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _describe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        result = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else None,
        }
        if np.issubdtype(value.dtype, np.number) and value.size:
            result.update({"min": float(value.min()), "max": float(value.max())})
        return result
    return value


def main(args: Args) -> None:
    dataset = BehaviorPrecomputed(
        args.dataset_dir,
        split=args.split,
        horizon=args.horizon,
        n_history=args.n_history,
        n_memory_frames=args.n_memory_frames,
        t_memory=args.t_memory,
        normalize=args.normalize,
        reward_mode="disabled",
        collapse_prob=0.0,
    )
    if not 0 <= args.position < len(dataset):
        raise ValueError(f"position must be in [0,{len(dataset) - 1}], got {args.position}")
    manifest = dataset.manifests[args.position]
    min_t = args.n_history - 1
    max_t = int(manifest["sampled_count"]) - args.horizon - 1
    t = (min_t + max_t) // 2
    batch = dataset.window_at(args.position, t, collapse=False)
    report = {
        "dataset_episodes": len(dataset),
        "episode_index": manifest["episode_index"],
        "split": manifest["split"],
        "prompt": manifest["prompt"],
        "target_hz": manifest["target_hz"],
        "sampled_count": manifest["sampled_count"],
        "window_t": t,
        "window": _describe(batch),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
