"""Benchmark the complete B1K loader through transforms, collate, and JAX sharding."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
import statistics
import time
from typing import Literal

import jax
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi05_b1k_full_abs"
    batches: int = 64
    warmup_batches: int = 8
    num_workers: int | None = None
    prefetch_factor: int | None = None
    block_size: int | None = None
    loader_microbatch_size: int | None = None
    in_order: bool = True
    video_backend: Literal["pyav", "torchcodec"] | None = None
    torchcodec_decoder_cache_size: int | None = None
    tolerance_s: float | None = None
    output_json: str | None = None


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _apply_overrides(args: Args) -> _config.TrainConfig:
    config = _config.get_config(args.config_name)
    if not isinstance(config.data, _config.LeRobotB1KDataConfig):
        raise TypeError(f"{args.config_name!r} is not a LeRobotB1KDataConfig")

    dataset_kwargs = dict(config.data.base_config.dataset_kwargs)
    if args.video_backend is not None:
        dataset_kwargs["video_backend"] = args.video_backend
    if args.tolerance_s is not None:
        dataset_kwargs["tolerance_s"] = args.tolerance_s
    data = dataclasses.replace(
        config.data,
        base_config=dataclasses.replace(config.data.base_config, dataset_kwargs=dataset_kwargs),
    )
    replacements = {
        "data": data,
        "b1k_block_size": args.block_size,
        "loader_microbatch_size": args.loader_microbatch_size,
        "loader_in_order": args.in_order,
    }
    if args.num_workers is not None:
        replacements["num_workers"] = args.num_workers
    if args.prefetch_factor is not None:
        replacements["prefetch_factor"] = args.prefetch_factor
    if args.torchcodec_decoder_cache_size is not None:
        replacements["torchcodec_decoder_cache_size"] = args.torchcodec_decoder_cache_size
    return dataclasses.replace(config, **replacements)


def main(args: Args) -> None:
    if args.batches <= 0:
        raise ValueError("batches must be positive")
    if not 0 <= args.warmup_batches < args.batches:
        raise ValueError("warmup_batches must be in [0, batches)")

    config = _apply_overrides(args)
    started = time.perf_counter()
    loader = _data_loader.create_data_loader(config, shuffle=True, num_batches=args.batches)
    loader_created_s = time.perf_counter() - started

    latencies = []
    iterator = iter(loader)
    for batch_index in range(args.batches):
        batch_started = time.perf_counter()
        batch = next(iterator)
        jax.block_until_ready(batch)
        latency = time.perf_counter() - batch_started
        latencies.append(latency)
        print(f"batch={batch_index + 1}/{args.batches} latency_s={latency:.4f}", flush=True)

    steady = latencies[args.warmup_batches :]
    steady_elapsed = sum(steady)
    summary = {
        "config_name": args.config_name,
        "batch_size": config.batch_size,
        "batches": args.batches,
        "warmup_batches": args.warmup_batches,
        "loader_created_s": loader_created_s,
        "first_batch_s": latencies[0],
        "steady_samples_per_s": config.batch_size * len(steady) / steady_elapsed,
        "steady_latency_mean_s": statistics.fmean(steady),
        "steady_latency_p50_s": _percentile(steady, 0.50),
        "steady_latency_p95_s": _percentile(steady, 0.95),
        "steady_latency_p99_s": _percentile(steady, 0.99),
        "steady_latency_max_s": max(steady),
        "num_workers": config.num_workers,
        "prefetch_factor": config.prefetch_factor,
        "block_size": config.b1k_block_size,
        "loader_microbatch_size": config.loader_microbatch_size,
        "in_order": config.loader_in_order,
        "torchcodec_decoder_cache_size": config.torchcodec_decoder_cache_size,
        "video_backend": config.data.base_config.dataset_kwargs.get("video_backend"),
        "tolerance_s": config.data.base_config.dataset_kwargs.get("tolerance_s"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(tyro.cli(Args))
