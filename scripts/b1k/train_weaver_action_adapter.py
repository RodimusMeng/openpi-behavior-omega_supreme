"""Train the causal B1K policy-command -> WEAVER-action adapter.

This is deliberately separate from WEAVER fine-tuning.  It consumes the
prepared full-rate command/timestamp provenance, computes normalization from
the train split only, and writes a small resumable PyTorch checkpoint.  It does
not use simulator state, rewards, or held-out labels as online inputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from openpi.weaver import causal_action_adapter as adapter


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, torch: Any | None = None) -> dict[str, Any]:
    torch = torch or _require_torch()
    _validate_args(args)
    _seed(args.seed, torch=torch)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = output_dir / "checkpoint.pt"
    train_rng = random.Random(args.seed)
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requires {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != adapter.ADAPTER_SCHEMA:
            raise ValueError("Resume checkpoint has the wrong causal-adapter schema")
        expected_semantics = _dataset_semantics(
            dataset_dir,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        checkpoint_dataset = checkpoint.get("dataset_semantics")
        if checkpoint_dataset != expected_semantics:
            raise ValueError(
                f"Resume checkpoint dataset/training semantics differ: checkpoint={checkpoint_dataset!r}, current={expected_semantics!r}"
            )
        spec = adapter.AdapterSpec(**checkpoint["spec"])
        normalization = adapter.AdapterNormalization.from_dict(checkpoint["normalization"])
        model = adapter.build_torch_model(spec).to(args.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        _restore_rng_state(checkpoint.get("rng_state"), train_rng=train_rng, torch=torch)
    else:
        if checkpoint_path.exists():
            raise FileExistsError(f"Refusing to overwrite {checkpoint_path} without --resume")
        train_samples = tuple(
            adapter.iter_adapter_samples(dataset_dir, split="train", max_samples=args.max_train_samples)
        )
        normalization = adapter.compute_normalization(train_samples)
        spec = adapter.AdapterSpec(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        model = adapter.build_torch_model(spec).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        start_step = 0
    if args.resume:
        train_samples = tuple(adapter.iter_adapter_samples(dataset_dir, split="train", max_samples=args.max_train_samples))
    val_samples = tuple(adapter.iter_adapter_samples(dataset_dir, split="val", max_samples=args.max_val_samples))
    if not train_samples or not val_samples:
        raise ValueError("Adapter training requires non-empty train and val samples")

    model.train()
    reports = []
    for step in range(start_step, args.max_steps):
        selected = [train_samples[train_rng.randrange(len(train_samples))] for _ in range(args.batch_size)]
        batch = _torch_batch(adapter.collate_samples(selected, normalization), torch=torch, device=args.device)
        optimizer.zero_grad(set_to_none=True)
        loss, log = adapter.adapter_loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite adapter loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        report = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "normalized_mae": float(log["normalized_mae"].cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        reports.append(report)
        completed = step + 1
        if args.checkpoint_every and completed % args.checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, model, optimizer, spec, normalization, completed, dataset_dir, rng_state=_capture_rng_state(train_rng=train_rng, torch=torch), dataset_semantics=_dataset_semantics(dataset_dir, max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples, batch_size=args.batch_size, seed=args.seed, learning_rate=args.learning_rate, weight_decay=args.weight_decay))

    validation = evaluate(model, val_samples, normalization, batch_size=args.val_batch_size, torch=torch, device=args.device)
    completed_step = reports[-1]["step"] if reports else start_step
    _save_checkpoint(checkpoint_path, model, optimizer, spec, normalization, completed_step, dataset_dir, rng_state=_capture_rng_state(train_rng=train_rng, torch=torch), dataset_semantics=_dataset_semantics(dataset_dir, max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples, batch_size=args.batch_size, seed=args.seed, learning_rate=args.learning_rate, weight_decay=args.weight_decay))
    summary = {
        "schema_version": adapter.ADAPTER_SCHEMA,
        "step": completed_step,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "spec": dataclasses.asdict(spec),
        "validation": validation,
        "last_train": reports[-1] if reports else None,
        "device": str(args.device),
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def evaluate(model: Any, samples: Sequence[adapter.AdapterSample], normalization: adapter.AdapterNormalization, *, batch_size: int, torch: Any, device: str) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            selected = samples[start : start + batch_size]
            batch = _torch_batch(adapter.collate_samples(selected, normalization), torch=torch, device=device)
            loss, log = adapter.adapter_loss(model, batch)
            count = len(selected)
            loss_sum += float(loss.cpu()) * count
            mae_sum += float(log["normalized_mae"].cpu()) * count
            sample_count += count
    return {"loss": loss_sum / sample_count, "normalized_mae": mae_sum / sample_count}


def _save_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    spec: adapter.AdapterSpec,
    normalization: adapter.AdapterNormalization,
    step: int,
    dataset_dir: Path,
    *,
    rng_state: dict[str, Any],
    dataset_semantics: dict[str, Any],
) -> None:
    payload = adapter.checkpoint_payload(
        model,
        optimizer,
        spec=spec,
        normalization=normalization,
        step=step,
        dataset_semantics=dataset_semantics,
        rng_state=rng_state,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        _require_torch().save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_batch(batch: dict[str, np.ndarray], *, torch: Any, device: str) -> dict[str, Any]:
    return {
        key: torch.as_tensor(value, dtype=torch.bool if key == "mask" else torch.float32, device=device)
        for key, value in batch.items()
    }


def _dataset_semantics(
    dataset_dir: Path,
    *,
    max_train_samples: int | None,
    max_val_samples: int | None,
    batch_size: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing adapter manifest index: {manifest_path}")
    selected: dict[str, Any] = {}
    for split, limit in (("train", max_train_samples), ("val", max_val_samples)):
        manifests = adapter.read_manifest_index(dataset_dir, split=split)
        semantic = [str(item.get("semantic_sha256", "")) for item in manifests]
        if any(len(value) != 64 for value in semantic):
            raise ValueError(f"Every {split} manifest must expose a semantic SHA-256")
        selected[split] = semantic
        selected[f"{split}_limit"] = limit
    return {
        "dataset_dir": str(dataset_dir),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_semantic_sha256": selected,
        "split_for_stats": "train",
        "target": "observed_5hz_state_delta",
        "batch_size": int(batch_size),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
    }


def _capture_rng_state(*, train_rng: random.Random, torch: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "train_rng": train_rng.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(value: Any, *, train_rng: random.Random, torch: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("Resume checkpoint has no RNG state; refusing non-reproducible continuation")
    try:
        random.setstate(value["python"])
        np.random.set_state(value["numpy"])
        train_rng.setstate(value["train_rng"])
        torch.set_rng_state(value["torch"])
        if torch.cuda.is_available() and "torch_cuda" in value:
            torch.cuda.set_rng_state_all(value["torch_cuda"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Resume checkpoint contains an invalid RNG state") from exc


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "max_steps", "val_batch_size"):
        if isinstance(getattr(args, name), bool) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("max_train_samples", "max_val_samples"):
        value = getattr(args, name)
        if value is not None and (isinstance(value, bool) or value <= 0):
            raise ValueError(f"{name} must be positive when supplied")
    if isinstance(args.checkpoint_every, bool) or args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")


def _seed(seed: int, *, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Action-adapter training requires PyTorch") from exc
    return torch


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
