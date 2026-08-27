from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openpi.weaver import causal_action_adapter as adapter  # noqa: E402
from scripts.b1k import train_weaver_action_adapter as train_adapter  # noqa: E402


def _sample(split: str, index: int) -> adapter.AdapterSample:
    rng = np.random.default_rng(index)
    return adapter.AdapterSample(
        current_state=rng.normal(size=23).astype(np.float32),
        policy_actions=rng.normal(size=(4 + index % 3, 23)).astype(np.float32),
        action_durations_s=np.full(4 + index % 3, 0.2 / (4 + index % 3), dtype=np.float32),
        target_state_delta=rng.normal(scale=0.1, size=23).astype(np.float32),
        split=split,
        episode_index=index,
        interval_index=0,
    )


def _args(dataset: Path, output: Path, *, resume: bool, max_steps: int) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=str(dataset),
        output_dir=str(output),
        device="cpu",
        batch_size=2,
        max_steps=max_steps,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        val_batch_size=2,
        max_train_samples=None,
        max_val_samples=None,
        checkpoint_every=1,
        resume=resume,
        seed=5,
    )


def _manifest(dataset: Path) -> None:
    (dataset / "manifest.jsonl").write_text(
        "\n".join(
            [
                '{"split":"train","episode_index":0,"semantic_sha256":"' + "a" * 64 + '"}',
                '{"split":"val","episode_index":1,"semantic_sha256":"' + "b" * 64 + '"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_real_train_checkpoint_and_resume(tmp_path, monkeypatch):
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    _manifest(dataset)
    output = tmp_path / "adapter"
    train_samples = tuple(_sample("train", index) for index in range(4))
    val_samples = tuple(_sample("val", index + 10) for index in range(2))

    def samples(_root, *, split, max_samples=None):
        values = train_samples if split == "train" else val_samples
        return iter(values if max_samples is None else values[:max_samples])

    monkeypatch.setattr(adapter, "iter_adapter_samples", samples)
    first = train_adapter.run(_args(dataset, output, resume=False, max_steps=2), torch=torch)
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)

    assert first["step"] == 2
    assert checkpoint["schema_version"] == adapter.ADAPTER_SCHEMA
    assert checkpoint["step"] == 2
    assert checkpoint["dataset_semantics"]["split_for_stats"] == "train"
    assert first["validation"]["loss"] >= 0

    resumed = train_adapter.run(_args(dataset, output, resume=True, max_steps=3), torch=torch)
    assert resumed["step"] == 3


def test_resume_rejects_different_dataset(tmp_path, monkeypatch):
    first_dataset = tmp_path / "prepared-a"
    second_dataset = tmp_path / "prepared-b"
    first_dataset.mkdir()
    second_dataset.mkdir()
    _manifest(first_dataset)
    _manifest(second_dataset)
    output = tmp_path / "adapter"
    train_samples = tuple(_sample("train", index) for index in range(2))
    val_samples = tuple(_sample("val", index + 10) for index in range(2))
    monkeypatch.setattr(
        adapter,
        "iter_adapter_samples",
        lambda _root, *, split, max_samples=None: iter(train_samples if split == "train" else val_samples),
    )
    train_adapter.run(_args(first_dataset, output, resume=False, max_steps=1), torch=torch)

    with pytest.raises(ValueError, match="semantics differ"):
        train_adapter.run(_args(second_dataset, output, resume=True, max_steps=2), torch=torch)


def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    _manifest(dataset)
    train_samples = tuple(_sample("train", index) for index in range(4))
    val_samples = tuple(_sample("val", index + 10) for index in range(2))
    monkeypatch.setattr(
        adapter,
        "iter_adapter_samples",
        lambda _root, *, split, max_samples=None: iter(train_samples if split == "train" else val_samples),
    )

    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    train_adapter.run(_args(dataset, uninterrupted, resume=False, max_steps=3), torch=torch)
    train_adapter.run(_args(dataset, resumed, resume=False, max_steps=2), torch=torch)
    train_adapter.run(_args(dataset, resumed, resume=True, max_steps=3), torch=torch)
    left = torch.load(uninterrupted / "checkpoint.pt", map_location="cpu", weights_only=False)
    right = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    for key, value in left["model"].items():
        torch.testing.assert_close(value, right["model"][key], rtol=0, atol=0)
