from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from openpi.weaver import b1k_dataset_adapter as episode_adapter
from openpi.weaver import behavior_precomputed
from openpi.weaver import frame_cache
from openpi.weaver import latent_cache
from openpi.weaver import task_cache


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode(index: int, *, length: int = 9, reward: bool = False, sentinel: float = 0.0):
    states = np.zeros((length, 23), dtype=np.float32)
    states[:, 0] = sentinel + np.arange(length, dtype=np.float32)
    states[:, 3] = np.linspace(3.0, 3.4, length, dtype=np.float32)
    actions = episode_adapter.observed_state_delta_actions(states)
    valid = np.ones(length, dtype=bool)
    valid[-1] = False
    return episode_adapter.AdaptedEpisode(
        episode_index=index,
        prompt=f"task {index}",
        target_hz=5.0,
        source_frame_indices=np.arange(length, dtype=np.int64) * 6,
        timestamps_s=np.arange(length, dtype=np.float64) / 5.0,
        timing_errors_s=np.zeros(length, dtype=np.float64),
        states=states,
        state_delta_actions=actions,
        policy_actions_at_frame=np.full((length, 23), index, dtype=np.float32),
        transition_valid=valid,
        rewards=(
            np.concatenate((np.full(length - 1, 0.05, dtype=np.float32), np.zeros(1, dtype=np.float32)))
            if reward
            else None
        ),
    )


def _write_episode(root: Path, episode, *, split: str = "train", latent_shape: tuple[int, int] = (3, 4)) -> dict:
    manifest = episode_adapter.episode_manifest(
        episode,
        split=split,
        dataset_repo_id="b1k-v3again",
        dataset_metadata_sha256="a" * 64,
    )
    source_start = episode.episode_index * 1000
    full_length = int(episode.source_frame_indices[-1]) + 1
    full_policy_actions = np.full((full_length, 23), episode.episode_index, dtype=np.float32)
    manifest["source"] = {
        "dataset_from_index": source_start,
        "dataset_to_index": source_start + full_length,
        "source_fps": 30,
        "full_rate_count": full_length,
        "full_rate_columns": {
            "index": "index",
            "frame_index": "frame_index",
            "timestamp": "source_timestamps_s",
            "policy_action": "source_policy_actions",
            **(
                {
                    "next.reward": "source_rewards",
                    "next.terminated": "source_terminated",
                    "next.truncated": "source_truncated",
                }
                if episode.rewards is not None
                else {}
            ),
        },
        "policy_action_aggregation_5hz": "undefined",
        "raw_state_dim": 61,
    }
    numeric_path = root / "episodes" / split / f"{episode.episode_index:06d}.npz"
    numeric_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "index": np.arange(source_start, source_start + full_length, dtype=np.int64),
        "frame_index": np.arange(full_length, dtype=np.int64),
        "source_timestamps_s": np.arange(full_length, dtype=np.float64) / 30.0,
        "source_policy_actions": full_policy_actions,
        "source_frame_indices": episode.source_frame_indices,
        "source_absolute_indices": episode.source_frame_indices + source_start,
        "timestamps_s": episode.timestamps_s,
        "timing_errors_s": episode.timing_errors_s,
        "states": episode.states,
        "state_delta_actions": episode.state_delta_actions,
        "policy_actions_at_frame": episode.policy_actions_at_frame,
        "transition_valid": episode.transition_valid,
    }
    if episode.rewards is not None:
        source_rewards = np.zeros(full_length, dtype=np.float32)
        source_rewards[episode.source_frame_indices[:-1]] = episode.rewards[:-1]
        source_terminated = np.zeros(full_length, dtype=bool)
        source_terminated[-1] = True
        arrays["source_rewards"] = source_rewards
        arrays["source_terminated"] = source_terminated
        arrays["source_truncated"] = np.zeros(full_length, dtype=bool)
        arrays["rewards"] = episode.rewards
    np.savez_compressed(numeric_path, **arrays)
    manifest["numeric_artifact"] = {
        "complete": True,
        "path": numeric_path.relative_to(root).as_posix(),
        "size_bytes": numeric_path.stat().st_size,
        "sha256": _sha256(numeric_path),
        "keys": sorted(arrays),
    }

    encoder = latent_cache.EncoderSpec(
        model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        revision="0123456789abcdef",
        image_size=(192, 320),
        spatial_size=2,
        compute_dtype="float32",
        storage_dtype="float16",
        vae_snapshot_sha256="c" * 64,
        checkpoint_config_sha256="d" * 64,
    )
    frame_semantics = frame_cache.frame_semantics(
        manifest,
        source_absolute_indices=arrays["source_absolute_indices"],
        view_order=behavior_precomputed.DEFAULT_IMAGE_KEYS,
        decoder=frame_cache.DecoderSpec(
            lerobot_git_sha="a" * 40,
            pyav_version="15.0.0",
        ),
        exporter_git_sha="deadbeef",
    )
    frame_manifest = {
        **frame_semantics,
        "complete": True,
        "artifact": {"sha256": "e" * 64},
    }
    semantics = latent_cache.latent_semantics(
        manifest,
        frame_manifest=frame_manifest,
        encoder=encoder,
        view_order=behavior_precomputed.DEFAULT_IMAGE_KEYS,
        adapter_git_sha="deadbeef",
    )
    latent_path = root / "latents" / split / f"{episode.episode_index:06d}.npz"
    latent_manifest_path = latent_path.with_suffix(".json")
    value = np.arange(episode.states.shape[0], dtype=np.float16)[:, None, None]
    latent_cache.write_latent_artifact(
        latent_path,
        latent_manifest_path,
        semantics=semantics,
        latents={
            "wrist_left_features": np.broadcast_to(
                value + episode.episode_index,
                (len(value), *latent_shape),
            ),
            "exterior_1_left_features": np.broadcast_to(
                value + 100 + episode.episode_index,
                (len(value), *latent_shape),
            ),
        },
    )
    return manifest


def _write_index(root: Path, manifests: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(manifest, sort_keys=True) + "\n" for manifest in manifests),
        encoding="utf-8",
    )


def _write_norm(root: Path) -> None:
    manifest_sha = _sha256(root / "manifest.jsonl")
    (root / "norm_stats.json").write_text(
        json.dumps(
            {
                "schema_version": behavior_precomputed.NORM_SCHEMA_VERSION,
                "dataset": {"repo_id": "b1k-v3again", "metadata_sha256": "a" * 64},
                "source_split": "train",
                "target_hz": 5.0,
                "train_manifest_sha256": manifest_sha,
                "episode_count": 1,
                "norm_stats": {
                    "state": {
                        "representation": "r1pro_compact_state_v1",
                        "mean": [1.0] * 23,
                        "std": [2.0] * 23,
                    },
                    "actions": {
                        "representation": episode_adapter.STATE_DELTA_ACTION_REPRESENTATION,
                        "mean": [0.5] * 23,
                        "std": [0.5] * 23,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_task_features(root: Path, manifests: list[dict]) -> Path:
    prompts = {f"task_{index}": str(manifest["prompt"]) for index, manifest in enumerate(manifests)}
    catalog = {
        "schema_version": task_cache.TASK_PROMPT_SCHEMA_VERSION,
        "dataset_metadata_sha256": "a" * 64,
        "task_catalog_sha256": "b" * 64,
        "prompts": prompts,
    }
    (root / "task_prompts.json").write_text(json.dumps(catalog), encoding="utf-8")
    encoder = task_cache.TaskEncoderSpec(
        model_id="openai/clip-vit-large-patch14",
        revision="c" * 40,
        feature_dim=768,
        snapshot_sha256="d" * 64,
        upstream_git_sha="3e41e907",
    )
    semantics = task_cache.task_feature_semantics(catalog, encoder=encoder, adapter_git_sha="deadbeef")
    feature_root = root / "task_features"
    task_cache.write_task_feature_artifact(
        feature_root,
        semantics=semantics,
        features={name: np.full(768, index + 1, dtype=np.float32) for index, name in enumerate(prompts)},
    )
    return feature_root


def test_norm_stats_provenance_must_match_manifest_dataset_and_rate(tmp_path):
    manifest = _write_episode(tmp_path, _episode(0))
    _write_index(tmp_path, [manifest])
    _write_norm(tmp_path)
    norm_path = tmp_path / "norm_stats.json"
    value = json.loads(norm_path.read_text(encoding="utf-8"))
    value["train_manifest_sha256"] = "f" * 64
    norm_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="different manifest index"):
        behavior_precomputed.BehaviorPrecomputed(
            tmp_path,
            horizon=3,
            n_history=2,
            normalize=True,
            reward_mode="disabled",
        )


def test_dynamics_batch_matches_weaver_contract_without_fake_rewards(tmp_path):
    manifest = _write_episode(tmp_path, _episode(0))
    _write_index(tmp_path, [manifest])
    dataset = behavior_precomputed.BehaviorPrecomputed(
        tmp_path,
        horizon=3,
        n_history=2,
        normalize=False,
        reward_mode="disabled",
    )

    batch = dataset.window_at(0, 1)

    assert set(batch) == {"obs", "actions", "transition_valid", "task"}
    assert set(batch["obs"]) == {"wrist_left_features", "exterior_1_left_features", "states"}
    assert batch["obs"]["states"].shape == (5, 23)
    assert batch["actions"].shape == (5, 23)
    assert batch["obs"]["wrist_left_features"].shape == (5, 3, 4)
    assert batch["task"] == {"text": "task 0"}
    assert "rewards" not in batch
    assert all(np.isfinite(value).all() for value in batch["obs"].values())


def test_explicit_task_feature_cache_completes_released_weaver_batch_abi(tmp_path):
    manifest = _write_episode(tmp_path, _episode(0), latent_shape=(60, 256))
    _write_index(tmp_path, [manifest])
    task_feature_root = _write_task_features(tmp_path, [manifest])
    dataset = behavior_precomputed.BehaviorPrecomputed(
        tmp_path,
        horizon=3,
        n_history=2,
        normalize=False,
        reward_mode="disabled",
        task_feature_root=task_feature_root,
    )

    batch = dataset.window_at(0, 1)

    assert batch["task"]["text"] == "task 0"
    assert batch["task"]["features"].shape == (768,)
    assert batch["task"]["features"].dtype == np.float32
    assert np.isfinite(batch["task"]["features"]).all()
    assert batch["obs"]["wrist_left_features"].shape == (5, 60, 256)
    assert batch["obs"]["exterior_1_left_features"].shape == (5, 60, 256)
    assert batch["obs"]["states"].shape == (5, 23)
    assert batch["actions"].shape == (5, 23)
    assert dataset.task_feature_provenance["semantic_sha256"]


def test_windows_and_sparse_memory_never_cross_episode_boundaries(tmp_path):
    manifests = [
        _write_episode(tmp_path, _episode(0, sentinel=10.0)),
        _write_episode(tmp_path, _episode(1, sentinel=1000.0)),
    ]
    _write_index(tmp_path, manifests)
    dataset = behavior_precomputed.BehaviorPrecomputed(
        tmp_path,
        horizon=3,
        n_history=2,
        n_memory_frames=2,
        t_memory=3,
        normalize=False,
        reward_mode="disabled",
    )

    batch = dataset.window_at(0, 5)

    assert batch["obs"]["states"][:, 0].max() < 1000
    assert batch["memory"]["states"][:, 0].max() < 1000
    np.testing.assert_array_equal(batch["memory"]["states"][:, 0], [10.0, 12.0])


def test_history_and_memory_collapse_are_coupled(tmp_path):
    manifest = _write_episode(tmp_path, _episode(0, sentinel=10.0))
    _write_index(tmp_path, [manifest])
    dataset = behavior_precomputed.BehaviorPrecomputed(
        tmp_path,
        horizon=3,
        n_history=3,
        n_memory_frames=2,
        t_memory=2,
        normalize=False,
        reward_mode="disabled",
    )

    batch = dataset.window_at(0, 4, collapse=True)

    np.testing.assert_array_equal(batch["obs"]["states"][:3, 0], [14.0, 14.0, 14.0])
    np.testing.assert_array_equal(batch["actions"][:2], 0.0)
    np.testing.assert_array_equal(batch["memory"]["states"][:, 0], [14.0, 14.0])
    for key in ("wrist_left_features", "exterior_1_left_features"):
        np.testing.assert_array_equal(batch["obs"][key][0], batch["obs"][key][2])
        np.testing.assert_array_equal(batch["memory"][key][0], batch["memory"][key][1])


def test_real_rewards_are_required_or_explicitly_disabled(tmp_path):
    no_reward = _write_episode(tmp_path, _episode(0, reward=False))
    _write_index(tmp_path, [no_reward])
    with pytest.raises(ValueError, match="no real reward labels"):
        behavior_precomputed.BehaviorPrecomputed(
            tmp_path,
            horizon=3,
            n_history=2,
            normalize=False,
        )

    reward_root = tmp_path / "reward"
    with_reward = _write_episode(reward_root, _episode(0, reward=True))
    _write_index(reward_root, [with_reward])
    dataset = behavior_precomputed.BehaviorPrecomputed(
        reward_root,
        horizon=3,
        n_history=2,
        normalize=False,
        reward_mode="required",
    )
    batch = dataset.window_at(0, 4)
    assert batch["rewards"].shape == (5,)
    assert batch["transition_valid"].all()
    assert batch["rewards"][-1] == pytest.approx(0.05)
    with pytest.raises(ValueError, match=r"\[1, 4\]"):
        dataset.window_at(0, 5)


def test_normalization_uses_named_23d_contract(tmp_path):
    manifest = _write_episode(tmp_path, _episode(0, sentinel=1.0))
    _write_index(tmp_path, [manifest])
    _write_norm(tmp_path)
    dataset = behavior_precomputed.BehaviorPrecomputed(
        tmp_path,
        horizon=3,
        n_history=2,
        reward_mode="disabled",
    )

    batch = dataset.window_at(0, 1)

    np.testing.assert_allclose(
        batch["obs"]["states"][0], (manifest_value := _episode(0, sentinel=1.0).states[0] - 1) / 2
    )
    np.testing.assert_allclose(batch["actions"][0], (_episode(0, sentinel=1.0).state_delta_actions[0] - 0.5) / 0.5)
    assert manifest_value.shape == (23,)


def test_tampered_numeric_bytes_and_semantics_are_rejected(tmp_path):
    manifest = _write_episode(tmp_path, _episode(0))
    _write_index(tmp_path, [manifest])
    numeric_path = tmp_path / manifest["numeric_artifact"]["path"]
    numeric_path.write_bytes(numeric_path.read_bytes() + b"tamper")
    dataset = behavior_precomputed.BehaviorPrecomputed(
        tmp_path,
        horizon=3,
        n_history=2,
        normalize=False,
        reward_mode="disabled",
    )
    with pytest.raises(ValueError, match="bytes do not match"):
        dataset.window_at(0, 1)

    manifest["target_hz"] = 4.0
    _write_index(tmp_path, [manifest])
    with pytest.raises(ValueError, match="semantic SHA-256 mismatch"):
        behavior_precomputed.BehaviorPrecomputed(
            tmp_path,
            horizon=3,
            n_history=2,
            normalize=False,
            reward_mode="disabled",
        )


def test_wrong_latent_view_order_is_rejected(tmp_path):
    manifest = _write_episode(tmp_path, _episode(0))
    _write_index(tmp_path, [manifest])
    dataset = behavior_precomputed.BehaviorPrecomputed(
        tmp_path,
        horizon=3,
        n_history=2,
        image_keys=("exterior_1_left", "wrist_left"),
        normalize=False,
        reward_mode="disabled",
    )
    with pytest.raises(ValueError, match="view order"):
        dataset.window_at(0, 1)


def test_short_episodes_are_excluded_before_sampling(tmp_path):
    manifests = [
        _write_episode(tmp_path, _episode(0, length=4)),
        _write_episode(tmp_path, _episode(1, length=9)),
    ]
    _write_index(tmp_path, manifests)
    dataset = behavior_precomputed.BehaviorPrecomputed(
        tmp_path,
        horizon=3,
        n_history=2,
        normalize=False,
        reward_mode="disabled",
    )
    assert len(dataset) == 1
    assert dataset.manifests[0]["episode_index"] == 1
