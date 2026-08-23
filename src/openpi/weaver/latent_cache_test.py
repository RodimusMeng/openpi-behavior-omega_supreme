import json

import numpy as np
import pytest

from openpi.weaver import b1k_dataset_adapter as episode_adapter
from openpi.weaver import frame_cache
from openpi.weaver import latent_cache


def _numeric_manifest() -> dict:
    timestamps = np.arange(31, dtype=np.float64) / 30.0
    raw_states = np.zeros((31, 61), dtype=np.float32)
    policy_actions = np.zeros((31, 23), dtype=np.float32)
    episode = episode_adapter.adapt_episode(
        episode_index=5,
        prompt="cache task",
        timestamps_s=timestamps,
        raw_states=raw_states,
        policy_actions=policy_actions,
    )
    manifest = episode_adapter.episode_manifest(
        episode,
        split="train",
        dataset_repo_id="b1k-v3again",
        dataset_metadata_sha256="b" * 64,
    )
    manifest["numeric_artifact"] = {
        "complete": True,
        "path": "episodes/train/000005.npz",
        "sha256": "c" * 64,
    }
    return manifest


def _encoder(**changes) -> latent_cache.EncoderSpec:
    values = {
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "revision": "0123456789abcdef",
        "image_size": (192, 320),
        "spatial_size": 2,
        "compute_dtype": "float32",
        "storage_dtype": "float16",
        "vae_snapshot_sha256": "a" * 64,
        "checkpoint_config_sha256": "d" * 64,
    }
    values.update(changes)
    return latent_cache.EncoderSpec(**values)


def _frame_manifest(
    numeric_manifest=None,
    *,
    artifact_sha="e" * 64,
    pyav_version="15.0.0",
    view_order=("wrist_left", "exterior_1_left"),
):
    numeric_manifest = numeric_manifest or _numeric_manifest()
    semantics = frame_cache.frame_semantics(
        numeric_manifest,
        source_absolute_indices=np.arange(int(numeric_manifest["sampled_count"]), dtype=np.int64),
        view_order=view_order,
        decoder=frame_cache.DecoderSpec(
            lerobot_git_sha="a" * 40,
            pyav_version=pyav_version,
        ),
        exporter_git_sha="deadbeef",
    )
    return {
        **semantics,
        "complete": True,
        "artifact": {"sha256": artifact_sha},
    }


def _semantics(numeric_manifest=None, **changes):
    numeric_manifest = numeric_manifest or _numeric_manifest()
    values = {
        "numeric_episode_manifest": numeric_manifest,
        "frame_manifest": _frame_manifest(numeric_manifest),
        "encoder": _encoder(),
        "view_order": ("wrist_left", "exterior_1_left"),
        "adapter_git_sha": "deadbeef",
    }
    values.update(changes)
    return latent_cache.latent_semantics(**values)


def _latents(length: int = 6) -> dict[str, np.ndarray]:
    return {
        "wrist_left_features": np.full((length, 2, 4), 1.0, dtype=np.float16),
        "exterior_1_left_features": np.full((length, 2, 4), 2.0, dtype=np.float16),
    }


def _change_numeric_manifest(field, value):
    manifest = _numeric_manifest()
    manifest[field] = value
    semantic = {key: manifest[key] for key in episode_adapter.EPISODE_SEMANTIC_FIELDS}
    manifest["semantic_sha256"] = episode_adapter.semantic_sha256(semantic)
    return manifest


def test_verified_cache_hit_does_not_call_encoder_twice(tmp_path):
    npz_path = tmp_path / "episode.npz"
    manifest_path = tmp_path / "episode.json"
    semantics = _semantics()
    calls = 0

    def encode():
        nonlocal calls
        calls += 1
        return _latents()

    first, first_was_cached = latent_cache.load_or_encode(
        npz_path,
        manifest_path,
        semantics=semantics,
        encode=encode,
    )
    second, second_was_cached = latent_cache.load_or_encode(
        npz_path,
        manifest_path,
        semantics=semantics,
        encode=encode,
    )

    assert calls == 1
    assert not first_was_cached
    assert second_was_cached
    assert latent_cache.cache_status(npz_path, manifest_path, expected_semantic_sha256=semantics["semantic_sha256"]).hit
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
        assert not second[key].flags.writeable


@pytest.mark.parametrize(
    "changed",
    [
        lambda: _semantics(numeric_manifest=_change_numeric_manifest("target_hz", 4.0)),
        lambda: _semantics(numeric_manifest=_change_numeric_manifest("split", "val")),
        lambda: _semantics(numeric_manifest=_change_numeric_manifest("timestamps_sha256", "c" * 64)),
        lambda: _semantics(encoder=_encoder(revision="fedcba9876543210")),
        lambda: _semantics(encoder=_encoder(image_size=(256, 256))),
        lambda: _semantics(encoder=_encoder(resize_mode="center-crop")),
        lambda: _semantics(encoder=_encoder(checkpoint_config_sha256="e" * 64)),
        lambda: _semantics(encoder=_encoder(spatial_size=4)),
        lambda: _semantics(encoder=_encoder(storage_dtype="float32")),
        lambda: _semantics(encoder=_encoder(vae_snapshot_sha256="f" * 64)),
        lambda: _semantics(frame_manifest=_frame_manifest(artifact_sha="9" * 64)),
        lambda: _semantics(frame_manifest=_frame_manifest(pyav_version="16.0.0")),
        lambda: _semantics(
            view_order=("exterior_1_left", "wrist_left"),
            frame_manifest=_frame_manifest(view_order=("exterior_1_left", "wrist_left")),
        ),
        lambda: _semantics(adapter_git_sha="cafebabe"),
    ],
)
def test_every_semantic_input_invalidates_the_cache(changed):
    assert changed()["semantic_sha256"] != _semantics()["semantic_sha256"]


def test_only_npz_without_complete_manifest_is_not_a_cache_hit(tmp_path):
    npz_path = tmp_path / "orphan.npz"
    manifest_path = tmp_path / "orphan.json"
    np.savez_compressed(npz_path, **_latents())

    status = latent_cache.cache_status(npz_path, manifest_path, expected_semantic_sha256="0" * 64)

    assert not status.hit
    assert "both exist" in status.reason


def test_corrupt_or_modified_npz_is_not_a_cache_hit(tmp_path):
    npz_path = tmp_path / "episode.npz"
    manifest_path = tmp_path / "episode.json"
    semantics = _semantics()
    latent_cache.write_latent_artifact(
        npz_path,
        manifest_path,
        semantics=semantics,
        latents=_latents(),
    )
    with npz_path.open("ab") as stream:
        stream.write(b"modified")

    status = latent_cache.cache_status(
        npz_path,
        manifest_path,
        expected_semantic_sha256=semantics["semantic_sha256"],
    )

    assert not status.hit
    assert "SHA-256" in status.reason


def test_wrong_latent_keys_or_time_shape_are_rejected_before_write(tmp_path):
    semantics = _semantics()
    with pytest.raises(ValueError, match="exactly"):
        latent_cache.write_latent_artifact(
            tmp_path / "wrong-keys.npz",
            tmp_path / "wrong-keys.json",
            semantics=semantics,
            latents={"wrist_left_features": _latents()["wrist_left_features"]},
        )
    with pytest.raises(ValueError, match="sampled time dimension"):
        latent_cache.write_latent_artifact(
            tmp_path / "wrong-time.npz",
            tmp_path / "wrong-time.json",
            semantics=semantics,
            latents=_latents(length=5),
        )


def test_stale_cache_is_not_silently_overwritten(tmp_path):
    npz_path = tmp_path / "episode.npz"
    manifest_path = tmp_path / "episode.json"
    original = _semantics()
    latent_cache.write_latent_artifact(
        npz_path,
        manifest_path,
        semantics=original,
        latents=_latents(),
    )
    changed = _semantics(encoder=_encoder(revision="new-revision"))

    with pytest.raises(latent_cache.CacheMissError, match="move it aside"):
        latent_cache.load_or_encode(
            npz_path,
            manifest_path,
            semantics=changed,
            encode=_latents,
        )


@pytest.mark.parametrize(
    "changed_frame",
    [
        lambda: _frame_manifest(artifact_sha="9" * 64),
        lambda: _frame_manifest(pyav_version="16.0.0"),
    ],
)
def test_frame_bytes_or_decoder_change_makes_existing_latent_a_miss(tmp_path, changed_frame):
    npz_path = tmp_path / "episode.npz"
    manifest_path = tmp_path / "episode.json"
    original = _semantics()
    latent_cache.write_latent_artifact(
        npz_path,
        manifest_path,
        semantics=original,
        latents=_latents(),
    )
    changed = _semantics(frame_manifest=changed_frame())

    status = latent_cache.cache_status(
        npz_path,
        manifest_path,
        expected_semantic_sha256=changed["semantic_sha256"],
    )

    assert not status.hit
    assert "fingerprint changed" in status.reason


def test_manifest_records_named_arrays_without_pickle(tmp_path):
    npz_path = tmp_path / "episode.npz"
    manifest_path = tmp_path / "episode.json"
    semantics = _semantics()
    written = latent_cache.write_latent_artifact(
        npz_path,
        manifest_path,
        semantics=semantics,
        latents=_latents(),
    )

    on_disk = json.loads(manifest_path.read_text())
    assert written == on_disk
    assert on_disk["view_order"] == ["wrist_left", "exterior_1_left"]
    assert set(on_disk["artifact"]["arrays"]) == {
        "wrist_left_features",
        "exterior_1_left_features",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        assert archive.files == ["wrist_left_features", "exterior_1_left_features"]


def test_directory_sha256_covers_relative_names_and_exact_bytes(tmp_path):
    snapshot = tmp_path / "vae"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"config-v1")
    weights = snapshot / "diffusion_pytorch_model.safetensors"
    weights.write_bytes(b"weights-v1")
    original = latent_cache.directory_sha256(snapshot)

    weights.write_bytes(b"weights-v2")
    changed_bytes = latent_cache.directory_sha256(snapshot)
    assert changed_bytes != original

    weights.rename(snapshot / "renamed.safetensors")
    assert latent_cache.directory_sha256(snapshot) != changed_bytes


def test_encoder_spec_rejects_non_float32_compute():
    with pytest.raises(ValueError, match="float32 compute"):
        _encoder(compute_dtype="float16")
