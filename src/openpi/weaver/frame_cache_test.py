import json

import numpy as np
import pytest

from openpi.weaver import b1k_dataset_adapter as episode_adapter
from openpi.weaver import frame_cache

VIEW_ORDER = ("wrist_left", "exterior_1_left")
DECODER = frame_cache.DecoderSpec(
    lerobot_git_sha="a" * 40,
    pyav_version="15.0.0",
)


def _numeric_manifest_and_indices() -> tuple[dict, np.ndarray]:
    timestamps = np.arange(31, dtype=np.float64) / 30.0
    episode = episode_adapter.adapt_episode(
        episode_index=5,
        prompt="frame cache task",
        timestamps_s=timestamps,
        raw_states=np.zeros((31, 61), dtype=np.float32),
        policy_actions=np.zeros((31, 23), dtype=np.float32),
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
        "size_bytes": 123,
        "sha256": "c" * 64,
    }
    indices = episode.source_frame_indices + 100
    return manifest, indices


def _frames(length: int, *, dtype=np.uint8) -> dict[str, np.ndarray]:
    return {
        "wrist_left": np.full((length, 4, 6, 3), 17, dtype=dtype),
        "exterior_1_left": np.full((length, 5, 7, 3), 29, dtype=dtype),
    }


def _write(root, manifest, indices):
    semantics = frame_cache.frame_semantics(
        manifest,
        source_absolute_indices=indices,
        view_order=VIEW_ORDER,
        decoder=DECODER,
        exporter_git_sha="deadbeef",
    )
    npz_path = root / "train" / "000005.npz"
    manifest_path = npz_path.with_suffix(".json")
    written = frame_cache.write_frame_artifact(
        npz_path,
        manifest_path,
        semantics=semantics,
        source_absolute_indices=indices,
        frames=_frames(len(indices)),
    )
    return npz_path, manifest_path, written


def test_round_trip_proves_numeric_linkage_and_preserves_named_uint8_frames(tmp_path):
    manifest, indices = _numeric_manifest_and_indices()
    npz_path, manifest_path, written = _write(tmp_path, manifest, indices)

    loaded = frame_cache.load_episode_frames(
        tmp_path,
        manifest,
        source_absolute_indices=indices,
        view_order=VIEW_ORDER,
        expected_exporter_git_sha="deadbeef",
        expected_decoder=DECODER,
    )

    np.testing.assert_array_equal(loaded.source_absolute_indices, indices)
    assert not loaded.source_absolute_indices.flags.writeable
    assert tuple(loaded.frames) == VIEW_ORDER
    for view, value in loaded.frames.items():
        assert value.dtype == np.uint8
        assert value.shape[0] == len(indices)
        assert value.shape[-1] == 3
        assert not value.flags.writeable
        np.testing.assert_array_equal(value, _frames(len(indices))[view])
    assert written == json.loads(manifest_path.read_text())
    with np.load(npz_path, allow_pickle=False) as archive:
        assert archive.files == [
            "source_absolute_indices",
            "wrist_left_rgb",
            "exterior_1_left_rgb",
        ]


def test_numeric_artifact_or_absolute_index_change_invalidates_linkage(tmp_path):
    manifest, indices = _numeric_manifest_and_indices()
    _write(tmp_path, manifest, indices)

    changed_manifest = dict(manifest)
    changed_manifest["numeric_artifact"] = dict(manifest["numeric_artifact"], sha256="d" * 64)
    with pytest.raises(frame_cache.FrameCacheError, match="numeric_artifact_sha256"):
        frame_cache.load_episode_frames(
            tmp_path,
            changed_manifest,
            source_absolute_indices=indices,
            view_order=VIEW_ORDER,
        )

    changed_indices = indices.copy()
    changed_indices[-1] += 1
    with pytest.raises(frame_cache.FrameCacheError, match="source_absolute_indices_sha256"):
        frame_cache.load_episode_frames(
            tmp_path,
            manifest,
            source_absolute_indices=changed_indices,
            view_order=VIEW_ORDER,
        )


def test_modified_npz_is_rejected_before_decode(tmp_path):
    manifest, indices = _numeric_manifest_and_indices()
    npz_path, manifest_path, written = _write(tmp_path, manifest, indices)
    with npz_path.open("ab") as stream:
        stream.write(b"modified")

    with pytest.raises(frame_cache.FrameCacheError, match="SHA-256"):
        frame_cache.load_frame_artifact(
            npz_path,
            manifest_path,
            expected_semantic_sha256=written["semantic_sha256"],
        )


def test_writer_rejects_non_uint8_or_wrong_time_shape(tmp_path):
    manifest, indices = _numeric_manifest_and_indices()
    semantics = frame_cache.frame_semantics(
        manifest,
        source_absolute_indices=indices,
        view_order=VIEW_ORDER,
        decoder=DECODER,
        exporter_git_sha="deadbeef",
    )
    with pytest.raises(ValueError, match="uint8"):
        frame_cache.write_frame_artifact(
            tmp_path / "float.npz",
            tmp_path / "float.json",
            semantics=semantics,
            source_absolute_indices=indices,
            frames=_frames(len(indices), dtype=np.float32),
        )
    wrong = _frames(len(indices))
    wrong["wrist_left"] = wrong["wrist_left"][:-1]
    with pytest.raises(ValueError, match="must have shape"):
        frame_cache.write_frame_artifact(
            tmp_path / "time.npz",
            tmp_path / "time.json",
            semantics=semantics,
            source_absolute_indices=indices,
            frames=wrong,
        )

    zero_height = _frames(len(indices))
    zero_height["wrist_left"] = np.empty((len(indices), 0, 6, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must have shape"):
        frame_cache.write_frame_artifact(
            tmp_path / "zero-height.npz",
            tmp_path / "zero-height.json",
            semantics=semantics,
            source_absolute_indices=indices,
            frames=zero_height,
        )


def test_every_frame_provenance_input_changes_semantic_digest():
    manifest, indices = _numeric_manifest_and_indices()
    original = frame_cache.frame_semantics(
        manifest,
        source_absolute_indices=indices,
        view_order=VIEW_ORDER,
        decoder=DECODER,
        exporter_git_sha="deadbeef",
    )

    changed_manifest = dict(manifest)
    changed_manifest["numeric_artifact"] = dict(manifest["numeric_artifact"], sha256="d" * 64)
    changed_artifact = frame_cache.frame_semantics(
        changed_manifest,
        source_absolute_indices=indices,
        view_order=VIEW_ORDER,
        decoder=DECODER,
        exporter_git_sha="deadbeef",
    )
    changed_exporter = frame_cache.frame_semantics(
        manifest,
        source_absolute_indices=indices,
        view_order=VIEW_ORDER,
        decoder=DECODER,
        exporter_git_sha="cafebabe",
    )
    changed_decoder = frame_cache.frame_semantics(
        manifest,
        source_absolute_indices=indices,
        view_order=VIEW_ORDER,
        decoder=frame_cache.DecoderSpec(lerobot_git_sha="a" * 40, pyav_version="16.0.0"),
        exporter_git_sha="deadbeef",
    )
    assert changed_artifact["semantic_sha256"] != original["semantic_sha256"]
    assert changed_exporter["semantic_sha256"] != original["semantic_sha256"]
    assert changed_decoder["semantic_sha256"] != original["semantic_sha256"]


def test_cache_hit_requires_current_exporter_and_decoder_revision(tmp_path):
    manifest, indices = _numeric_manifest_and_indices()
    _write(tmp_path, manifest, indices)

    with pytest.raises(frame_cache.FrameCacheError, match="exporter revision changed"):
        frame_cache.load_episode_frames(
            tmp_path,
            manifest,
            source_absolute_indices=indices,
            view_order=VIEW_ORDER,
            expected_exporter_git_sha="cafebabe",
            expected_decoder=DECODER,
        )
    with pytest.raises(frame_cache.FrameCacheError, match="decoder contract changed"):
        frame_cache.load_episode_frames(
            tmp_path,
            manifest,
            source_absolute_indices=indices,
            view_order=VIEW_ORDER,
            expected_exporter_git_sha="deadbeef",
            expected_decoder=frame_cache.DecoderSpec(
                lerobot_git_sha="b" * 40,
                pyav_version="15.0.0",
            ),
        )
