from __future__ import annotations

import math

# Tests intentionally exercise private helpers: the command has no public Python API.
# ruff: noqa: SLF001
from typing import ClassVar

import numpy as np
import pytest

from scripts.local import audit_b1k_video_timestamps as audit


class _FakeMeta:
    total_episodes = 4
    video_keys: ClassVar = [audit.DEFAULT_VIDEO_KEY]

    def get_video_file_path(self, episode_index: int, video_key: str) -> str:
        assert video_key == audit.DEFAULT_VIDEO_KEY
        return (
            "videos/observation.rgb.left_realsense_link_camera_0/"
            f"chunk-{episode_index // 2:03d}/file-{episode_index % 2:03d}.mp4"
        )


def _query(timestamp_s: float, *, frame_index: int = 0) -> audit.TimestampQuery:
    return audit.TimestampQuery(
        episode_index=7,
        frame_index=frame_index,
        local_timestamp_s=timestamp_s,
        shifted_timestamp_s=timestamp_s,
        video_path="video.mp4",
    )


def test_nearest_timestamp_matches_uses_neighbors_on_both_sides() -> None:
    matches = audit._nearest_timestamp_matches(
        [_query(1.01, frame_index=1), _query(1.032, frame_index=2)],
        [1.0, 1.02, 1.04],
    )

    assert matches[0].nearest_timestamp_s == pytest.approx(1.0)
    assert matches[0].error_s == pytest.approx(0.01, abs=1e-7)
    assert matches[1].nearest_timestamp_s == pytest.approx(1.04)
    assert matches[1].error_s == pytest.approx(0.008, abs=1e-7)


def test_nearest_timestamp_matches_rejects_unsorted_decoded_timestamps() -> None:
    with pytest.raises(ValueError, match="nondecreasing"):
        audit._nearest_timestamp_matches([_query(1.0)], [1.1, 1.0])


def test_float64_reference_exposes_float32_timestamp_quantization() -> None:
    midpoint = 8641.63330078125
    query = _query(midpoint - 0.0001)
    decoded = [midpoint + 0.0001]

    float32_error = audit._nearest_timestamp_matches([query], decoded, dtype=np.float32)[0].error_s
    float64_error = audit._nearest_timestamp_matches([query], decoded, dtype=np.float64)[0].error_s

    assert float32_error == pytest.approx(0.0009765625)
    assert float64_error == pytest.approx(0.0002)


def test_summarize_errors_matches_strict_lerobot_guard() -> None:
    summary = audit._summarize_errors([0.0001, 0.0005, 0.0011, 0.0021], tolerance_s=0.0005)

    # LeRobot accepts only error < tolerance, so equality is a violation.
    assert summary["violates_current_tolerance"] == 3
    assert summary["counts_strictly_above_ms"] == {"0.5": 2, "1.0": 2, "2.0": 1, "5.0": 0}
    assert math.isclose(summary["max_ms"], 2.1)


def test_select_episode_indices_by_video_path() -> None:
    args = audit.Args(video_path_contains="chunk-001/file-000.mp4")

    assert audit._select_episode_indices(args, _FakeMeta()) == [2]


def test_select_episode_indices_seeded_subset_is_reproducible() -> None:
    args = audit.Args(episode_count=2, seed=11)

    first = audit._select_episode_indices(args, _FakeMeta())
    second = audit._select_episode_indices(args, _FakeMeta())

    assert first == second
    assert first == sorted(first)
    assert len(first) == 2


def test_select_episode_indices_requires_exactly_one_selector() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        audit._select_episode_indices(audit.Args(), _FakeMeta())
    with pytest.raises(ValueError, match="exactly one"):
        audit._select_episode_indices(
            audit.Args(episode_indices=(0,), episode_count=2),
            _FakeMeta(),
        )
