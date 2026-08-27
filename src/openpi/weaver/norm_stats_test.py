import numpy as np
import pytest

from openpi.weaver import norm_stats
from openpi.weaver.behavior_precomputed import NumericEpisode


def _numeric(states: np.ndarray, actions: np.ndarray) -> NumericEpisode:
    length = len(states)
    valid = np.ones(length, dtype=bool)
    valid[-1] = False
    return NumericEpisode(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        policy_actions=np.zeros((length, 23), dtype=np.float32),
        source_timestamps_s=np.arange(length, dtype=np.float64) / 5,
        source_policy_actions=np.zeros((length, 23), dtype=np.float32),
        rewards=None,
        transition_valid=valid,
        source_frame_indices=np.arange(length),
        source_absolute_indices=np.arange(length),
        timestamps_s=np.arange(length, dtype=np.float64) / 5,
        timing_errors_s=np.zeros(length, dtype=np.float64),
    )


def test_running_moments_matches_numpy_across_episode_chunks():
    values = np.arange(12 * 23, dtype=np.float32).reshape(12, 23)
    moments = norm_stats.RunningMoments(23)
    moments.update(values[:3])
    moments.update(values[3:10])
    moments.update(values[10:])
    np.testing.assert_allclose(moments.mean, values.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(moments.population_std(), values.std(axis=0), rtol=1e-7)


def test_build_stats_excludes_invalid_terminal_action():
    states = np.stack([np.arange(23), np.arange(23) + 1, np.arange(23) + 3]).astype(np.float32)
    actions = np.stack([np.ones(23), np.full(23, 2), np.full(23, 999)]).astype(np.float32)
    stats = norm_stats.build_norm_stats(
        [_numeric(states, actions)],
        dataset_repo_id="b1k-v3again",
        dataset_metadata_sha256="a" * 64,
        target_hz=5.0,
        train_manifest_sha256="b" * 64,
        clamp_small_std=True,
    )
    action_stats = stats["norm_stats"]["actions"]
    assert action_stats["count"] == 2
    np.testing.assert_allclose(action_stats["mean"], 1.5)
    np.testing.assert_allclose(action_stats["std"], 0.5)


def test_near_zero_dimensions_fail_unless_clamping_is_audited():
    states = np.zeros((3, 23), dtype=np.float32)
    actions = np.zeros((3, 23), dtype=np.float32)
    episode = _numeric(states, actions)
    kwargs = {
        "dataset_repo_id": "b1k-v3again",
        "dataset_metadata_sha256": "a" * 64,
        "target_hz": 5.0,
        "train_manifest_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="Near-zero normalization std"):
        norm_stats.build_norm_stats([episode], **kwargs)
    stats = norm_stats.build_norm_stats([episode], **kwargs, clamp_small_std=True)
    assert all(stats["norm_stats"]["state"]["zero_std_mask"])
    assert min(stats["norm_stats"]["actions"]["std"]) == 1e-6
