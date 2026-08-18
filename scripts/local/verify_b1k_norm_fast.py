import numpy as np

import openpi.transforms as transforms
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

from compute_b1k_norm_stats_fast import B1KStateActionDataset


CONFIG_NAME = "pi05_b1k_radio"


def as_int(x):
    return int(np.asarray(x))


def compare_one(
    index: int,
    fast_dataset,
    slow_dataset,
    official_transform,
):
    # --------------------------------------------------------
    # 1. Fast implementation
    # --------------------------------------------------------
    fast = fast_dataset[index]

    fast_episode = int(fast_dataset.episode_index[index])
    fast_frame = int(fast_dataset.frame_index[index])

    # --------------------------------------------------------
    # 2. Official implementation
    #
    # This WILL decode images. That's fine:
    # we only test a handful of samples.
    # --------------------------------------------------------
    raw = slow_dataset[index]

    slow_episode = as_int(raw["episode_index"])
    slow_frame = as_int(raw["frame_index"])

    # Before comparing values, first prove that index i means
    # the same physical frame in both datasets.
    assert slow_episode == fast_episode, (
        f"Episode mismatch at dataset index {index}: "
        f"fast={fast_episode}, slow={slow_episode}"
    )

    assert slow_frame == fast_frame, (
        f"Frame mismatch at dataset index {index}: "
        f"fast={fast_frame}, slow={slow_frame}"
    )

    slow = official_transform(raw)

    # --------------------------------------------------------
    # 3. Shape checks
    # --------------------------------------------------------
    assert fast["state"].shape == slow["state"].shape
    assert fast["actions"].shape == slow["actions"].shape

    # --------------------------------------------------------
    # 4. Numerical comparison
    # --------------------------------------------------------
    state_error = np.max(
        np.abs(
            np.asarray(fast["state"])
            - np.asarray(slow["state"])
        )
    )

    action_error = np.max(
        np.abs(
            np.asarray(fast["actions"])
            - np.asarray(slow["actions"])
        )
    )

    np.testing.assert_allclose(
        fast["state"],
        slow["state"],
        rtol=0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        fast["actions"],
        slow["actions"],
        rtol=0,
        atol=1e-6,
    )

    print(
        f"PASS "
        f"index={index:6d} "
        f"episode={fast_episode:3d} "
        f"frame={fast_frame:5d} "
        f"state_err={state_error:.3e} "
        f"action_err={action_error:.3e}"
    )


def indices_for_episode(
    fast_dataset,
    episode: int,
):
    indices = np.flatnonzero(
        fast_dataset.episode_index == episode
    )

    if len(indices) == 0:
        raise ValueError(f"Episode {episode} not found")

    start = int(indices[0])
    end = int(indices[-1])
    middle = (start + end) // 2

    # H = 32:
    #
    # end - 31 : exactly reaches terminal action, no padding
    # end - 30 : first case that requires terminal padding
    #
    candidates = [
        start,
        middle,
        max(start, end - 40),
        max(start, end - 31),
        max(start, end - 30),
        max(start, end - 20),
        max(start, end - 1),
        end,
    ]

    return sorted(set(candidates))


def main():
    # --------------------------------------------------------
    # Resolve exact training config.
    # --------------------------------------------------------
    config = _config.get_config(CONFIG_NAME)

    data_config = config.data.create(
        config.assets_dirs,
        config.model,
    )

    # --------------------------------------------------------
    # Fast dataset.
    # --------------------------------------------------------
    print("Creating FAST dataset...")

    fast_dataset = B1KStateActionDataset(
        data_config=data_config,
        action_horizon=config.model.action_horizon,
    )

    # --------------------------------------------------------
    # Official dataset.
    #
    # This is the original data reader.
    # --------------------------------------------------------
    print()
    print("Creating OFFICIAL dataset...")

    slow_dataset = _data_loader.create_b1k_dataset(
        data_config=data_config,
        action_horizon=config.model.action_horizon,
    )

    # Exact transforms used by official norm-stat path:
    #
    # RepackTransform
    # B1KInputs
    # MappedDeltaActions
    official_transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ]
    )

    # --------------------------------------------------------
    # Test several different episodes.
    # --------------------------------------------------------
    episodes_to_test = [
        0,
        37,
        123,
        199,
    ]

    test_indices = []

    for episode in episodes_to_test:
        test_indices.extend(
            indices_for_episode(
                fast_dataset,
                episode,
            )
        )

    print()
    print(f"Testing {len(test_indices)} samples...")
    print()

    for index in test_indices:
        compare_one(
            index,
            fast_dataset,
            slow_dataset,
            official_transform,
        )

    print()
    print("=" * 60)
    print("ALL DIFFERENTIAL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()