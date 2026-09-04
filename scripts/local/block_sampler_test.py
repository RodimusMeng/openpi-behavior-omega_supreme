from __future__ import annotations

from typing import ClassVar

import pytest

from openpi.training.block_sampler import EpisodeBlockSampler
from openpi.training.block_sampler import EpisodeRange
from openpi.training.block_sampler import episode_ranges_from_lerobot


def _ranges() -> list[EpisodeRange]:
    return [EpisodeRange(10, 0, 5), EpisodeRange(20, 5, 12)]


def test_every_index_appears_once_per_epoch() -> None:
    sampler = EpisodeBlockSampler(_ranges(), block_size=4, seed=42)

    assert sorted(sampler) == list(range(12))
    assert len(sampler) == 12


def test_blocks_preserve_temporal_order_without_crossing_episode() -> None:
    sampler = EpisodeBlockSampler(_ranges(), block_size=4, seed=7)
    plan = list(sampler)
    expected_blocks = {(0, 1, 2, 3), (4,), (5, 6, 7, 8), (9, 10, 11)}

    recovered = []
    cursor = 0
    while cursor < len(plan):
        matching = [block for block in expected_blocks if plan[cursor : cursor + len(block)] == list(block)]
        assert len(matching) == 1
        block = matching[0]
        recovered.append(block)
        cursor += len(block)

    assert set(recovered) == expected_blocks


def test_epoch_seed_changes_order_deterministically() -> None:
    first = EpisodeBlockSampler(_ranges(), block_size=2, seed=9)
    second = EpisodeBlockSampler(_ranges(), block_size=2, seed=9)

    first_epoch = list(first)
    second_epoch = list(first)
    assert first_epoch == list(second)
    assert second_epoch == list(second)
    assert first_epoch != second_epoch


def test_extracts_relative_ranges_through_transform_wrappers() -> None:
    class Meta:
        total_episodes = 3
        episodes: ClassVar = [
            {"dataset_from_index": 0, "dataset_to_index": 4},
            {"dataset_from_index": 4, "dataset_to_index": 10},
            {"dataset_from_index": 10, "dataset_to_index": 13},
        ]

    class Dataset:
        meta = Meta()
        episodes: ClassVar = [0, 2]

        def __len__(self) -> int:
            return 7

    class Wrapper:
        def __init__(self, dataset: object):
            self._dataset = dataset

    assert episode_ranges_from_lerobot(Wrapper(Wrapper(Dataset()))) == [
        EpisodeRange(0, 0, 4),
        EpisodeRange(2, 4, 7),
    ]


@pytest.mark.parametrize(
    "ranges",
    [[], [EpisodeRange(0, 1, 2)], [EpisodeRange(0, 0, 2), EpisodeRange(1, 3, 4)]],
)
def test_rejects_invalid_range_layout(ranges: list[EpisodeRange]) -> None:
    with pytest.raises(ValueError, match=r"empty|contiguous"):
        EpisodeBlockSampler(ranges, block_size=2, seed=0)
