"""Episode-aware block shuffling for video datasets.

The sampler preserves ordinary epoch semantics: every dataset index appears once
per epoch. It only changes the permutation by keeping up to ``block_size``
consecutive frames from one episode adjacent. Blocks never cross an episode
boundary.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import random


@dataclasses.dataclass(frozen=True)
class EpisodeRange:
    episode_index: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(f"Invalid episode range: {self}")


class EpisodeBlockSampler:
    """Shuffle episode-local frame blocks, advancing to a new seed each epoch."""

    def __init__(self, episode_ranges: Sequence[EpisodeRange], *, block_size: int, seed: int):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not episode_ranges:
            raise ValueError("episode_ranges must not be empty")

        expected_start = 0
        for episode_range in episode_ranges:
            if episode_range.start != expected_start:
                raise ValueError("episode_ranges must be contiguous and start at zero")
            expected_start = episode_range.stop

        self._episode_ranges = tuple(episode_ranges)
        self._block_size = block_size
        self._seed = seed
        self._epoch = 0
        self._length = expected_start

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[int]:
        blocks = [
            range(start, min(start + self._block_size, episode_range.stop))
            for episode_range in self._episode_ranges
            for start in range(episode_range.start, episode_range.stop, self._block_size)
        ]
        random.Random(self._seed + self._epoch).shuffle(blocks)
        self._epoch += 1
        return (index for block in blocks for index in block)


def episode_ranges_from_lerobot(dataset: object) -> list[EpisodeRange]:
    """Extract relative episode ranges through OpenPI's transform wrappers."""
    base_dataset = dataset
    while not hasattr(base_dataset, "meta") and hasattr(base_dataset, "_dataset"):
        base_dataset = base_dataset._dataset  # noqa: SLF001 - unwrap OpenPI's dataset decorator.

    if not hasattr(base_dataset, "meta"):
        raise TypeError("Could not find LeRobot metadata through dataset wrappers")

    meta = base_dataset.meta
    selected = base_dataset.episodes
    episode_indices = range(int(meta.total_episodes)) if selected is None else sorted(int(index) for index in selected)

    result = []
    relative_start = 0
    for episode_index in episode_indices:
        episode = meta.episodes[episode_index]
        length = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
        result.append(EpisodeRange(episode_index, relative_start, relative_start + length))
        relative_start += length

    if relative_start != len(base_dataset):
        raise RuntimeError(
            f"Episode metadata covers {relative_start} samples, but the loaded dataset contains {len(base_dataset)}"
        )
    return result
