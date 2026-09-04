from __future__ import annotations

# Tests intentionally exercise private helpers: the command has no public Python API.
# ruff: noqa: SLF001
import dataclasses

import pytest

from scripts.local import benchmark_b1k_reader_memory as benchmark


@dataclasses.dataclass
class _Metadata:
    episodes: list[dict[str, int]]

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)


def _ranges() -> list[benchmark.EpisodeRange]:
    return [
        benchmark.EpisodeRange(7, 0, 32),
        benchmark.EpisodeRange(9, 32, 80),
    ]


def test_block_random_uses_same_sample_set_for_every_k() -> None:
    plans = {
        block_size: benchmark._build_block_random_plan(
            _ranges(), samples=64, block_size=block_size, comparison_block_size=16, seed=42
        )
        for block_size in (1, 2, 4, 8, 16)
    }

    assert all(sorted(plan) == sorted(plans[1]) for plan in plans.values())
    assert len({benchmark._plan_hash(plan, sort_indices=True) for plan in plans.values()}) == 1
    assert len({benchmark._plan_hash(plan, sort_indices=False) for plan in plans.values()}) > 1


@pytest.mark.parametrize("block_size", [2, 4, 8, 16])
def test_block_random_preserves_order_inside_each_block(block_size: int) -> None:
    plan = benchmark._build_block_random_plan(
        _ranges(), samples=64, block_size=block_size, comparison_block_size=16, seed=3
    )

    for offset in range(0, len(plan), block_size):
        block = plan[offset : offset + block_size]
        assert block == list(range(block[0], block[0] + block_size))
        assert not (block[0] < 32 <= block[-1])


def test_block_random_is_deterministic() -> None:
    kwargs = {"samples": 64, "block_size": 8, "comparison_block_size": 16, "seed": 123}
    assert benchmark._build_block_random_plan(_ranges(), **kwargs) == benchmark._build_block_random_plan(
        _ranges(), **kwargs
    )


@pytest.mark.parametrize(
    ("samples", "block_size", "comparison_block_size"),
    [(63, 1, 16), (64, 3, 16), (64, 0, 16)],
)
def test_block_random_rejects_incomparable_plans(samples: int, block_size: int, comparison_block_size: int) -> None:
    with pytest.raises(ValueError, match=r"block|divisible"):
        benchmark._build_block_random_plan(
            _ranges(),
            samples=samples,
            block_size=block_size,
            comparison_block_size=comparison_block_size,
            seed=0,
        )


def test_episode_ranges_are_relative_to_filtered_dataset() -> None:
    meta = _Metadata(
        [
            {"dataset_from_index": 0, "dataset_to_index": 10},
            {"dataset_from_index": 10, "dataset_to_index": 22},
            {"dataset_from_index": 22, "dataset_to_index": 31},
        ]
    )

    assert benchmark._episode_ranges(meta, [0, 2]) == [
        benchmark.EpisodeRange(0, 0, 10),
        benchmark.EpisodeRange(2, 10, 19),
    ]


def test_latency_summary_uses_interpolated_percentiles() -> None:
    summary = benchmark._latency_summary([4.0, 1.0, 3.0, 2.0])

    assert summary["mean_s"] == pytest.approx(2.5)
    assert summary["p50_s"] == pytest.approx(2.5)
    assert summary["p95_s"] == pytest.approx(3.85)
    assert summary["p99_s"] == pytest.approx(3.97)
    assert summary["max_s"] == 4.0
