from __future__ import annotations

from scripts.b1k import select_weaver_transfer_cohort


def _manifest(index: int, task: str, split: str) -> dict:
    return {
        "episode_index": index,
        "task_names": [task],
        "split": split,
        "semantic_sha256": f"{index:064x}",
    }


def test_task_matched_cohort_is_deterministic_disjoint_and_excludes_radio(monkeypatch) -> None:
    monkeypatch.setattr(
        select_weaver_transfer_cohort.b1k_dataset_adapter,
        "verify_episode_manifest",
        lambda manifest: None,
    )
    tasks = ["task_a", "task_b", "turn_on_radio"]
    train = [
        _manifest(task_index * 100 + ordinal, task, "train")
        for task_index, task in enumerate(tasks)
        for ordinal in range(4)
    ]
    held = [
        _manifest(1000 + task_index * 100 + ordinal, task, "val")
        for task_index, task in enumerate(tasks)
        for ordinal in range(2)
    ]
    kwargs = {
        "num_tasks": 2,
        "train_per_task": 4,
        "held_out_per_task": 1,
        "exclude_tasks": ("turn_on_radio",),
        "seed": 7,
    }

    selected = select_weaver_transfer_cohort.select_cohort(train, held, **kwargs)
    repeated = select_weaver_transfer_cohort.select_cohort(train, held, **kwargs)

    assert selected == repeated
    assert selected["tasks"] == ["task_a", "task_b"]
    assert len(selected["train_episode_indices"]) == 8
    assert len(selected["held_out_episode_indices"]) == 2
    assert not set(selected["train_episode_indices"]) & set(selected["held_out_episode_indices"])
