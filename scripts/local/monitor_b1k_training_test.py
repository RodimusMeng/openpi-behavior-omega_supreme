# ruff: noqa: SLF001

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("monitor_b1k_training.py")
    spec = importlib.util.spec_from_file_location("monitor_b1k_training", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


monitor = _load_module()


def _sample(timestamp: float, memory_gib: float, gpu_util: int) -> dict:
    return {
        "timestamp": timestamp,
        "trainer": {"running": True, "rss_bytes": 2 * monitor.GIB},
        "workers": {
            "count": 8,
            "cpu_pct": 800.0,
            "rss_bytes": 32 * monitor.GIB,
            "fds": 96,
        },
        "cgroup_memory": {"current_bytes": memory_gib * monitor.GIB},
        "gpus": [
            {"index": index, "utilization_pct": gpu_util, "memory_used_mib": 100, "memory_total_mib": 200}
            for index in range(4)
        ],
        "host_cpu": {"idle_pct": 50.0, "iowait_pct": 1.0},
    }


def test_summarize_window_computes_gpu_and_memory_metrics() -> None:
    samples = [
        _sample(0.0, 10.0, 100),
        _sample(60.0, 11.0, 0),
        _sample(120.0, 12.0, 0),
        _sample(180.0, 13.0, 100),
    ]

    summary = monitor._summarize_window(samples, sample_interval=60.0)

    assert summary["average_gpu_util_pct"] == 50.0
    assert summary["all_gpu_zero_pct"] == 50.0
    assert summary["all_gpu_at_least_80_pct"] == 50.0
    assert summary["longest_all_gpu_zero_s"] == 120.0
    assert abs(summary["cgroup_memory_slope_gib_per_min"] - 1.0) < 1e-9
    assert summary["worker_count_min"] == 8
    assert summary["worker_fds_max"] == 96


def test_parse_train_log_extracts_metrics_and_checkpoint(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "Initialized data loader:\n"
        "Initialized train state:\n"
        "Step 300: grad_norm=0.8031, loss=0.1090, param_norm=1802.3829\n"
        "100%|x| 1000/1000 [25:00<00:00, 2.21s/it]\n"
        "Waiting for checkpoint manager to finish\n"
        "Finished saving checkpoint (finalized tmp dir)\n"
    )

    summary = monitor._parse_train_log(log_path)

    assert summary is not None
    assert summary["initialized_data_loader"]
    assert summary["initialized_train_state"]
    assert summary["last_step"] == {
        "step": 300,
        "grad_norm": 0.8031,
        "loss": 0.109,
        "param_norm": 1802.3829,
    }
    assert summary["progress"] == {"step": 1000, "total": 1000, "seconds_per_iteration": 2.21}
    assert summary["checkpoint_save_count"] == 1
    assert summary["waiting_for_checkpoint_manager"]


def test_parse_train_log_supports_tqdm_loggable_progress(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "05:38:35.100 [I] Progress on: 1.30kit/1.30kit rate:6.6s/it remaining:00:00 elapsed:15:32 postfix:-\n"
    )

    summary = monitor._parse_train_log(log_path)

    assert summary is not None
    assert summary["progress"] == {"step": 1300, "total": 1300, "seconds_per_iteration": 6.6}


def test_make_summary_excludes_warmup_and_cooldown() -> None:
    samples = [_sample(float(second), 10.0, 100) for second in range(11)]

    summary = monitor._make_summary(
        samples,
        sample_interval=1.0,
        warmup_seconds=2.0,
        cooldown_seconds=2.0,
        window_seconds=3.0,
        metadata={"pid": 1},
        train_log=None,
        exit_reason="process_exited",
    )

    assert summary["overall"]["samples"] == 11
    assert summary["steady"]["samples"] == 7
    assert summary["last_steady_window"]["samples"] == 4
