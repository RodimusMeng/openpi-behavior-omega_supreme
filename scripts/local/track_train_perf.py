from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


STEP_RE = re.compile(r"Progress on:\s+([0-9.]+)(?:it)?/([0-9.]+)(?:kit|it)?")


@dataclass
class Sample:
    ts: float
    step: float | None
    ram_gib: float
    gpu_util: list[float]
    gpu_mem_util: list[float]
    gpu_mem_mib: list[float]
    gpu_power_w: list[float]


def read_cgroup_ram_gib() -> float:
    with open("/sys/fs/cgroup/memory.current") as f:
        return int(f.read().strip()) / 1024**3


def read_gpu():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)

    util, mem_util, mem, power = [], [], [], []

    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        _, u, mu, m, p = parts
        util.append(float(u))
        mem_util.append(float(mu))
        mem.append(float(m))
        power.append(float(p))

    return util, mem_util, mem, power


def read_latest_step(log_path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            ["tail", "-n", "120", str(log_path)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    matches = []

    for line in out.splitlines():
        m = STEP_RE.search(line)
        if m:
            matches.append(float(m.group(1)))

    return matches[-1] if matches else None


def percentile(xs, q):
    if not xs:
        return float("nan")

    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo

    return ys[lo] * (1 - frac) + ys[hi] * frac


def format_window(samples, batch_size, label):
    if len(samples) < 2:
        return f"{label}\n  collecting..."

    dt = samples[-1].ts - samples[0].ts

    step_samples = [s for s in samples if s.step is not None]

    sec_per_it = float("nan")
    samples_per_sec = float("nan")
    dstep = float("nan")

    if len(step_samples) >= 2:
        dstep = step_samples[-1].step - step_samples[0].step
        step_dt = step_samples[-1].ts - step_samples[0].ts

        if dstep > 0:
            sec_per_it = step_dt / dstep
            samples_per_sec = batch_size / sec_per_it

    all_utils = [u for s in samples for u in s.gpu_util]

    avg_gpu = mean(all_utils)
    p10 = percentile(all_utils, 0.10)
    p50 = percentile(all_utils, 0.50)
    p90 = percentile(all_utils, 0.90)

    active80 = sum(u >= 80 for u in all_utils) / len(all_utils) * 100
    active95 = sum(u >= 95 for u in all_utils) / len(all_utils) * 100

    ram_delta = samples[-1].ram_gib - samples[0].ram_gib
    ram_rate = ram_delta / (dt / 60) if dt > 0 else 0.0

    avg_total_power = mean(
        sum(s.gpu_power_w)
        for s in samples
    )

    return (
        f"{label}\n"
        f"  step delta        : {dstep:.0f}\n"
        f"  sec / it          : {sec_per_it:.3f}\n"
        f"  samples / sec     : {samples_per_sec:.2f}\n"
        f"  GPU util mean     : {avg_gpu:.1f}%\n"
        f"  GPU util p10/50/90: {p10:.0f}% / {p50:.0f}% / {p90:.0f}%\n"
        f"  active >=80%      : {active80:.1f}%\n"
        f"  active >=95%      : {active95:.1f}%\n"
        f"  avg total GPU P   : {avg_total_power:.0f} W\n"
        f"  RAM delta         : {ram_delta:+.2f} GiB\n"
        f"  RAM slope         : {ram_rate:+.3f} GiB/min"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    csv_path = Path(args.csv)
    summary_path = Path(args.summary)

    history = deque()
    all_history = []

    is_new = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)

        if is_new:
            writer.writerow([
                "timestamp",
                "step",
                "ram_gib",
                "gpu_util",
                "gpu_mem_util",
                "gpu_mem_mib",
                "gpu_power_w",
            ])
            f.flush()

        while True:
            ts = time.time()
            step = read_latest_step(log_path)
            ram = read_cgroup_ram_gib()
            util, mem_util, mem, power = read_gpu()

            sample = Sample(
                ts=ts,
                step=step,
                ram_gib=ram,
                gpu_util=util,
                gpu_mem_util=mem_util,
                gpu_mem_mib=mem,
                gpu_power_w=power,
            )

            history.append(sample)
            all_history.append(sample)

            cutoff = ts - 30 * 60
            while history and history[0].ts < cutoff:
                history.popleft()

            writer.writerow([
                ts,
                step,
                ram,
                ";".join(map(str, util)),
                ";".join(map(str, mem_util)),
                ";".join(map(str, mem)),
                ";".join(map(str, power)),
            ])
            f.flush()

            def window(seconds):
                return [s for s in history if s.ts >= ts - seconds]

            lines = [
                "========== B1K TRAIN PERF ==========",
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "",
                f"latest step       : {step}",
                f"current RAM       : {ram:.2f} GiB",
                f"GPU util          : {' / '.join(f'{x:.0f}%' for x in util)}",
                f"GPU mem util      : {' / '.join(f'{x:.0f}%' for x in mem_util)}",
                f"GPU memory        : {' / '.join(f'{x/1024:.1f}G' for x in mem)}",
                f"GPU power         : {' / '.join(f'{x:.0f}W' for x in power)}",
                "",
                format_window(window(5 * 60), args.batch_size, "[ LAST 5 MIN ]"),
                "",
                format_window(window(10 * 60), args.batch_size, "[ LAST 10 MIN ]"),
                "",
                format_window(window(30 * 60), args.batch_size, "[ LAST 30 MIN ]"),
                "",
                format_window(all_history, args.batch_size, "[ TOTAL SINCE MONITOR START ]"),
                "",
                f"CSV: {csv_path}",
                "====================================",
            ]

            tmp = Path(str(summary_path) + ".tmp")
            tmp.write_text("\n".join(lines))
            os.replace(tmp, summary_path)

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
