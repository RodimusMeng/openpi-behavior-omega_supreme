# WEAVER × BEHAVIOR-1K `numeric-full-v6` 数据准备实验报告

> 本文在最初的 resume 故障日志上续写。早期记录解决的是 correctness、
> artifact 事务性和恢复安全；后续实验解决的是 full-data 吞吐、错误 benchmark
> 结论和 GPFS 上的生产配置。本文所说的“v6 resume 成功”，指已有 pair 被正确
> 复用、程序跨过历史断点并持续写入新 episode；除非同时观察到 20,000 个完整
> pair、最终 `summary.json` 和完整 `manifest.jsonl`，不把它扩大解释为全数据链路
> 已全部完成。

## 1. 实验对象与当前结论

- 数据集：BEHAVIOR-1K 2026 challenge demos `v3again`，20,000 episodes，
  30 Hz，R1Pro。
- source numeric contract：`observation.state=61D`、policy/controller
  `action=23D`，保留 30 Hz timestamp、reward、terminated、truncated。
- WEAVER numeric contract：5 Hz、23D compact state、23D observed state-delta
  training action，并单独保留 sampled source policy action。
- 生产 artifact：`$WEAVER_WORK_ROOT/numeric-full-v6`。
- active branch：`tc-b1k-weaver`，canonical integration 基于
  `tc-b1k-core@33fe694`。
- 本轮性能修复提交：`4382b87`。
- 当前实测生产选择：

  ```text
  prepare_weaver_dataset_parallel.py
  workers=1
  worker_threads=1
  manifest_flush_every=256
  ```

这里的 `parallel.py + workers=1` 并不矛盾。实际价值来自独立 worker、单
coordinator、快速 resume、atomic commit 和 batched manifest，而不是盲目增加
进程数。真实 write benchmark 表明更多 worker 会在共享 GPFS/Arrow/压缩写路径上
互相争抢。

## 2. 证据等级与指标约定

- **服务器事实**：来自实际日志、artifact 数量/hash、Git revision 或进程状态。
- **实现事实**：由当前源码和回归测试直接支持。
- **解释**：由多轮 timing 和对照实验支持，但仍可能受 page cache、GPFS 负载和
  episode 长度影响。
- **尚未完成的 gate**：不能由 helper test、dry-run 或部分 episode 推断。

吞吐比较优先使用 `source_rows_per_second`，不用 `episodes_per_minute` 作为主要
指标。B1K episode 的 raw-row 数差异很大；同样的 ep/min 可能对应完全不同的
30 Hz 数据量。ep/min 只在完全相同 episode 集合上作为辅助指标。

### 2.1 本报告覆盖的版本阶段

| version | 角色 | 本报告中的意义 |
|---|---|---|
| `226e8fb` | 原实验日志截止点 | output lock、strict late-pair reuse；当时尚无真实 full resume 结论 |
| `b84a3c5` | archive lineage | 加入 isolated preparation benchmark harness |
| `bfac330` | archive lineage | 提取 stage timings、完成 spawn worker/coordinator、process-tree telemetry |
| `9be4f61` | active canonical integration | 将 WEAVER 功能作为 core 上的一层集成，旧 lineage 留在 archive |
| `4382b87` | active performance fix | batched manifest、fast resume、NumPy numeric view、指标修正 |

这里列版本是为了定位实验阶段，不要求后续工作沿 archive commits 继续开发；新的
WEAVER 工作只基于 active `tc-b1k-weaver`。

## 3. 从最初故障到 v6 的 correctness 修复

### 3.1 服务启动变量为空

最初的 `KeyError: ''` 和 `policy_dir` 为空来自 shell/启动参数，而不是 WEAVER
模型本身。后续服务入口要求显式提供 task、OpenPI policy config/checkpoint、
WEAVER config/checkpoint、adapter、norm stats 和 dataset directory，并在进入服务
循环前 fail closed。

### 3.2 terminal/reward 语义误判

旧实现曾假设 terminal marker 出现后 reward 必须为零，并试图用 terminal 驱动
裁剪。真实 episode 320 证明官方 exporter 仍会在 marker 之后记录 observation、
action 和非零 progress reward；真正的 episode boundary 是 metadata 中的
`[dataset_from_index, dataset_to_index)`。

修复后的 v6 contract：

- 完整保留 source `next.reward`、`next.terminated`、`next.truncated`；
- 5 Hz reward 是相邻 sampled source indices 上 `[i,j)` 的 30 Hz reward 和；
- terminal marker 只进入 diagnostics，不控制裁剪；
- 最后 sampled observation 因没有真实 successor，固定
  `transition_valid[-1]=False`；
- source policy action 与 WEAVER observed state-delta action 分开命名，禁止混用。

这一修复把“官方 marker 的持续状态”与“监督 transition 是否存在”分离，避免了
对官方数据作未经证实的语义改写。

### 3.3 episode artifact 的事务边界

每个 episode 的 durable truth 是同 stem 的一对文件：

```text
episodes/<split>/<episode>.npz
episodes/<split>/<episode>.json
```

NPZ 包含固定 numeric arrays；JSON 绑定 schema、dataset provenance、prompt、
reward diagnostics、path、size、SHA-256 和 keys。`manifest.jsonl` 是可重建索引，
不是 episode boundary 或 durable truth。

正式写入采用 temporary file、`fsync()` 和 `os.replace()`。因此：

- NPZ + JSON 都存在且严格校验通过：可恢复；
- 只有 NPZ 或只有 JSON：partial，必须失败；
- pair 完整但 manifest 落后：可以重建 index，不能把 pair 当作损坏；
- schema、context、size、hash 或 keys 不一致：stale/corrupt，禁止覆盖。

### 3.4 episode 199 与单 writer 修复

历史现场中 `000199.npz/.json` 本身 hash、size 和 schema 全部正确，但
`manifest.jsonl` 最大只到 198；随后还有 200 以及一个临时 204。旧进程只在启动
时建立一次 `existing_pairs`，之后看到未被该 snapshot 收录的正式 pair 就报
orphan。

`226e8fb` 增加了：

- output-directory `flock`，同一目录只允许一个 writer；
- startup scan 后出现的完整 pair 走 strict late validation；
- 匹配后返回 `reused-late`，partial/stale 继续 fail closed；
- lock owner 记录 PID/host，避免重复启动静默并发刷新 manifest。

## 4. 早期修复后来怎样被真实运行验证

最初日志只能证明 helper contract，不能证明服务器 resume 真能跨过 episode 199。
后续真实运行补上了这一证据：

1. resume 完成已有 artifact validation；
2. 进程没有再次卡在 199/200；
3. 进入新的 episode 写入阶段；
4. pair 数从历史基线继续增长，而不是反复重做旧 episode；
5. 分 split 审计没有发现 orphan：

   ```text
   train  npz=3830 json=3830 paired=3830 npz_only=0 json_only=0
   val    npz= 426 json= 426 paired= 426 npz_only=0 json_only=0
   test   npz= 600 json= 600 paired= 600 npz_only=0 json_only=0
   total paired=4856
   ```

6. 稍后的实时计数到 5205，表示相对 4856 又完成 349 个新 episode。

这证明 226e8fb 所针对的 late-pair/index 问题已经被越过；当时“完整 pair 可作为
恢复事实、manifest 可重建”的判断是正确的。它仍不等价于 20,000 episode 最终
完成，但已经从 contract test 升级为服务器真实 resume 成功。

## 5. v6 首次全量运行暴露的性能问题

### 5.1 真实 per-episode timing

新 episode 写入稳定推进后，日志给出了典型 timing：

| stage | 典型耗时/episode | 解释 |
|---|---:|---|
| `dataset_read_seconds` | 3.6–3.9 s | HuggingFace/Arrow numeric slice 与格式转换 |
| `numpy_conversion_seconds` | 0.5–1.0 s | 中间表示转 NumPy |
| `npz_compress_fsync_seconds` | 0.28–0.39 s | 压缩、写盘、fsync |
| `manifest_checkpoint_seconds` | 0.26–0.38 s | 当时每 episode 全量重建 index |
| `adapter_seconds` | 0.005–0.007 s | 61D→23D、30→5 Hz adaptation |
| `reward_validation_seconds` | 约 0.0004 s | reward/marker contract |

结论很明确：adapter 和 reward contract 已经快到可以忽略；瓶颈是数据访问和
持久化路径。把 v6 称为 numeric-only 是正确的，因为它没有读视频；但
numeric-only 不自动等于高吞吐。

### 5.2 manifest 的 O(N²) 问题

旧 serial loop 每完成一个 episode 都调用：

```python
_atomic_write_manifest_index(output_root / "manifest.jsonl", manifest_index)
```

该函数不是 append，而是按 episode index 重写全部 manifest。20,000 episodes
对应 `1+2+...+N` 的累计 line writes，并产生 20,000 次 replace/fsync。随着输出
增长，单 episode manifest 时间也会继续上升。

episode NPZ/JSON pair 已经是 durable truth，resume 也通过扫描这些 pair 重建
index，因此没有 correctness 理由要求每 episode 全量刷新。

### 5.3 看似是 Arrow I/O，实际还夹着 Torch 往返

进一步沿 LeRobot reader 追到：

```text
lerobot/datasets/dataset_reader.py
  _load_hf_dataset()
    hf_dataset.set_transform(hf_transform_to_torch)
```

而 `hf_transform_to_torch()` 会对 batch 中每个 key 的每个 row 单独执行
`torch.tensor(x)`。numeric preparation 随后又在 `_to_numpy()` 中把它转回
NumPy。实际路径是：

```text
Arrow → Python row/list → 大量小 Torch tensor → NumPy
```

对于约 10k raw rows、九个 numeric columns 的 episode，这可能产生接近十万个
小 tensor。它既不是训练所需，也不是 numeric artifact contract 所需。日志中
`dataset_read_seconds + numpy_conversion_seconds` 的组合与这一实现吻合。

这个发现改变了优化优先级：先删除无意义的 Torch round trip，再决定是否值得
实现更复杂的 32/64-episode contiguous block reader。

## 6. 第一轮 benchmark 为什么没有回答并发问题

早期 32-episode dry-run 得到：

| CPU affinity | elapsed | ep/min |
|---:|---:|---:|
| 1 core | 131 s | 14.656 |
| 4 cores | 136 s | 14.118 |
| 16 cores | 155 s | 12.387 |
| 55 cores | 135 s | 14.222 |

这轮只改变 `taskset` CPU affinity，episode loop 仍是同一个 serial Python
进程，因此它证明的是“增加可用 core 不会自动并行”，而不是“多进程 preparation
一定无效”。当时用 ep/min 外推约 20 小时也只能作为量级提示，因为 32 个 episode
长度不具代表性，dry-run 又排除了压缩和写盘。

由此新增 isolated benchmark harness 和真正的 parallel coordinator，用相同
episode 列表比较 worker count，并同时记录 process tree I/O、rows/s 和 stage
timings。

## 7. parallel coordinator 的设计与安全边界

parallel 版本不是让多个 worker 同时改最终目录：

- Linux 使用 `spawn`，不 fork/pickle Arrow 或 LeRobot runtime handle；
- 每个 worker 独立打开 numeric-only dataset view，并在进程内复用；
- worker 只写本次 run-scoped `.staging` NPZ；
- coordinator 单独执行 final `os.replace()`、JSON commit 和 manifest checkpoint；
- in-flight task 有上界；worker BLAS/OpenMP threads 固定为 1；
- partial/stale/corrupt pair 继续 fail closed；
- SIGKILL 最坏留下 staging 或 final-NPZ-only，下一次启动明确拒绝，不静默覆盖。

这使并发只发生在 read/adapt/compress 阶段，最终 namespace 仍保持单 writer。

## 8. `4382b87` 的性能修复

### 8.1 batched manifest checkpoint

serial 和 parallel 都增加 `manifest_flush_every`，默认 256。完整 20k 运行的全量
index rebuild 次数从 20,000 次降到约 80 次（78 个整 checkpoint、收尾及可能的
恢复刷新），数量级约减少 250 倍。正常结束仍写最终完整 index；异常中断时至多
丢失可重建的 index 更新，不丢已提交 pair。

### 8.2 existing pair 直接复用

旧 resume 即使 startup 已验证 pair，episode loop 仍会重新读 source Arrow、重建
arrays，甚至在 parallel worker 中重新压缩 staging NPZ，再逐数组比较。这让
5205 个已完成 episode 在每次 resume 时重复支付昂贵的 preparation 成本。

新逻辑先验证：

- JSON schema 与 semantic fields；
- dataset metadata/provenance、task、prompt、split、target Hz；
- source bounds 和 full-rate count；
- NPZ path containment、size、SHA-256 和 archive keys。

通过后，按精确集合计算：

```text
pending = selected_episode_indices - validated_pair_indices
```

只有 pending indices 进入 serial loop/process pool。startup discovery 在没有
`expected_arrays` 时只读 NPZ archive directory，不再解压全部 arrays；SHA-256
仍然覆盖完整 artifact bytes。

这也说明生产恢复不应该手工使用一个 `first_missing` 截断。pair 集合可能有中间
空洞，单个断点会漏 episode；`first_missing` 只适合显示，不能替代 set difference。

### 8.3 一次性 NumPy numeric view

每个进程初始化时只执行一次：

```python
selected = dataset.hf_dataset.select_columns(columns)
numeric_dataset = selected.with_format("numpy")
```

之后每个 episode 直接 slice 该 view。这样同时消除：

- 每 episode 重复构造 `select_columns()` view；
- LeRobot frame-oriented Arrow→Torch transform；
- 紧接着的 Torch→NumPy round trip。

### 8.4 summary 指标修正

resume summary 明确区分：

```text
selected
reused_existing
newly_prepared
remaining_after_run
```

`episodes_per_minute` 和 `source_rows_per_second` 只用本次 newly prepared 数据
计算，避免把 5k 个瞬间复用的旧 pair 计入吞吐并制造虚假加速。

本机回归：

```text
110 passed
Ruff passed
```

这些测试证明 contract/resume 行为，没有替代服务器 GPFS benchmark。

## 9. benchmark harness 本身踩过的坑

### 9.1 shell 文件没有 executable bit

直接执行 `scripts/b1k/benchmark_weaver_prepare.sh` 曾得到 `Permission denied`。
显式使用 `bash scripts/...` 可以运行；长期可由维护者单独提交 executable bit。

### 9.2 `seq` 的换行被 `read -a` 截成一个 episode

脚本要求 `BENCH_EPISODES` 是空格分隔，但：

```bash
BENCH_EPISODES="$(seq 18000 18099)"
```

产生换行。`read -r -a EPISODES <<< "$BENCH_EPISODES"` 只消费第一行，日志因此
显示 `episodes=18000`，而不是 100 episodes。正确调用是：

```bash
BENCH_EPISODES="$(seq -s ' ' 18000 18099)"
```

### 9.3 stale `OPENPI_ROOT` 让 benchmark 跑到 core checkout

一轮失败日志显示：

```text
git_revision=33fe694
episodes=18000
status=2
completed_episodes=0
```

shell prompt 虽在 `openpi-weaver`，但继承的 `$OPENPI_ROOT` 仍指向另一个
`openpi` checkout。该轮既没有运行 canonical WEAVER，也没有完成 episode，不能
作为性能证据。有效实验必须显式设置：

```bash
OPENPI_ROOT="$PWD"
```

并确认 `git_revision=4382b87`、`status=0`、`completed_episodes` 等于目标数量。

### 9.4 GPFS 容量是实验条件，不只是背景信息

日志曾显示 dataset mount 88%，output project mount 按 `df` 四舍五入为 100%，
剩余约 701 GiB。即使短 benchmark 能写入，full run 仍要检查项目 quota、inode 和
预计剩余 artifact 体积。接近 quota 时的 latency/ENOSPC 不能误判为 worker scaling
规律。

## 10. 有效 write benchmark 与生产配置

在相同 50-episode write workload 上，主要结果为：

| workers | source rows/s | ep/min | 50 episodes wall time |
|---:|---:|---:|---:|
| **1** | **4959** | **71.4** | **42 s** |
| 2 | 4033 | 66.7 | 45 s |
| 4 | 3348 | 52.6 | 57 s |

辅助轮次还记录到：

- 2 workers：4696 rows/s；
- 8 workers：1941 rows/s；
- 12 workers：2316 rows/s。

不同轮次不能只用 ep/min 横比，但 rows/s 的总体趋势稳定：worker 增加没有带来
吞吐扩展，反而造成显著下降。相对同一 50-episode workload，1 worker 的 rows/s
比 2 workers 高约 23%，比 4 workers 高约 48%。8/12 worker 的非单调波动也更像
共享 GPFS/page-cache/压缩写竞争，而不是 CPU 不足。

因此 production 选择 `parallel + 1 worker`：

- 保留已实测最好的顺序 read/write locality；
- 保留 coordinator 的 atomic final commit；
- 保留每 256 episode manifest checkpoint；
- 保留精确 pair-set resume；
- 避免多个 LeRobot/Arrow view 同时争抢 GPFS。

需要注意：`4382b87` 后 optimized serial 也具有 batched manifest、NumPy view 和
fast resume，所以“这些能力只属于 parallel”已经不是事实。选择 parallel-1 是
因为它是当前服务器真实 write benchmark 的已测生产路径，而不是因为文件名带
`parallel`。

## 11. 效果评估

### correctness

- v6 pair 审计中 train/val/test 均没有 NPZ-only 或 JSON-only；
- 历史 episode 199/200 不再阻塞 resume；
- resume 跨过既有 4856/5205 pair 后进入新写入；
- output lock 与 coordinator single-writer 保留 fail-closed 边界。

### performance

- adapter/reward 不是优化目标，避免了在 0.01 s 以内的代码上浪费工程量；
- manifest 从 O(N²) 每 episode rebuild 改为约每 256 episode rebuild；
- 已完成 episode 不再重读 source/recompress；
- numeric slice 绕过逐 row Torch tensor construction；
- 有效 write benchmark 达到约 4959 source rows/s，50 episodes/42 s；
- 生产并发从猜测性的 4/8/55 调整为证据支持的 1 worker。

不能从这些结果推出：

- 所有 episode 长度下都固定达到 71.4 ep/min；
- 20,000 episode 最终 artifact 已经齐全；
- frame export、SD3 latent cache 或 GPU training 也会有同样吞吐；
- contiguous block reader 永远没有价值。

## 12. 当前生产续跑与验收

停止旧 serial 时应在一条 `artifact=written` 日志后发送 SIGINT/Ctrl-C，不使用
SIGKILL。确认旧进程退出、同步 `4382b87` 后，直接对完整 selection resume：

```bash
uv run python scripts/b1k/prepare_weaver_dataset_parallel.py \
  --config-name pi05_b1k_full \
  --dataset-root "$B1K_DATASET_ROOT" \
  --output-dir "$WEAVER_WORK_ROOT/numeric-full-v6" \
  --held-out-task-fraction 0.1 \
  --selection-seed 2026 \
  --split-seed 42 \
  --min-sampled-frames 10 \
  --workers 1 \
  --worker-threads 1 \
  --manifest-flush-every 256
```

启动后首先核对：

```text
resume_reused_episodes == 实际 validated pair 数
episodes_to_prepare     == 20000 - validated pair 数
```

不要用人工猜测的 `first_missing` 限制 `--episode-indices`。最终 numeric gate：

1. train/val/test 合计 20,000 个同 stem NPZ/JSON pair；
2. `npz_only=0`、`json_only=0`；
3. `manifest.jsonl` 覆盖全部 pair；
4. `summary.json.processed=20000` 且 `rejected=[]`；
5. `task_prompts.json`、dataset metadata hash、held-out task split 和 schema 一致；
6. 随机抽样和边界 episode 通过 strict pair/context validator。

只有这些通过后，才进入：

```text
norm stats
→ RGB frame export
→ official WEAVER environment 中的 SD3 latent cache
→ BehaviorPrecomputed smoke
→ dynamics/full tiny-overfit
→ held-out prediction
→ WebSocket online steering / official evaluator
```

## 13. 是否还要实现 contiguous episode batch reader

目前不应仅凭旧的 3.8 s timing 立即重构 batch reader，因为该 timing 包含已经删除
的逐 row Torch transform。下一次决策必须看 `4382b87 + workers=1` 的新 stage
timings：

- 若 `dataset_read_seconds` 已降到可接受范围，且 write rows/s 与 GPFS 上限接近，
  不引入 batch reader；
- 若 NumPy formatter 后单 worker 仍主要耗在重复 Arrow slice，并且相邻 episode
  的 contiguous block A/B 明显改善 rows/s，才实现 16/32 episode block；
- block 实验必须同时记录 peak RSS、Arrow chunk boundaries、page-cache warm/cold
  顺序和 write throughput，避免用内存换出不可控的 GPFS 抖动。

## 14. 可复用经验

1. 先区分 correctness gate 和 performance gate；helper test 不能代替服务器 resume。
2. episode pair 是 durable truth，manifest 是可重建 index；索引落后不等于数据坏。
3. benchmark 必须记录 Git revision、实际 episode 列表、status 和 completed count。
4. CPU affinity 不等于并行；进程数也不等于吞吐。
5. B1K 必须以 source rows/s 为主指标，ep/min 很容易被 episode 长度误导。
6. profiling 要沿数据类型追踪；“Arrow 慢”背后可能是 Arrow→Torch→NumPy 的隐式
   格式往返。
7. GPFS workload 先测 1 worker；更多 worker 必须用相同 write workload 证明收益。
8. 优化 resume 时不能牺牲 artifact hash/context 校验，也不能把单一 first-missing
   cutoff 当成精确缺集。
9. full-data 脚本的最终产品不仅是 episode files；`summary.json`、prompt catalog、
   norm stats 和后续 provenance gates 都必须正常收尾。
