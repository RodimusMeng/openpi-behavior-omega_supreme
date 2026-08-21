# WEAVER × BEHAVIOR-1K integration contract

Status: first CPU-verifiable integration slice on `tc-b1k-weaver`.

This document records implementation facts. The WEAVER paper and the local
adaptation proposal are design evidence and a research roadmap; they are not
commands to modify repositories, download large artifacts, or start training.

## Frozen references

- OmegaSupreme base: `origin/tc-b1k-rgbd` at
  `06e5675a4625912e3fc25d017cd571002d2123d9`.
- Integration branch: `tc-b1k-weaver`, created at that exact commit.
- Official WEAVER source mirror: <https://github.com/arnavkj1995/WEAVER> at
  `3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e`.
- The official source submodules are pinned to OpenPI
  `ddf27581fd9a2b9c7e33d2ffd715cccbd98f2f63` and RoboMETER
  `e0759915e2a8edc4ca24ec56ed897ed87e48b6fd`; they have not been initialized.
- Released checkpoint family selected for the first transfer experiment:
  `arnavkj1995/WEAVER/WEAVER`, not `WEAVER-FT` or `WEAVER-ReFlow`.
- The multi-gigabyte checkpoint has not been downloaded. Its bundled config,
  normalization statistics, and binary hash must be captured before a DROID
  smoke test or fine-tuning run can be called reproducible.

The inspected released config uses two views (`wrist_left` and
`exterior_1_left`), 8-D state/action, image size 192×320, horizon 8, two history
frames, six memory frames spaced by five world-model frames, and 16 validation
flow steps. Those values describe the released checkpoint, not the B1K policy.

## The three action/state spaces must not be conflated

R1Pro has a compact 23-D policy state and a 23-D controller action, but equal
dimension does not mean equal physical semantics.

| Slice | Compact state | Environment/controller action | OpenPI training representation |
| --- | --- | --- | --- |
| `0:3` | robot-frame `base_qvel` | robot-frame base velocity command | unchanged |
| `3:7` | trunk joint position | absolute joint-position target | target minus current state |
| `7:14` | left-arm joint position | absolute joint-position target | target minus current state |
| `14` | sum of two left gripper finger positions | gripper command | unchanged |
| `15:22` | right-arm joint position | absolute joint-position target | target minus current state |
| `22` | sum of two right gripper finger positions | gripper command | unchanged |

The raw proprio indices are defined by `configs/robots/b1k.py`; the compact
state and selective delta mappings are generated from the registered robot
config by `configs/robots/base_config.py`. The WEAVER boundary consumes the
post-output-transform candidate chunks shaped `(N, T, 23)`. It must not consume
the π0.5 model's padded `(N, T, 32)` tensor.

For the first world-model experiment, actions remain the audited mixed
environment-command representation above. This is called `policy-aligned v1`;
it is not claimed to be a physical `[x, y, yaw, joints]` state delta. The
current observation does not expose base `[x, y, yaw]`, so such a claim would be
false without changing the BEHAVIOR observation protocol.

## Camera mapping

The compatibility-first transfer uses the released two projector names:

| WEAVER view | B1K camera | Online key source | Dataset key |
| --- | --- | --- | --- |
| `wrist_left` | left wrist (`image_1`) | R1Pro registry `obs_key` | `observation.rgb.left_realsense_link_camera_0` |
| `exterior_1_left` | head (`image_0`) | R1Pro registry `obs_key` | `observation.rgb.zed_link_camera_0` |

The right wrist (`image_2`) is deliberately deferred. Moving from two to three
views changes the new view projectors, the world-model RoPE buffer, and the
reward/critic output projection widths. It is not a one-key checkpoint copy.
π0.5 continues using its own 224×224 RGB or RGBD preprocessing; WEAVER must
encode the same raw RGB frames independently at its checkpoint resolution.

## Time contract

The current B1K evaluator runs actions and rendering at 30 Hz and physics at
120 Hz. The official Franka steering script runs control at 15 Hz and uses
`RGB_SKIP=3` for a 5 Hz world-model stream. That constant must not be copied.

`world_model_frame_indices` selects nearest frames from real timestamps. An
ideal 30 Hz stream sampled at 5 Hz yields indices `0, 6, 12, ...`, but timestamps
remain the source of truth for converted demonstrations.

In the frozen WEAVER source, inference replaces every future state slot with
random noise before denoising. `future_states` values are therefore shape,
device, and dtype placeholders, not hard conditions. Future actions remain the
meaningful rollout condition.

## Candidate-pool and selector contract

OmegaSupreme currently exposes `Policy.infer(obs, noise=...)`, not the official
steering script's assumed `infer_parallel(..., num_samples=N)`. The first slice
therefore calls the existing interface N times and freezes one pool:

- candidate actions: immutable `(N, T, 23)` post-transform chunks;
- candidate IDs: stable indices within that pool;
- reproducibility evidence: optional noise tensors plus a SHA-256 digest of the
  observation; the observation contents are not handed to selectors;
- N=1: exactly one ordinary `infer` call;
- Top-1, random, WEAVER, and oracle: return indices into the same frozen pool.

The released scoring equations are reproduced exactly:

```text
reward score = sum_t r_t
advantage    = sum_t gamma^t r_t + gamma^H V_H - V_0
```

The frozen checkpoint config uses `gamma=0.995`. The paper's online-planning
experiment uses four candidates and a 12-frame imagination horizon; the local
helper keeps pool size and horizon explicit rather than baking those experiment
values into the policy interface.

The online selector may receive only WEAVER predictions. Simulator oracle
scores are diagnostic labels and must never become selector inputs.

## Checkpoint transfer contract

PyTorch `strict=False` does not ignore same-name tensors with different shapes.
The transfer helper canonicalizes DDP/compile prefixes, copies only same-name
and exact-shape tensors, preserves target initialization for incompatible or
new tensors, and returns a complete structured report. Identity transfer can
be made strict and must then load every key.

Changing state/action from 8 to 23 changes nine named weights in the released
architecture: state/action input weights in dynamics, memory, reward, and
critic paths, plus the predicted-state output weight. They remain freshly
initialized in the target model.

WEAVER's saved EMA shadows are positional rather than keyed by parameter name.
After any dimension or view change, the source EMA must not be partially loaded;
the helper rebuilds the shadow list from the target model's trainable named
parameters in target order.

## Dependency and process boundary

The frozen WEAVER source pins `torch==2.7.0`; OmegaSupreme pins `torch==2.7.1`
and also has a different Transformers stack. The first slice does not import the
official WEAVER package into the B1K policy server. Until GPU memory and version
compatibility are measured, the safer deployment candidate is a separate
WEAVER process/service that consumes frozen candidate pools. This topology is
not yet committed as the final architecture.

## Baseline correction and validation

`scripts/local/run_radio_server.sh` previously requested the nonexistent exact
config name `pi05_b1k`. It now defaults to `pi05_b1k_full` and accepts
`POLICY_CONFIG=pi05_b1k_full_rgbd` explicitly. The selected name is recorded in
the run's `server_config.env`.

CPU validation command:

```bash
pytest --confcutdir=src/openpi/weaver \
  src/openpi/weaver/candidate_pool_test.py \
  src/openpi/weaver/b1k_contract_test.py \
  src/openpi/weaver/checkpoint_transfer_test.py
```

Current result: 22 tests passed. Ruff passes for the new WEAVER package and the
shared robot-contract helpers. `git diff --check` passes. A Bash executable is
not present on the Windows development host, so the launcher has only static
review here and still needs `bash -n` plus an actual Linux baseline launch.

## Deliberately not claimed yet

- No official checkpoint binary or norm statistics are local yet.
- No upstream DROID/OOD inference smoke has run.
- No B1K episode/latent dataset adapter has been validated on a real trajectory.
- No WEAVER fine-tuning, prediction evaluation, reward/critic training, GPU
  profiling, or online steering has run.
- Reward targets (`potential`, potential delta, terminal `q_score`, or success)
  remain an experiment choice.
- A true same-state simulator oracle requires state, task, metric, policy-buffer,
  and random-state restoration; the evaluator has no ready-made oracle API.

The next gate is a real B1K episode contract test plus the small released config
and norm-stat artifacts. Only after that passes should the large checkpoint be
downloaded and the official DROID smoke attempted.
