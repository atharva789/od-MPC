# od-MPC development spec

Working document. Records what has been **verified against source**, what is
**assumed**, and what must be **confirmed before the first training run**.
Update it as facts change; the distinction between the three is the point.

---

## 1. Goal

Plan MARS base motion by rolling candidate command sequences forward through a
learned latent world model and selecting the sequence whose predicted latent
lands nearest a goal-image latent.

Success is not "the video looks nice". It is: **latent rollouts track ground
truth for long enough to plan through**, measured by `drift_horizon`.

Non-goals for v0.1: manipulation, learned policies, sim-to-real transfer,
anything requiring hardware.

## 2. Verified facts

Everything here was read from source or official docs, not inferred.

### 2.1 MARS simulator

`VirtualMars` lives at
`ros2_ws/src/mars_bot/mars_sim_driver/mars_sim_driver/core.py` in
[innate-inc/innate-os](https://github.com/innate-inc/innate-os) (Apache-2.0,
`main`, read 2026-08-04).

| Member | Signature | Notes |
| --- | --- | --- |
| `__init__` | `(split_dir=None, render_wh=None, depth_render_wh=None)` | `render_wh` sets offscreen render size |
| `reset` | `() -> None` | back to spawn, zero velocity |
| `step` | `(duration: float) -> None` | advances physics by seconds |
| `set_cmd_vel` | `(vx: float, wz: float) -> None` | clamped to `MAX_LINEAR` / `MAX_YAW` |
| `pose` | `() -> tuple[float, float, float]` | ground-truth `(x, y, yaw)` |
| `render_rgb` | `(camera: str) -> np.ndarray` | cameras: `"main"`, `"wrist"` |
| `lidar_scan` | `(n_rays: int, max_range: float) -> np.ndarray` | 360°, CCW from robot `+x`, metres |
| `set_joint_target` | `(name: str, value: float) -> None` | **arm control exists** |
| `joint_targets` | `() -> dict[str, float]` | current arm targets |

Constants from the sibling `world.py`:

- `MAX_LINEAR = 0.4` (m/s), `MAX_YAW = 1.0` (rad/s)
- `SPAWN_X = -4.34`, `SPAWN_Y = -0.17`, `SPAWN_YAW_DEG = -89.8`
- `CMD_VEL_TIMEOUT_S = 0.5` — a stale command stops the base, mirroring a real
  watchdog. Control loops must issue commands faster than this.

**Physics is MuJoCo.** `core.py` calls `mujoco.mj_step` and indexes
`model.jnt_qposadr` / `jnt_dofadr`. The arm runs a PD servo (`KP_JOINT = 50.0`,
`KD_JOINT = 1.0`, `EFFORT_LIMIT = 50.0` N·m) with piecewise joint limits
mirroring the real firmware's `arm_control.cpp`.

`VirtualMars` needs neither ROS nor Docker, which is what makes headless
collection on a rented GPU box straightforward.

### 2.2 open-dreamer

[next-state/open-dreamer](https://github.com/next-state/open-dreamer),
**All Rights Reserved** — see §6.

- `ActionEncoder` accepts `binary`, `categorical`, and `continuous` channels;
  `continuous` is a single `Linear(continuous_action_dim, d_model)`.
- Every shipped dataset config sets `continuous_action_dim: 0`
  (`coinrun.yaml`, `minecraft_vpt.yaml`, `minecraft_vpt_latent.yaml`).
- The only continuous producer emits a bounded 2-vector normalised to `[-1, 1]`.
- Training data is read via `grain.sources.ArrayRecordDataSource` over
  `shard-NNNNN.array_record` files.
- Records are dicts carrying `video`, `video_shape`, and `actions`.
- Dynamics is a shortcut/flow-matching model doing **x-prediction**: it returns
  `x1_hat`, the predicted clean latent, conditioned on a step size and a signal
  level. Inference is autoregressive across frames, diffusion within a frame.

## 3. Assumptions

Flagged because they are load-bearing and **not yet verified**.

| # | Assumption | Risk if wrong | How to check |
| --- | --- | --- | --- |
| A1 | The loader reads a continuous action channel under `actions["continuous"]` | Collected data will not load | Read `dreamer/data/transforms.py` on a pinned revision |
| A2 | `continuous_action_dim: 2` trains without other config changes | Wasted training run | Single short run, watch the loss |
| A3 | 64×64 frames suffice for navigation | Poor reconstruction | Tokenizer PSNR after stage 1 |
| A4 | A 2-D action carries enough signal for the dynamics model | Weak action conditioning | Compare rollouts under different action sequences |

## 4. Pipelines

### 4.1 Data annotation

```
VirtualMars(render_wh=(64,64))
  → lidar_scan → explorer → cmd_vel
  → step(control_dt) → render_rgb + pose
  → Episode(frames, actions, poses) → ShardWriter
```

Conventions, both easy to get wrong and expensive to debug later:

- **Action indexing**: `actions[t]` is the command issued *from* `frames[t]`, so
  `(frames[t], actions[t]) -> frames[t+1]`.
- **Clipping**: the recorded action is clipped exactly as the simulator clamps,
  so the record matches what the robot actually executed.
- **Poses are never model input.** They exist only for §4.3.

Defaults: 160 steps/episode, `control_dt = 0.1` s (10 Hz, comfortably inside
`CMD_VEL_TIMEOUT_S`), 360 lidar rays, 12 m range.

### 4.2 Training

Two stages, matching upstream: tokenizer (encoder + decoder as an autoencoder),
then dynamics on frozen latents.

od-MPC only builds and runs the command. `run_stage` defaults to `dry_run=True`.

### 4.3 Validation

1. Encode held-out episodes to latents.
2. Fit ridge probe: latent → `(x, y, cos yaw, sin yaw)`. Intercept unpenalised.
3. Roll the dynamics model open-loop from a context window.
4. Decode predicted latents through the probe; compare to ground truth.
5. Report `drift_horizon(position_error, tolerance)`.

Also reported: tokenizer PSNR, and per-step position/heading error curves.

**Control needed:** a probe fitted on a model that has learned nothing should
fail. `test_probe_fails_on_unpredictable_latents` asserts this on synthetic
noise; repeat it against an untrained checkpoint before trusting any number.

### 4.4 Planning

CEM: sample `population` sequences, clip to `[-1, 1]`, score, refit to elites,
repeat `iterations` times. Execute only the first action, replan. Warm-start
from the previous mean.

Cost = terminal latent distance + path distance + action effort + jerk.

## 5. Milestones

| # | Deliverable | Exit criterion |
| --- | --- | --- |
| M0 | Harness + tests | ✅ 138 tests, 98% coverage, no external deps |
| M1 | Real-sim collection | 1k episodes written; frames visually sane |
| M2 | Loader integration | A1 confirmed; upstream reads od-MPC shards |
| M3 | Tokenizer trained | Reconstruction PSNR reported on held-out data |
| M4 | Dynamics trained | `drift_horizon` reported at 10 cm and 25 cm |
| M5 | Closed loop | Goal-image MPC success rate over ≥50 trials |

## 6. Licensing

open-dreamer's LICENSE grants **no** permission to "use, copy, modify, merge,
publish, distribute, sublicense, or sell". od-MPC is public, so:

- No open-dreamer source, config files, or snippets appear in this repository,
  including in documentation.
- Describing its API surface ("`ActionEncoder` takes a continuous channel") is
  factual interface description and is fine. Reproducing its source is not.
- Integration is by subprocess and by user-supplied checkout.

innate-os is Apache-2.0, so vendoring would be permissible with attribution.
It is still not vendored, to keep one integration pattern rather than two.

**Before every push:** confirm no upstream code has drifted in.

## 7. Open questions

1. **A1** — exact record key for continuous actions. Blocks M2.
2. **Latent normalisation** — `DynamicsModelConfig` carries `latent_mean` /
   `latent_std`. These must be computed from the MARS dataset, not inherited.
3. **Context length** — upstream defaults to 192 frames. At 10 Hz that is 19 s,
   probably longer than a navigation task needs; shorter windows are cheaper.
4. **`k_max`** — controls the shortcut ladder's resolution. Upstream's dynamics
   default and the blog's stated value differ; pick deliberately.
5. **Goal specification** — a goal *image* must be encoded by the same
   tokenizer. Where do goal images come from at evaluation time? Simplest
   answer: replay a held-out episode's final frame.
6. **Checkout/GPU availability** — M1 (real-sim collection) needs a real
   `INNATE_OS_ROOT` checkout and a machine to run MuJoCo on; resolving A1
   needs an `OPEN_DREAMER_ROOT` checkout to read `dreamer/data/transforms.py`
   from. A 2026-08-04 session had neither and no GPU, so it could not advance
   M1 or A1 and worked on closing pure-Python test coverage instead
   (`od_mpc/sim/adapter.py`'s checkout-resolution logic, 0% → 96%). Both
   remain the correct next actions once a checkout is supplied. A 2026-08-05
   session had the same blocker (still no `INNATE_OS_ROOT`,
   `OPEN_DREAMER_ROOT`, or GPU in this environment) and closed the next
   pure-Python gap: `od_mpc/data/writer.py`'s optional `array_record` backend
   (rotate/write/flush and the `array_record`-present-but-`msgpack`-missing
   fallback) was 80% covered because neither package is installed here. Tests
   inject fake `array_record`/`msgpack` modules via `sys.modules` — both are
   generic third-party libraries, not open-dreamer or innate-os, so this
   raises no licensing concern — taking the module to 100%. M1/A1 remain
   blocked pending a checkout.
7. A second 2026-08-05 session confirmed the blocker is unchanged (no
   `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
   environment) and closed the next pure-Python gap: `od_mpc/validate/metrics.py`
   was 85% covered, missing its four input-validation branches (`psnr`'s
   non-positive `data_range`, `pose_drift`'s shape/width checks,
   `drift_horizon`'s non-1-D and non-positive-tolerance checks). Added five
   tests exercising each `ValueError` path, taking the module to 100% and the
   suite to 118 tests / 95% overall coverage. M1/A1 remain blocked pending a
   checkout; `od_mpc/data/collector.py` (90%) and `od_mpc/sim/stub.py` (92%)
   are the next-lowest pure-Python gaps for a future session.
8. A 2026-08-06 session confirmed the blocker is unchanged again (no
   `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
   environment) and closed the next pure-Python gap: `od_mpc/data/collector.py`
   was 90% covered, missing `CollectorConfig`'s `control_dt`/`settle_seconds`
   validation, the `settle_seconds > 0` pre-roll branch in `collect_episode`,
   and `Episode`'s frame/action/pose shape-validation branches. Added six
   tests exercising each path, taking the module to 100% and the suite to
   124 tests / 96% overall coverage. M1/A1 remain blocked pending a checkout;
   `od_mpc/mpc/cost.py` (92%), `od_mpc/sim/stub.py` (92%), and
   `od_mpc/validate/probe.py` (91%) are the next-lowest pure-Python gaps for a
   future session.
9. A second 2026-08-06 session confirmed the blocker is unchanged yet again
   (no `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
   environment) and closed the next pure-Python gap: `od_mpc/validate/probe.py`
   was 91% covered, missing `pose_targets`' and `targets_to_pose`'s
   shape-validation branches and `fit_pose_probe`'s non-2-D-latents and
   fewer-than-two-samples branches. Added four tests exercising each
   `ValueError` path, taking the module to 100% and the suite to 128 tests
   (still 96% overall coverage, since the same source lines are now counted
   as covered). M1/A1 remain blocked pending a checkout; `od_mpc/mpc/cost.py`
   (92%) and `od_mpc/sim/stub.py` (92%) are the next-lowest pure-Python gaps
   for a future session.
10. A 2026-08-07 session confirmed the blocker is unchanged yet again (no
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment) and closed the next pure-Python gap: `od_mpc/mpc/cost.py`
    was 92% covered, missing `GoalCostConfig`'s negative-weight rejection,
    `goal_latent_cost`'s `predicted_latents`-too-few-dimensions rejection, and
    the horizon-of-one branch where there is no successive-action pair to
    charge a jerk penalty against. Added three tests exercising each path,
    taking the module to 100% and the suite to 131 tests / 97% overall
    coverage. M1/A1 remain blocked pending a checkout; `od_mpc/sim/stub.py`
    (92%) and `od_mpc/actions.py` (94%) are the next-lowest pure-Python gaps
    for a future session.
11. A second 2026-08-07 session confirmed the blocker is unchanged yet again
    (no `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment) and closed the next pure-Python gap: `od_mpc/sim/stub.py`
    was 92% covered, missing `RoomBounds`' degenerate-extent rejection,
    `StubMars.__init__`'s non-positive-`render_wh` and non-positive-`dt_substep`
    rejections, `step`'s negative-duration rejection, `lidar_scan`'s
    non-positive-`n_rays` and non-positive-`max_range` rejections, and the
    `wrist` camera's vertical-flip branch in `render_rgb`. Added seven tests
    exercising each path, taking the module to 100% and the suite to 138
    tests / 98% overall coverage. M1/A1 remain blocked pending a checkout;
    `od_mpc/actions.py` (94%) and `od_mpc/mpc/planner.py` (94%) are the
    next-lowest pure-Python gaps for a future session.
