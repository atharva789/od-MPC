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
| M0 | Harness + tests | ✅ 151 tests, 100% line and branch coverage, no external deps |
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
12. A 2026-08-08 session confirmed the blocker is unchanged yet again (no
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment) and closed the next pure-Python gap: `od_mpc/mpc/planner.py`
    was 94% covered, missing `PlannerConfig.__post_init__`'s sub-unit-horizon,
    sub-unit-iterations, non-positive-std, and momentum-out-of-range rejection
    branches. Added five tests exercising each `ValueError` path, taking the
    module to 100% and the suite to 142 tests / 99% overall coverage. M1/A1
    remain blocked pending a checkout; `od_mpc/actions.py` (94%),
    `od_mpc/data/explorer.py` (96%), `od_mpc/sim/adapter.py` (96%), and
    `od_mpc/train/pipeline.py` (96%) are the next-lowest pure-Python gaps for
    a future session.
13. A second 2026-08-08 session confirmed the blocker is unchanged yet again
    (no `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment) and closed the next pure-Python gap: `od_mpc/actions.py` was
    94% covered, missing the `isinstance(commands, np.ndarray)` true branch in
    `to_continuous_actions` and the post-normalisation non-2-D rejection
    branches in both `to_continuous_actions` and `from_continuous_actions`
    (a bare `(2,)` array has the right trailing dimension but no time axis).
    Added three tests exercising each path, taking the module to 100% and the
    suite to 145 tests (still 99% overall coverage, since the same source
    lines are now counted as covered). M1/A1 remain blocked pending a
    checkout; `od_mpc/data/explorer.py` (96%), `od_mpc/sim/adapter.py` (96%),
    and `od_mpc/train/pipeline.py` (96%) are the next-lowest pure-Python gaps
    for a future session.
14. A 2026-08-09 session confirmed the blocker is unchanged yet again (no
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment) and closed two of the three remaining pure-Python gaps.
    `od_mpc/train/pipeline.py` was 96% covered, missing `run_stage`'s
    `dry_run=False` branch (the actual `subprocess.run` call and its return);
    added tests that execute it against a fake checkout's empty stub scripts
    (success) and a script that exits non-zero (`CalledProcessError`),
    taking the module to 100%. `od_mpc/data/explorer.py` was 96% covered,
    missing three lines; two (`ExplorerConfig`'s non-positive-speed-ceiling
    and sub-one-step-count rejections) were straightforward reachable gaps
    and got tests, taking the module to 99%. The third, `_front_clearance`'s
    "no ray in the front arc" fallback (line 101), is **unreachable dead
    code**: `_bearings` always maps index 0 to bearing exactly `0.0`, and
    `ExplorerConfig.__post_init__` requires `front_arc_deg > 0`, so
    `abs(bearings[0]) <= half_arc` is true for every validly-constructed
    config regardless of scan content or ray count. No test was added for it
    rather than fake an unreachable state through a non-validated config.
    Suite is now 149 tests / 99% overall coverage (635 statements, 2 missed:
    `explorer.py:101` above and `sim/adapter.py:93`, the `return
    core.VirtualMars` success line, which needs either a real checkout or an
    injected fake `mars_sim_driver` package — left for a future session to
    weigh against the adapter's real-checkout caution in this file's
    guidance). M1/A1 remain blocked pending a checkout; `od_mpc/sim/adapter.py`
    (96%) is the only remaining pure-Python gap for a future session.
15. A third 2026-08-09 session confirmed the blocker is unchanged yet again
    (no `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment) and closed the last remaining pure-Python gap flagged by the
    prior session: `od_mpc/sim/adapter.py:93` (`return core.VirtualMars`, the
    success path of `load_virtual_mars`). Added a test that writes a
    stand-in `mars_sim_driver/core.py` to a `tmp_path` checkout — authored
    entirely by the test, not copied from innate-os — exposing a class named
    `VirtualMars` with no members, and asserts `load_virtual_mars` returns
    exactly that class object. This checks only the adapter's own
    import-and-return plumbing; it asserts nothing about the real
    `VirtualMars` API beyond what §2.1 already records as verified from
    source (that `mars_sim_driver.core` exports a `VirtualMars`), so it does
    not conflict with the "no behavioural changes without a real checkout"
    caution. Suite is now 150 tests / 99% overall coverage (635 statements,
    1 missed: `explorer.py:101`, the confirmed-unreachable dead code noted
    by the prior session). M1/A1 remain blocked pending a checkout. Every
    module under `od_mpc/` is now at 100% pure-Python coverage except that
    one unreachable line, so the coverage sweep this and prior sessions have
    run is effectively exhausted. A future session with still no
    checkout/GPU should either delete the confirmed-unreachable
    `explorer.py:101` branch (no reachable test can cover it as written) or
    look for genuinely new logic worth testing rather than continuing this
    sweep — the next real leverage on M1/A1 is still an `INNATE_OS_ROOT` /
    `OPEN_DREAMER_ROOT` checkout, not more coverage.
16. A 2026-08-10 session confirmed the blocker is unchanged yet again (no
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment; also re-confirmed there are no new source files under
    `od_mpc/` beyond the 18 modules already at 100%, so "genuinely new logic
    worth testing" was not available either). Took the prior session's other
    offered option: re-derived that `explorer.py:101` is unreachable
    (`_bearings` maps ray 0 to bearing exactly `0.0` for every `n_rays >= 1`,
    verified numerically, and `ExplorerConfig.__post_init__` requires
    `front_arc_deg > 0`, so the forward arc always contains at least ray 0)
    and deleted the dead `if not np.any(in_front)` branch and its fallback
    return, replacing it with a code comment recording why the remaining
    line is always safe. This is a pure deletion, not new logic beyond the
    milestone, and touches only `od_mpc/data/explorer.py`, not
    `sim/adapter.py`. Suite is now 150 tests / **100%** overall coverage
    (633 statements, 0 missed), with no coverage exclusions anywhere in the
    repo. Corrected the two current-facing "99% coverage" claims (README
    status line, SPEC's M0 row) to 100%; left this section's historical
    per-session entries as accurate records of what coverage stood at when
    each was written. M1/A1 remain blocked pending a checkout. With the
    coverage sweep now truly exhausted — no unreachable lines, no untested
    modules, no coverage exclusions left to close — a future session with
    still no checkout/GPU has no more pure-Python test-coverage gaps to
    close in this codebase; it should look for the next §3/§7 item that
    doesn't require `INNATE_OS_ROOT`/`OPEN_DREAMER_ROOT`/a GPU, or, if none
    exists, report the blocker plainly rather than inventing busywork.
17. A second 2026-08-10 session confirmed the blocker is unchanged yet again
    (no `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment). The prior session's line-coverage sweep was exhausted, but
    line coverage does not imply branch coverage: `pytest --cov-branch`
    showed `od_mpc/sim/adapter.py` at 97%, missing branch `81->84` —
    `load_virtual_mars`'s `if package_dir not in sys.path` had only ever been
    exercised on its True side (path not yet present), never its False side
    (a second call with the path already inserted, which should skip the
    insert rather than duplicate the entry). Added
    `test_load_virtual_mars_does_not_duplicate_sys_path_entry`, which
    pre-seeds `sys.path` with the driver directory before calling
    `load_virtual_mars` and asserts the entry is not duplicated — pure
    Python logic, no checkout needed, same fake-`mars_sim_driver`-module
    pattern as item 15. Suite is now 151 tests / 100% line coverage *and*
    100% branch coverage (633 statements, 180 branches, 0 missed either
    way). `pyproject.toml`'s pytest config does not turn on `--cov-branch`
    by default, so this gap was invisible to the sweep every session from
    2026-08-04 through the first 2026-08-10 session ran — worth remembering
    that "100% coverage" claims in this doc mean line coverage unless stated
    otherwise. M1/A1 remain blocked pending a checkout. With branch coverage
    now also exhausted, a future session with still no checkout/GPU has no
    further coverage gaps (line or branch) to close in this codebase and
    should look for the next §3/§7 item that doesn't require
    `INNATE_OS_ROOT`/`OPEN_DREAMER_ROOT`/a GPU, or report the blocker
    plainly if none exists.
18. A 2026-08-11 session confirmed the blocker is unchanged yet again (no
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment) and re-verified, independently, that the prior sessions'
    coverage sweep is still exhausted: `pytest --cov --cov-branch` shows all
    18 `od_mpc/` source files at 100% line and 100% branch coverage (1378
    statements, 190 branches, 0 missed either way), and no new source files
    exist under `od_mpc/` beyond those already covered. With no coverage gap
    to close and every other open §3/§7 item requiring either a checkout/GPU
    (A1, A2, A3, A4, latent normalisation) or being scoped to a later
    milestone (context length and `k_max` are M3/M4 decisions; goal-image
    sourcing is an M5 decision) and therefore out of bounds for this
    session's milestone (M1), the only real remaining gap was doc drift:
    §5's M0 row still read "150 tests, 100% coverage", left behind when the
    second 2026-08-10 session's branch-coverage test took the suite to 151
    tests without updating this table (only the narrative entry above and
    the README status line were corrected at the time). Fixed the M0 row to
    "151 tests, 100% line and branch coverage, no external deps" to match
    README and the verified test output. No code changed. M1/A1 remain
    blocked pending a checkout; there is no further pure-Python action
    available in this codebase until one of `INNATE_OS_ROOT`,
    `OPEN_DREAMER_ROOT`, or a GPU becomes available.
19. A 2026-08-12 session confirmed the blocker is unchanged yet again (no
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment, and no directory matching either checkout name exists
    anywhere on disk) and independently re-verified the prior session's
    exhaustion claim rather than taking it on faith: `pytest -q` gives 151
    passed, and `pytest -q --cov --cov-branch` shows all 18 `od_mpc/` source
    files at 100% line and 100% branch coverage (1378 statements, 190
    branches, 0 missed either way), matching both the README status line and
    this file's M0 row. No new source files exist under `od_mpc/`. Every
    remaining open item (A1, A2, A3, A4, latent normalisation) needs a
    checkout or GPU this environment does not have; context length and
    `k_max` are M3/M4 decisions and goal-image sourcing is an M5 decision,
    all out of bounds for the current M1 milestone. Made no code change,
    since inventing one would violate the "don't add features beyond the
    milestone" constraint with nothing left to test. M1/A1 remain blocked
    pending a checkout; a future session should check first whether an
    `INNATE_OS_ROOT` or `OPEN_DREAMER_ROOT` checkout has become available
    before re-running this same verification sweep.
20. A 2026-08-13 session confirmed the blocker is unchanged yet again (no
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment, and no directory matching either checkout name exists
    anywhere on disk) and independently re-verified: `pytest -q` gives 151
    passed, and `pytest -q --cov=od_mpc --cov-branch --cov-report=term-missing`
    shows all 18 `od_mpc/` source files at 100% line and 100% branch coverage
    (633 statements, 180 branches, 0 missed either way) — matching item 17's
    figures exactly. Items 18 and 19 both recorded "1378 statements, 190
    branches" for this same fully-covered state; that number does not match
    item 17's measurement one session earlier, this session's fresh
    measurement, or the 635-statement figures items 14-15 recorded before the
    branch-coverage sweep added a couple of statements. Likely those two
    sessions ran coverage unscoped (picking up test files too) or otherwise
    mis-totaled; noted here rather than editing their entries, since the
    qualitative claim they made (100% line and branch, 0 missed) was still
    correct. No new source files exist under `od_mpc/`. Also checked a lead
    no prior session had tried: `ruff check .`, `black --check .`, and
    `isort --check .` all fail (15 ruff findings, 10 files black would
    reformat, 3 files isort would resort) — real drift between the dev
    tooling `pyproject.toml` declares and the repo's actual state. Left
    unfixed this session: several ruff findings are not pure formatting
    (`B008` flags the mutable-default-argument pattern on `mpc/cost.py`'s
    `GoalCostConfig()` default; `UP035` wants `typing.Iterator` /
    `typing.Callable` rewritten to `collections.abc`), and an automated
    session with no human to review the diff should not blanket-`--fix`
    findings that touch signatures rather than whitespace. Recorded as a
    legitimate, not-yet-attempted lead for a future session to work through
    deliberately, one finding at a time, rather than a mechanical sweep.
    M1/A1 remain blocked pending a checkout. This is ten consecutive daily
    sessions (2026-08-04 through 2026-08-13, several days with two) that have
    confirmed the same checkout/GPU blocker; the milestone has not moved
    since M0 closed out around 2026-08-10/11. Only a human supplying
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, or GPU access unblocks M1 or A1
    from here — the lint/format drift above is a legitimate but minor lead,
    not a substitute for that.
21. A second 2026-08-13 session confirmed the blocker is unchanged yet again
    (no `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment) and worked through the lint/format drift the prior session
    left as a lead, one finding at a time rather than a blanket `--fix`.
    All 15 `ruff check` findings: import-sort (`I001`) and `typing.Iterator`
    / `typing.Callable` → `collections.abc` (`UP035`) were mechanical, safe
    under `from __future__ import annotations` (both modules already defer
    annotation evaluation, so the rewrite is a no-op at runtime); the quoted
    forward reference in `writer.py` (`UP037`) was equally safe to unquote
    for the same reason. `tests/test_actions.py`'s `zip()` without `strict=`
    (`B905`) got `strict=True`, since the surrounding test already asserts
    the two sequences are the same length — a silent truncation there would
    hide a real bug. The nine `E501` long lines were manual re-wraps, no
    semantic change. `mpc/cost.py`'s `B008` (`GoalCostConfig()` called in an
    argument default) was the one finding that touches a signature: rather
    than either leave it or change the parameter to `GoalCostConfig | None`
    (a public-API type change this session had no reviewer to sign off on),
    it was resolved by hoisting the same call to a module-level
    `_DEFAULT_GOAL_COST_CONFIG` singleton — behaviourally identical, since
    Python already evaluates a function's default expression once at def
    time and `GoalCostConfig` is a frozen (immutable) dataclass, so there
    was never an aliasing bug, only a lint false-positive. `black` and
    `isort` were then run for real (not just `--check`): 5 files reformatted
    by `black`, all pure whitespace/line-wrapping with no AST change: `git
    diff --ignore-space-at-eol` confirms no non-whitespace edit landed
    outside the deliberate changes above. `ruff check .`, `black --check .`,
    and `isort --check .` all now pass clean. `pytest -q` gives 151 passed
    (unchanged count — no test logic changed) and `pytest -q --cov=od_mpc
    --cov-branch --cov-report=term-missing` still shows 100% line and 100%
    branch coverage (635 statements, 180 branches, 0 missed), confirming the
    cleanup was behaviour-neutral. M1/A1 remain blocked pending a checkout.
    With the lint/format lead now also closed, this repository has no
    outstanding pure-Python or tooling-hygiene gap left for a checkout-less,
    GPU-less session to work: the only lever that moves M1 or A1 from here
    is a human supplying `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, or GPU
    access. A future session in this same environment should check for
    those first, and if still absent, should say so rather than search for
    another manufactured task.
22. A 2026-08-14 session confirmed the blocker is unchanged yet again (no
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax` in this
    environment, and no directory matching either checkout name exists
    anywhere on disk). Independently re-verified rather than taking prior
    entries on faith: `pytest -q` gives 151 passed; `pytest -q --cov=od_mpc
    --cov-branch --cov-report=term-missing` shows all 18 `od_mpc/` source
    files at 100% line and 100% branch coverage (635 statements, 180
    branches, 0 missed); `ruff check .`, `black --check .`, and
    `isort --check .` all pass clean. `find od_mpc -name "*.py"` lists the
    same 18 files as every prior sweep — no new source to test. Every
    remaining open item (A1, A2, A3, A4, latent normalisation) still needs a
    checkout or GPU this environment does not have; context length and
    `k_max` are M3/M4 decisions and goal-image sourcing is an M5 decision,
    out of bounds for the current M1 milestone. Made no code change: with
    coverage and lint both already exhausted, there was nothing left to
    close without inventing work beyond the milestone. This is the twelfth
    session (2026-08-04 through 2026-08-14) to confirm the identical
    checkout/GPU blocker; nothing has changed since M0 closed. A future
    session should keep checking for `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`,
    or GPU access before re-running this sweep, and should feel free to stop
    re-confirming it in exhaustive prose once the pattern is this
    established — a short "still blocked, still nothing new" is enough.
23. A second 2026-08-14 session: still blocked, still nothing new. No
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax`; 151 tests
    pass; 100% line/branch coverage across the same 18 `od_mpc/` modules;
    `ruff`/`black`/`isort` all clean; README/SPEC status lines still match
    reality. No code change — thirteenth consecutive session confirming the
    identical checkout/GPU blocker.
24. A 2026-08-15 session: still blocked, still nothing new. No
    `INNATE_OS_ROOT`, `OPEN_DREAMER_ROOT`, `nvidia-smi`, or `jax`; 151 tests
    pass; 100% line/branch coverage across the same 18 `od_mpc/` modules;
    `ruff`/`black`/`isort` all clean; README/SPEC status lines still match
    reality. No code change — fourteenth consecutive session confirming the
    identical checkout/GPU blocker.
