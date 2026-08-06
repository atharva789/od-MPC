# od-MPC

Latent model-predictive control for Innate's MARS robot, planning through
[open-dreamer](https://next-state.github.io/open-dreamer/)'s world model.

Collect navigation trajectories in the MARS simulator, train a world model on
them, then plan action sequences by rolling candidate commands forward in
latent space and picking the one that lands nearest a goal image.

> **Status: scaffolding.** The data, validation, and planning logic is written
> and tested (124 tests, 96% coverage). Nothing has been trained yet. Every
> claim below about *what the code does* is true; there are no claims yet about
> what a trained model achieves. See [Roadmap](#roadmap).

---

## Why this is tractable

open-dreamer's `ActionEncoder` has three action channels: binary, categorical,
and continuous. Every dataset config in that repository sets
`continuous_action_dim: 0`, so the continuous channel exists but has never been
trained. The only continuous producer in the codebase emits a **bounded
two-vector**.

MARS's simulator exposes base control as `set_cmd_vel(vx, wz)`: forward
velocity clamped to ±0.4 m/s, yaw rate clamped to ±1.0 rad/s. That is also a
**bounded two-vector**.

So the action-space port is a normalisation, not a redesign. That is the whole
premise of this repository, and it lives in
[`od_mpc/actions.py`](od_mpc/actions.py).

## Pipelines

### 1. Data annotation — `od_mpc.data`

Rolls a scripted explorer against the simulator and records
`(frames, actions, poses)` episodes.

The explorer is **not** random. The open-dreamer write-up reports that their
first world model was trained on random actions and was unusable, fixed by
training a simple agent to collect trajectories instead. Random `cmd_vel` on a
mobile base is worse than random actions in a platformer: a robot that spins in
place or presses into a wall emits long runs of near-identical frames, and a
world model trained on that learns nothing ever happens. So
[`explorer.py`](od_mpc/data/explorer.py) steers toward open space with
hysteresis and reverses out of corners — no learned parameters, just coverage.

Frames render at training resolution directly (`VirtualMars` takes `render_wh`),
so a 64×64 run is far cheaper than 640×480 and skips a resize step.

### 2. Training — `od_mpc.train`

A **wrapper**, not a reimplementation. It renders config overrides, invokes
open-dreamer's `train_tokenizer.py` / `train_dynamics.py` as a subprocess, and
reports where the checkpoint landed. `run_stage` defaults to `dry_run=True`, so
launching GPU hours is always explicit.

### 3. Validation — `od_mpc.validate`

The metric this project is built around: **do the latents know where the robot
is?**

Nothing in the training objective forces them to. So we fit a ridge-regression
probe from frozen latent to ground-truth pose on held-out data, then measure how
fast probe error grows along an open-loop rollout. A model whose rollouts look
plausible while pose error diverges is producing video, not a world model, and
cannot be planned through. Linear probing is the right tool precisely because a
linear map cannot manufacture structure that is not already there.

Yaw is regressed as `(cos, sin)`, not as an angle — regressing the angle makes
the ±π wrap look like a huge error and teaches the probe to predict the circular
mean.

Headline number: `drift_horizon` — how many steps rollouts stay within *n* cm.

### 4. Planning — `od_mpc.mpc`

Cross-entropy method over action sequences, scored by distance to a goal latent.

CEM rather than backprop-through-the-model: the dynamics model is a few-step
denoiser, so differentiating a 20-step rollout means differentiating 20
denoising ladders. Sampling sidesteps that and parallelises across candidates.

The planner takes a `rollout_cost` callable and knows nothing about what it
plans through, which is why the search is fully tested without JAX or a GPU.

## Install

```bash
git clone https://github.com/atharva789/od-MPC && cd od-MPC
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The test suite runs with **no simulator, no JAX, and no checkout** — a
kinematic stand-in ([`StubMars`](od_mpc/sim/stub.py)) satisfies the same
protocol as the real simulator.

For real collection and training, supply both checkouts:

```bash
export INNATE_OS_ROOT=/path/to/innate-os
export OPEN_DREAMER_ROOT=/path/to/open-dreamer
```

## Licensing

od-MPC is MIT. It **vendors neither dependency**, and that is deliberate:

| Project | License | How it is used |
| --- | --- | --- |
| [open-dreamer](https://github.com/next-state/open-dreamer) | **All Rights Reserved** | User-supplied checkout, invoked as a subprocess. No code copied. |
| [innate-os](https://github.com/innate-inc/innate-os) | Apache-2.0 | User-supplied checkout, imported at runtime. No code copied. |

open-dreamer's license grants no permission to copy, modify, or redistribute,
so no part of it appears here — not source, not config files, not snippets in
this README. od-MPC describes its API surface and calls it. That is also better
engineering: upstream stays upstream.

## Roadmap

- [x] Action-space adapter, explorer, collector, shard writer
- [x] Pose probe, drift metrics, CEM planner, cost functions
- [x] Test suite green without external dependencies
- [ ] Wire the real `VirtualMars` adapter and collect a first dataset
- [ ] Confirm the loader's continuous-action key against a pinned open-dreamer revision
- [ ] Train the tokenizer; report reconstruction PSNR
- [ ] Train dynamics; report `drift_horizon`
- [ ] Close the loop: goal-image MPC, report success rate
- [ ] Stretch: arm control (`set_joint_target`) for manipulation

## Layout

```
od_mpc/
  actions.py        cmd_vel <-> continuous-action normalisation
  sim/              MarsSim protocol, kinematic stub, real-sim adapter
  data/             explorer, collector, shard writer
  train/            open-dreamer subprocess wrapper
  validate/         pose probe, drift metrics
  mpc/              CEM planner, goal costs
docs/SPEC.md        design decisions, verified facts, open questions
```

## Credit

The world model is [open-dreamer](https://next-state.github.io/open-dreamer/)
by Diego Marti Monso, Francesco Sacco, and Edward Hu. The robot and simulator
are [MARS](https://docs.innate.bot/) by Innate. This repository is only the
harness between them.
