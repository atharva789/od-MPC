"""A scripted exploration policy for collecting navigation trajectories.

Why this exists rather than sampling random commands: the open-dreamer write-up
reports that their first world model was trained on random actions and the
result was unusable, which they fixed by training a simple agent to collect
trajectories instead. Random ``cmd_vel`` on a mobile base is worse than random
actions in a platformer, because a robot that spins in place or presses into a
wall produces long runs of nearly identical frames. A world model trained on
that learns that nothing ever happens.

This policy is deliberately simple and has no learned parameters. It steers
toward whichever direction has the most free space, with hysteresis so it
commits to a heading instead of dithering, and it reverses out of corners. The
goal is coverage of the state space, not competence at any task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..actions import CmdVel

__all__ = ["ExplorerConfig", "ExplorerState", "initial_state", "step_explorer"]


@dataclass(frozen=True)
class ExplorerConfig:
    """Tuning for the reactive explorer.

    Attributes:
        max_linear: Forward speed ceiling, metres per second.
        max_yaw: Yaw rate ceiling, radians per second.
        clearance_stop: Front clearance below which the robot stops advancing.
        clearance_slow: Front clearance below which forward speed tapers.
        front_arc_deg: Width of the arc treated as "in front".
        commit_steps: Steps a chosen heading is held before reconsidering.
        reverse_steps: Steps spent backing up after a corner is detected.
        turn_gain: Proportional gain from heading error to yaw rate.
    """

    max_linear: float = 0.4
    max_yaw: float = 1.0
    clearance_stop: float = 0.45
    clearance_slow: float = 1.2
    front_arc_deg: float = 60.0
    commit_steps: int = 12
    reverse_steps: int = 8
    turn_gain: float = 1.5

    def __post_init__(self) -> None:
        if self.max_linear <= 0.0 or self.max_yaw <= 0.0:
            raise ValueError("Speed ceilings must be positive.")
        if self.clearance_stop >= self.clearance_slow:
            raise ValueError(
                "clearance_stop must be below clearance_slow, got "
                f"{self.clearance_stop} >= {self.clearance_slow}."
            )
        if not 0.0 < self.front_arc_deg <= 180.0:
            raise ValueError(
                f"front_arc_deg must be in (0, 180], got {self.front_arc_deg}."
            )
        if self.commit_steps < 1 or self.reverse_steps < 1:
            raise ValueError("Step counts must be at least 1.")


@dataclass(frozen=True)
class ExplorerState:
    """Immutable explorer state carried between steps.

    Attributes:
        target_bearing: Chosen heading relative to the robot, radians.
        commit_remaining: Steps left before the heading is reconsidered.
        reverse_remaining: Steps left in a reversing manoeuvre.
    """

    target_bearing: float = 0.0
    commit_remaining: int = 0
    reverse_remaining: int = 0


def initial_state() -> ExplorerState:
    """Return the state an episode should start from."""
    return ExplorerState()


def _bearings(n_rays: int) -> np.ndarray:
    """Ray bearings relative to the robot, wrapped to ``[-pi, pi)``."""
    raw = np.arange(n_rays) * (2.0 * math.pi / n_rays)
    return (raw + math.pi) % (2.0 * math.pi) - math.pi


def _front_clearance(scan: np.ndarray, config: ExplorerConfig) -> float:
    """Minimum distance within the forward arc."""
    bearings = _bearings(len(scan))
    half_arc = math.radians(config.front_arc_deg) / 2.0
    in_front = np.abs(bearings) <= half_arc
    if not np.any(in_front):
        return float(np.min(scan))
    return float(np.min(scan[in_front]))


def _choose_bearing(scan: np.ndarray) -> float:
    """Pick the bearing of the most open direction.

    Smoothing the scan first stops the policy from aiming at a single lucky ray
    poking through a gap it cannot fit through.
    """
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(scan, kernel, mode="same")
    return float(_bearings(len(scan))[int(np.argmax(smoothed))])


def step_explorer(
    scan: np.ndarray,
    state: ExplorerState,
    config: ExplorerConfig,
) -> tuple[CmdVel, ExplorerState]:
    """Choose the next command from a lidar scan.

    Pure: the same inputs always give the same output, and ``state`` is never
    mutated. The caller threads the returned state into the next call.

    Args:
        scan: Distances in metres, counter-clockwise from the robot's ``+x``.
        state: State returned by the previous call, or :func:`initial_state`.
        config: Policy tuning.

    Returns:
        The command to execute and the state for the next step.

    Raises:
        ValueError: If the scan is empty or contains non-finite values.
    """
    scan = np.asarray(scan, dtype=np.float64)
    if scan.ndim != 1 or scan.size == 0:
        raise ValueError(f"scan must be a non-empty 1-D array, got {scan.shape}.")
    if not np.all(np.isfinite(scan)):
        raise ValueError("scan contains non-finite values.")

    if state.reverse_remaining > 0:
        command = CmdVel(vx=-0.5 * config.max_linear, wz=0.6 * config.max_yaw)
        return command, replace(state, reverse_remaining=state.reverse_remaining - 1)

    clearance = _front_clearance(scan, config)

    # Boxed in: nothing ahead and nothing worth turning toward.
    if clearance < config.clearance_stop and float(np.max(scan)) < config.clearance_slow:
        command = CmdVel(vx=-0.5 * config.max_linear, wz=0.6 * config.max_yaw)
        return command, replace(state, reverse_remaining=config.reverse_steps)

    if state.commit_remaining > 0:
        bearing = state.target_bearing
        next_state = replace(state, commit_remaining=state.commit_remaining - 1)
    else:
        bearing = _choose_bearing(scan)
        next_state = ExplorerState(
            target_bearing=bearing,
            commit_remaining=config.commit_steps,
            reverse_remaining=0,
        )

    yaw = float(np.clip(config.turn_gain * bearing, -config.max_yaw, config.max_yaw))

    # Slow down near obstacles and while turning hard; a robot that turns at
    # full speed produces motion blur the tokenizer cannot reconstruct.
    clearance_factor = float(
        np.clip(
            (clearance - config.clearance_stop)
            / (config.clearance_slow - config.clearance_stop),
            0.0,
            1.0,
        )
    )
    turn_factor = float(np.clip(1.0 - abs(bearing) / math.pi, 0.0, 1.0))
    forward = config.max_linear * clearance_factor * turn_factor

    return CmdVel(vx=forward, wz=yaw), next_state
