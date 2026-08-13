"""Roll the explorer out against a simulator and record episodes.

An :class:`Episode` holds everything the downstream pipelines need: frames for
the tokenizer, normalised actions for the dynamics model, and ground-truth
poses that are never fed to the model but make the validation probe in
:mod:`od_mpc.validate` possible.

Frames are rendered at the training resolution rather than downsampled later.
``VirtualMars`` takes ``render_wh`` in its constructor, so a 64x64 collection
run costs a fraction of a 640x480 one and skips a resize step that would
otherwise sit between collection and training.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..actions import DEFAULT_LIMITS, CmdVel, CmdVelLimits, to_continuous_actions
from ..sim.protocol import MarsSim
from .explorer import ExplorerConfig, initial_state, step_explorer

__all__ = ["Episode", "CollectorConfig", "collect_episode", "collect_episodes"]


@dataclass(frozen=True)
class CollectorConfig:
    """Settings for a collection run.

    Attributes:
        steps_per_episode: Frames recorded per episode.
        control_dt: Seconds of simulation advanced per control step.
        camera: Camera name to record.
        lidar_rays: Rays per scan handed to the explorer.
        lidar_max_range: Maximum lidar range in metres.
        settle_seconds: Physics advanced after reset before recording starts,
            so the first frame is not mid-spawn.
        explorer: Exploration policy tuning.
        limits: Command limits used to normalise recorded actions.
    """

    steps_per_episode: int = 160
    control_dt: float = 0.1
    camera: str = "main"
    lidar_rays: int = 360
    lidar_max_range: float = 12.0
    settle_seconds: float = 0.5
    explorer: ExplorerConfig = ExplorerConfig()
    limits: CmdVelLimits = DEFAULT_LIMITS

    def __post_init__(self) -> None:
        if self.steps_per_episode < 2:
            raise ValueError(
                f"steps_per_episode must be at least 2, got {self.steps_per_episode}."
            )
        if self.control_dt <= 0.0:
            raise ValueError(f"control_dt must be positive, got {self.control_dt}.")
        if self.settle_seconds < 0.0:
            raise ValueError("settle_seconds must be non-negative.")


@dataclass(frozen=True)
class Episode:
    """One recorded trajectory.

    Attributes:
        frames: ``(T, H, W, 3)`` uint8 camera frames.
        actions: ``(T, 2)`` float32 commands normalised to ``[-1, 1]``.
        poses: ``(T, 3)`` float32 ground-truth ``(x, y, yaw)``.
    """

    frames: np.ndarray
    actions: np.ndarray
    poses: np.ndarray

    def __post_init__(self) -> None:
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError(f"frames must be (T, H, W, 3), got {self.frames.shape}.")
        lengths = {len(self.frames), len(self.actions), len(self.poses)}
        if len(lengths) != 1:
            raise ValueError(
                "frames, actions and poses must share a length; got "
                f"{len(self.frames)}, {len(self.actions)}, {len(self.poses)}."
            )
        if self.actions.shape[1:] != (2,):
            raise ValueError(f"actions must be (T, 2), got {self.actions.shape}.")
        if self.poses.shape[1:] != (3,):
            raise ValueError(f"poses must be (T, 3), got {self.poses.shape}.")

    @property
    def length(self) -> int:
        """Number of recorded steps."""
        return int(len(self.frames))


def collect_episode(sim: MarsSim, config: CollectorConfig) -> Episode:
    """Run one episode and return the recording.

    The action recorded at index ``t`` is the command issued *from* frame ``t``,
    so ``(frames[t], actions[t]) -> frames[t + 1]``. That is the convention the
    dynamics model is trained under, and getting it off by one is the single
    easiest way to train a model that appears to work but cannot plan.

    Args:
        sim: Any object satisfying :class:`~od_mpc.sim.protocol.MarsSim`.
        config: Collection settings.

    Returns:
        The recorded :class:`Episode`.
    """
    sim.reset()
    if config.settle_seconds > 0.0:
        sim.set_cmd_vel(0.0, 0.0)
        sim.step(config.settle_seconds)

    frames: list[np.ndarray] = []
    commands: list[CmdVel] = []
    poses: list[tuple[float, float, float]] = []

    state = initial_state()

    for _ in range(config.steps_per_episode):
        frame = np.asarray(sim.render_rgb(config.camera), dtype=np.uint8)
        frames.append(frame)
        poses.append(sim.pose())

        scan = sim.lidar_scan(config.lidar_rays, config.lidar_max_range)
        command, state = step_explorer(scan, state, config.explorer)
        commands.append(command)

        sim.set_cmd_vel(command.vx, command.wz)
        sim.step(config.control_dt)

    return Episode(
        frames=np.stack(frames).astype(np.uint8),
        actions=to_continuous_actions(commands, config.limits),
        poses=np.array(poses, dtype=np.float32),
    )


def collect_episodes(
    sim: MarsSim,
    config: CollectorConfig,
    count: int,
) -> Iterator[Episode]:
    """Yield ``count`` episodes from the same simulator instance.

    Yielding rather than returning a list keeps peak memory at one episode, so
    a long collection run streams straight into the shard writer.

    Args:
        sim: Simulator to roll out against.
        config: Collection settings.
        count: Number of episodes.

    Yields:
        One :class:`Episode` at a time.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}.")
    for _ in range(count):
        yield collect_episode(sim, config)
