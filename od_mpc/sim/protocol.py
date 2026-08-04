"""The simulator surface od-MPC depends on.

Everything in this package talks to MARS through :class:`MarsSim` rather than
importing ``mars_sim_driver`` directly. Two implementations satisfy it: a
thin adapter over the real ``VirtualMars`` (:mod:`od_mpc.sim.adapter`) and a
dependency-free kinematic stand-in (:mod:`od_mpc.sim.stub`).

The protocol is deliberately a strict subset of ``VirtualMars``. It covers the
base-velocity control loop, the sensors the collector records, and reset. Arm
control (``set_joint_target`` / ``joint_targets`` on the real class) is
excluded on purpose: the first milestone is navigation only, and widening this
protocol is the explicit gate for adding manipulation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["MarsSim"]


@runtime_checkable
class MarsSim(Protocol):
    """Minimal simulator interface required to collect a navigation episode."""

    def reset(self) -> None:
        """Return the robot to its spawn pose with zero velocity."""
        ...

    def step(self, duration: float) -> None:
        """Advance physics by ``duration`` seconds, holding the last command."""
        ...

    def set_cmd_vel(self, vx: float, wz: float) -> None:
        """Command base velocity in metres per second and radians per second.

        Implementations are expected to clamp to their own limits, as the real
        simulator does.
        """
        ...

    def pose(self) -> tuple[float, float, float]:
        """Return ground-truth ``(x, y, yaw)`` in world coordinates.

        Ground truth is what makes the pose probe in :mod:`od_mpc.validate`
        possible; it is never an input to the model.
        """
        ...

    def render_rgb(self, camera: str) -> np.ndarray:
        """Render a camera to a ``(H, W, 3)`` uint8 array."""
        ...

    def lidar_scan(self, n_rays: int, max_range: float) -> np.ndarray:
        """Planar 360-degree scan, counter-clockwise from the robot's ``+x``.

        Returns distances in metres shaped ``(n_rays,)``, with ``max_range``
        where a ray hit nothing.
        """
        ...
