"""A dependency-free stand-in for the MARS simulator.

``StubMars`` implements :class:`~od_mpc.sim.protocol.MarsSim` with unicycle
kinematics inside an axis-aligned rectangular room. It exists so the explorer,
the collector, the shard writer, and the planner can all be exercised in CI
without MuJoCo, ROS, or the innate-os checkout.

It is a test fixture, not a physics engine. It has no inertia, no collision
response beyond clamping, and its renderer draws a schematic view rather than
a photorealistic one. Do not train a released model on stub data; the frames
have none of the visual structure the tokenizer is meant to compress.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["StubMars", "RoomBounds"]

_MAX_LINEAR = 0.4
_MAX_YAW = 1.0


class RoomBounds:
    """Axis-aligned rectangular room the stub robot cannot leave."""

    def __init__(
        self,
        x_min: float = -5.0,
        x_max: float = 5.0,
        y_min: float = -5.0,
        y_max: float = 5.0,
    ) -> None:
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"Degenerate room bounds: x=({x_min}, {x_max}) y=({y_min}, {y_max})."
            )
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Clamp a position into the room."""
        return (
            min(max(x, self.x_min), self.x_max),
            min(max(y, self.y_min), self.y_max),
        )


class StubMars:
    """Deterministic unicycle robot in a box room.

    Args:
        render_wh: ``(width, height)`` of rendered frames.
        bounds: Room extents.
        spawn: Initial ``(x, y, yaw)``.
        dt_substep: Integration step used inside :meth:`step`.
    """

    def __init__(
        self,
        render_wh: tuple[int, int] = (64, 64),
        bounds: RoomBounds | None = None,
        spawn: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt_substep: float = 0.02,
    ) -> None:
        width, height = render_wh
        if width <= 0 or height <= 0:
            raise ValueError(f"render_wh must be positive, got {render_wh}.")
        if dt_substep <= 0.0:
            raise ValueError(f"dt_substep must be positive, got {dt_substep}.")

        self._width = width
        self._height = height
        self._bounds = bounds if bounds is not None else RoomBounds()
        self._spawn = spawn
        self._dt = dt_substep

        self._x, self._y, self._yaw = spawn
        self._cmd_vx = 0.0
        self._cmd_wz = 0.0

    def reset(self) -> None:
        self._x, self._y, self._yaw = self._spawn
        self._cmd_vx = 0.0
        self._cmd_wz = 0.0

    def set_cmd_vel(self, vx: float, wz: float) -> None:
        if not (math.isfinite(vx) and math.isfinite(wz)):
            raise ValueError(f"Non-finite command: vx={vx}, wz={wz}.")
        self._cmd_vx = min(max(vx, -_MAX_LINEAR), _MAX_LINEAR)
        self._cmd_wz = min(max(wz, -_MAX_YAW), _MAX_YAW)

    def step(self, duration: float) -> None:
        if duration < 0.0:
            raise ValueError(f"duration must be non-negative, got {duration}.")
        remaining = duration
        while remaining > 1e-12:
            dt = min(self._dt, remaining)
            self._yaw = _wrap_angle(self._yaw + self._cmd_wz * dt)
            nx = self._x + self._cmd_vx * math.cos(self._yaw) * dt
            ny = self._y + self._cmd_vx * math.sin(self._yaw) * dt
            self._x, self._y = self._bounds.clamp(nx, ny)
            remaining -= dt

    def pose(self) -> tuple[float, float, float]:
        return (self._x, self._y, self._yaw)

    def lidar_scan(self, n_rays: int, max_range: float) -> np.ndarray:
        """Exact ray-box intersection against the room walls."""
        if n_rays <= 0:
            raise ValueError(f"n_rays must be positive, got {n_rays}.")
        if max_range <= 0.0:
            raise ValueError(f"max_range must be positive, got {max_range}.")

        angles = self._yaw + np.arange(n_rays) * (2.0 * math.pi / n_rays)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # Slab method: distance to each of the four walls along the ray.
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(
                cos_a > 0.0,
                (self._bounds.x_max - self._x) / cos_a,
                (self._bounds.x_min - self._x) / cos_a,
            )
            ty = np.where(
                sin_a > 0.0,
                (self._bounds.y_max - self._y) / sin_a,
                (self._bounds.y_min - self._y) / sin_a,
            )

        tx = np.where(np.isfinite(tx) & (tx > 0.0), tx, np.inf)
        ty = np.where(np.isfinite(ty) & (ty > 0.0), ty, np.inf)
        distance = np.minimum(tx, ty)
        return np.minimum(distance, max_range).astype(np.float32)

    def render_rgb(self, camera: str) -> np.ndarray:
        """Draw a schematic forward view.

        The image encodes pose deterministically so that a tokenizer trained on
        stub data has *something* learnable, and so tests can assert that
        different poses produce different frames.
        """
        if camera not in ("main", "wrist"):
            raise ValueError(
                f"Unknown camera {camera!r}; StubMars provides 'main' and 'wrist'."
            )

        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        scan = self.lidar_scan(self._width, max_range=12.0)

        # Forward-facing 90-degree window, nearest wall drawn as a horizon bar.
        quarter = self._width // 4
        window = np.concatenate([scan[-quarter:], scan[:quarter]])
        window = np.interp(
            np.linspace(0, len(window) - 1, self._width),
            np.arange(len(window)),
            window,
        )

        normalized = np.clip(window / 12.0, 0.0, 1.0)
        horizon = (normalized * (self._height - 1)).astype(np.int32)

        for col in range(self._width):
            top = self._height - 1 - horizon[col]
            frame[top:, col, 0] = 200 - int(120 * normalized[col])
            frame[top:, col, 1] = 90
            frame[:top, col, 2] = 40 + int(120 * normalized[col])

        if camera == "wrist":
            frame = frame[::-1]
        return frame


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
