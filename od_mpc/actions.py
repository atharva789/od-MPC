"""Adapters between MARS base-velocity commands and continuous action tensors.

MARS exposes a two-dimensional continuous command through the simulator's
``set_cmd_vel(vx, wz)``: forward velocity in metres per second and yaw rate in
radians per second, each clamped by the simulator to its own limit.

open-dreamer's ``ActionEncoder`` accepts a continuous channel shaped
``(B, T, continuous_action_dim)`` and passes it through a single linear
projection. Every dataset config shipped in that repository sets
``continuous_action_dim: 0``, so the channel exists but is never trained; the
only continuous producer in the codebase emits a bounded two-vector.

A MARS command is therefore a near drop-in for that channel once it is
normalised to ``[-1, 1]``. This module owns that normalisation and nothing
else, so the mapping is testable without a simulator or a trained model.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "CmdVel",
    "CmdVelLimits",
    "DEFAULT_LIMITS",
    "normalize_cmd_vel",
    "denormalize_cmd_vel",
    "to_continuous_actions",
    "from_continuous_actions",
]


class CmdVel(NamedTuple):
    """A single base-velocity command in physical units."""

    vx: float
    """Forward velocity, metres per second."""

    wz: float
    """Yaw rate, radians per second."""


class CmdVelLimits(NamedTuple):
    """Symmetric saturation limits for a base-velocity command.

    Defaults mirror ``MAX_LINEAR`` and ``MAX_YAW`` in innate-os
    (``mars_sim_driver/world.py``) as of the pinned revision recorded in
    ``docs/SPEC.md``. Read them from your own install rather than trusting
    these if you depend on exact saturation behaviour.
    """

    max_linear: float = 0.4
    """Metres per second."""

    max_yaw: float = 1.0
    """Radians per second."""


DEFAULT_LIMITS = CmdVelLimits()

_ACTION_DIM = 2


def _validate_limits(limits: CmdVelLimits) -> None:
    if limits.max_linear <= 0.0 or limits.max_yaw <= 0.0:
        raise ValueError(
            f"CmdVelLimits must be strictly positive, got {limits!r}. "
            "Zero or negative limits would make normalisation undefined."
        )


def _as_pairs(values: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a float64 copy shaped ``(..., 2)``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != _ACTION_DIM:
        raise ValueError(
            f"{name} must have trailing dimension {_ACTION_DIM} (vx, wz), "
            f"got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains non-finite values; a NaN command would "
            "silently poison every downstream latent."
        )
    return array


def normalize_cmd_vel(
    commands: np.ndarray,
    limits: CmdVelLimits = DEFAULT_LIMITS,
) -> np.ndarray:
    """Map physical ``(vx, wz)`` commands onto ``[-1, 1]``.

    Values beyond the limits are clipped, matching the simulator, which clamps
    the command itself before applying it. Clipping here keeps the recorded
    action equal to the action the robot actually executed.

    Args:
        commands: Array shaped ``(..., 2)`` of ``(vx, wz)`` in physical units.
        limits: Saturation limits to normalise against.

    Returns:
        A new float32 array shaped like ``commands``, bounded to ``[-1, 1]``.
    """
    _validate_limits(limits)
    array = _as_pairs(commands, "commands")
    scale = np.array([limits.max_linear, limits.max_yaw], dtype=np.float64)
    return np.clip(array / scale, -1.0, 1.0).astype(np.float32)


def denormalize_cmd_vel(
    normalized: np.ndarray,
    limits: CmdVelLimits = DEFAULT_LIMITS,
) -> np.ndarray:
    """Invert :func:`normalize_cmd_vel`, returning physical units.

    Args:
        normalized: Array shaped ``(..., 2)``, expected within ``[-1, 1]``.
        limits: Saturation limits the values were normalised against.

    Returns:
        A new float64 array of ``(vx, wz)`` in physical units.
    """
    _validate_limits(limits)
    array = _as_pairs(normalized, "normalized")
    scale = np.array([limits.max_linear, limits.max_yaw], dtype=np.float64)
    return np.clip(array, -1.0, 1.0) * scale


def to_continuous_actions(
    commands: list[CmdVel] | np.ndarray,
    limits: CmdVelLimits = DEFAULT_LIMITS,
) -> np.ndarray:
    """Build a ``(T, 2)`` float32 action array from a command sequence.

    This is the array that belongs in the ``continuous`` field of an
    open-dreamer ``Actions`` pytree, with ``continuous_action_dim = 2``.

    Args:
        commands: Sequence of :class:`CmdVel`, or an array shaped ``(T, 2)``.
        limits: Saturation limits to normalise against.

    Returns:
        Normalised actions shaped ``(T, 2)``, dtype float32.
    """
    if isinstance(commands, np.ndarray):
        raw = commands
    else:
        if not commands:
            raise ValueError("commands is empty; an episode needs at least one step.")
        raw = np.array([(c.vx, c.wz) for c in commands], dtype=np.float64)

    normalized = normalize_cmd_vel(raw, limits)
    if normalized.ndim != 2:
        raise ValueError(
            f"Expected a (T, 2) command sequence, got shape {normalized.shape}."
        )
    return normalized


def from_continuous_actions(
    actions: np.ndarray,
    limits: CmdVelLimits = DEFAULT_LIMITS,
) -> list[CmdVel]:
    """Decode a ``(T, 2)`` normalised action array back into commands.

    Used when a planner proposes actions in normalised space and they must be
    handed to the simulator in physical units.

    Args:
        actions: Normalised actions shaped ``(T, 2)``.
        limits: Saturation limits to denormalise against.

    Returns:
        A list of ``T`` :class:`CmdVel` in physical units.
    """
    physical = denormalize_cmd_vel(actions, limits)
    if physical.ndim != 2:
        raise ValueError(f"Expected a (T, 2) action array, got shape {physical.shape}.")
    return [CmdVel(vx=float(row[0]), wz=float(row[1])) for row in physical]
