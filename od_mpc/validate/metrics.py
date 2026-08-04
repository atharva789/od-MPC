"""Metrics for judging a rollout.

Three questions, three metrics:

* Does the tokenizer preserve the image? :func:`psnr`.
* Does the rollout stay in the right place? :func:`pose_drift`.
* Does it stay there for long enough to plan with? :func:`drift_horizon`.

The last one is the number that matters for MPC. A planner searching over a
twenty-step horizon needs the model to be trustworthy across twenty steps; the
mean error over a rollout can look fine while the tail is already useless.
"""

from __future__ import annotations

import numpy as np

__all__ = ["wrap_angle", "psnr", "pose_drift", "drift_horizon"]


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angles to ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def psnr(prediction: np.ndarray, target: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio between two image arrays.

    Args:
        prediction: Predicted frames, any shape.
        target: Ground-truth frames, matching shape.
        data_range: Maximum representable value; 255 for uint8 frames.

    Returns:
        PSNR in decibels, or ``inf`` for an exact match.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {true.shape}.")
    if data_range <= 0.0:
        raise ValueError(f"data_range must be positive, got {data_range}.")

    mse = float(np.mean((pred - true) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def pose_drift(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step position and heading error along a rollout.

    Args:
        predicted: ``(T, 3)`` probe-decoded poses.
        actual: ``(T, 3)`` ground-truth poses.

    Returns:
        ``(position_error, heading_error)``, each ``(T,)``. Position error is
        Euclidean in metres; heading error is the absolute wrapped difference
        in radians.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    true = np.asarray(actual, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {true.shape}.")
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"Poses must be (T, 3), got {pred.shape}.")

    position_error = np.linalg.norm(pred[:, :2] - true[:, :2], axis=1)
    heading_error = np.abs(wrap_angle(pred[:, 2] - true[:, 2]))
    return position_error, heading_error


def drift_horizon(position_error: np.ndarray, tolerance: float) -> int:
    """Number of leading steps whose position error stays within tolerance.

    Reported as the headline number for a trained model: "latent rollouts stay
    within ``tolerance`` metres for N steps."

    Args:
        position_error: ``(T,)`` per-step position error.
        tolerance: Allowed error in metres.

    Returns:
        Count of leading steps under tolerance; ``0`` if the first already
        exceeds it.
    """
    errors = np.asarray(position_error, dtype=np.float64)
    if errors.ndim != 1:
        raise ValueError(f"position_error must be 1-D, got {errors.shape}.")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}.")

    exceeded = np.flatnonzero(errors > tolerance)
    return int(exceeded[0]) if exceeded.size else int(errors.size)
