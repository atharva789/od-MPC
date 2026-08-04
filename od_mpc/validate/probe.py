"""A linear probe from latents to ground-truth pose.

The world model is trained only to predict future latents; nothing forces those
latents to encode where the robot is. The probe tests whether they do anyway,
and it is the measurement this project is actually built around.

Method: fit ridge regression from a frozen latent to pose on a held-out split,
then measure how fast probe error grows along an open-loop rollout. A model
whose rollouts stay visually plausible while pose error diverges is producing
video, not a world model, and would be useless for planning. Linear probing is
the standard tool here precisely because a linear map cannot manufacture
structure that is not already in the representation.

Yaw is regressed as ``(cos, sin)`` rather than an angle. Regressing the angle
directly makes the wrap from ``+pi`` to ``-pi`` look like an enormous error and
teaches the probe to predict the circular mean, which is near zero everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PoseProbe", "fit_pose_probe", "pose_targets", "targets_to_pose"]

_TARGET_DIM = 4


def pose_targets(poses: np.ndarray) -> np.ndarray:
    """Encode ``(x, y, yaw)`` poses as ``(x, y, cos yaw, sin yaw)`` targets.

    Args:
        poses: ``(N, 3)`` array of ground-truth poses.

    Returns:
        ``(N, 4)`` float64 regression targets.
    """
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"poses must be (N, 3), got {array.shape}.")
    yaw = array[:, 2]
    return np.column_stack([array[:, 0], array[:, 1], np.cos(yaw), np.sin(yaw)])


def targets_to_pose(targets: np.ndarray) -> np.ndarray:
    """Decode ``(x, y, cos yaw, sin yaw)`` back into ``(x, y, yaw)``.

    Args:
        targets: ``(N, 4)`` predictions.

    Returns:
        ``(N, 3)`` poses with yaw wrapped to ``[-pi, pi)``.
    """
    array = np.asarray(targets, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != _TARGET_DIM:
        raise ValueError(f"targets must be (N, 4), got {array.shape}.")
    yaw = np.arctan2(array[:, 3], array[:, 2])
    return np.column_stack([array[:, 0], array[:, 1], yaw])


@dataclass(frozen=True)
class PoseProbe:
    """A fitted linear map from latent to pose.

    Attributes:
        weights: ``(D + 1, 4)`` coefficients, last row being the intercept.
        latent_dim: Latent dimension the probe was fitted on.
    """

    weights: np.ndarray
    latent_dim: int

    def predict(self, latents: np.ndarray) -> np.ndarray:
        """Predict ``(x, y, yaw)`` for a batch of latents.

        Args:
            latents: ``(N, D)`` flattened latents.

        Returns:
            ``(N, 3)`` predicted poses.
        """
        array = np.asarray(latents, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.latent_dim:
            raise ValueError(
                f"latents must be (N, {self.latent_dim}), got {array.shape}."
            )
        design = np.hstack([array, np.ones((len(array), 1))])
        return targets_to_pose(design @ self.weights)


def fit_pose_probe(
    latents: np.ndarray,
    poses: np.ndarray,
    ridge: float = 1e-3,
) -> PoseProbe:
    """Fit a ridge-regression probe from latents to pose.

    Solved in closed form. The intercept is excluded from the penalty, since
    shrinking it would bias every prediction toward the origin of whatever
    arbitrary coordinate frame the room happens to use.

    Args:
        latents: ``(N, D)`` flattened latents from held-out data.
        poses: ``(N, 3)`` matching ground-truth poses.
        ridge: L2 penalty. Must be positive; the latent dimension usually
            exceeds the sample count on small collection runs, which makes the
            unregularised solution singular.

    Returns:
        The fitted :class:`PoseProbe`.
    """
    features = np.asarray(latents, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"latents must be 2-D (N, D), got {features.shape}.")
    if ridge <= 0.0:
        raise ValueError(f"ridge must be positive, got {ridge}.")

    targets = pose_targets(poses)
    if len(features) != len(targets):
        raise ValueError(
            f"latents and poses must align: {len(features)} vs {len(targets)}."
        )
    if len(features) <= 1:
        raise ValueError("Need at least two samples to fit a probe.")

    n_features = features.shape[1]
    design = np.hstack([features, np.ones((len(features), 1))])

    penalty = np.eye(n_features + 1)
    penalty[-1, -1] = 0.0  # do not penalise the intercept

    gram = design.T @ design + ridge * penalty
    weights = np.linalg.solve(gram, design.T @ targets)
    return PoseProbe(weights=weights, latent_dim=n_features)
