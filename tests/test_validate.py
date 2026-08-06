"""Tests for the pose probe and rollout metrics."""

from __future__ import annotations

import numpy as np
import pytest

from od_mpc.validate.metrics import drift_horizon, pose_drift, psnr, wrap_angle
from od_mpc.validate.probe import fit_pose_probe, pose_targets, targets_to_pose

pytestmark = pytest.mark.unit


# --- angle handling ---------------------------------------------------------


def test_wrap_angle_maps_into_range() -> None:
    wrapped = wrap_angle(np.array([0.0, 3 * np.pi, -3 * np.pi, np.pi]))
    assert np.all(wrapped >= -np.pi) and np.all(wrapped < np.pi)


def test_wrap_angle_handles_the_seam() -> None:
    """The whole point: +pi and -pi are the same heading."""
    assert abs(wrap_angle(np.pi - 0.01) - wrap_angle(-np.pi - 0.01 + 2 * np.pi)) < 1e-9


# --- probe ------------------------------------------------------------------


def test_probe_recovers_an_exactly_linear_mapping() -> None:
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(200, 6))
    true_map = rng.normal(size=(6, 2))
    xy = latents @ true_map
    yaw = np.arctan2(latents[:, 0], latents[:, 1])
    poses = np.column_stack([xy, yaw])

    probe = fit_pose_probe(latents, poses, ridge=1e-8)
    predicted = probe.predict(latents)
    np.testing.assert_allclose(predicted[:, :2], poses[:, :2], atol=1e-4)


def test_probe_generalises_to_held_out_samples() -> None:
    rng = np.random.default_rng(1)
    latents = rng.normal(size=(300, 5))
    weights = rng.normal(size=(5, 2))
    poses = np.column_stack([latents @ weights, np.zeros(300)])

    probe = fit_pose_probe(latents[:200], poses[:200], ridge=1e-8)
    error = np.linalg.norm(probe.predict(latents[200:])[:, :2] - poses[200:, :2], axis=1)
    assert float(np.mean(error)) < 1e-3


def test_probe_fails_on_unpredictable_latents() -> None:
    """Noise must not yield a good fit, or the metric proves nothing."""
    rng = np.random.default_rng(2)
    latents = rng.normal(size=(400, 4))
    poses = np.column_stack(
        [rng.normal(size=400) * 5.0, rng.normal(size=400) * 5.0, np.zeros(400)]
    )
    probe = fit_pose_probe(latents[:300], poses[:300], ridge=1e-3)
    error = np.linalg.norm(probe.predict(latents[300:])[:, :2] - poses[300:, :2], axis=1)
    assert float(np.mean(error)) > 1.0


def test_yaw_encoding_roundtrips() -> None:
    poses = np.column_stack(
        [np.zeros(5), np.zeros(5), np.array([0.0, 1.5, -1.5, 3.0, -3.0])]
    )
    np.testing.assert_allclose(targets_to_pose(pose_targets(poses)), poses, atol=1e-9)


def test_probe_predicts_yaw_across_the_seam() -> None:
    """Yaw near +/-pi must not collapse toward the circular mean."""
    angles = np.linspace(-np.pi, np.pi, 200, endpoint=False)
    latents = np.column_stack([np.cos(angles), np.sin(angles)])
    poses = np.column_stack([np.zeros(200), np.zeros(200), angles])

    probe = fit_pose_probe(latents, poses, ridge=1e-9)
    error = np.abs(wrap_angle(probe.predict(latents)[:, 2] - angles))
    assert float(np.max(error)) < 1e-4


def test_probe_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        fit_pose_probe(np.zeros((10, 3)), np.zeros((9, 3)))


def test_probe_rejects_non_positive_ridge() -> None:
    with pytest.raises(ValueError, match="ridge must be positive"):
        fit_pose_probe(np.zeros((10, 3)), np.zeros((10, 3)), ridge=0.0)


def test_probe_rejects_wrong_latent_width_at_predict() -> None:
    probe = fit_pose_probe(np.random.default_rng(3).normal(size=(20, 4)), np.zeros((20, 3)))
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        probe.predict(np.zeros((5, 7)))


def test_probe_rejects_non_2d_latents() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        fit_pose_probe(np.zeros(10), np.zeros((10, 3)))


def test_probe_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        fit_pose_probe(np.zeros((1, 4)), np.zeros((1, 3)))


def test_pose_targets_rejects_wrong_pose_width() -> None:
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        pose_targets(np.zeros((5, 2)))


def test_targets_to_pose_rejects_wrong_target_width() -> None:
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        targets_to_pose(np.zeros((5, 3)))


# --- metrics ----------------------------------------------------------------


def test_psnr_is_infinite_for_identical_images() -> None:
    frames = np.full((4, 4, 3), 128, dtype=np.uint8)
    assert psnr(frames, frames) == float("inf")


def test_psnr_decreases_as_error_grows() -> None:
    target = np.zeros((8, 8, 3))
    assert psnr(np.full((8, 8, 3), 5.0), target) > psnr(np.full((8, 8, 3), 50.0), target)


def test_psnr_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        psnr(np.zeros((4, 4)), np.zeros((5, 5)))


def test_psnr_rejects_non_positive_data_range() -> None:
    with pytest.raises(ValueError, match="data_range must be positive"):
        psnr(np.zeros((2, 2)), np.zeros((2, 2)), data_range=0.0)


def test_pose_drift_is_zero_for_a_perfect_rollout() -> None:
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    position, heading = pose_drift(poses, poses)
    np.testing.assert_allclose(position, 0.0, atol=1e-12)
    np.testing.assert_allclose(heading, 0.0, atol=1e-12)


def test_pose_drift_measures_euclidean_distance() -> None:
    predicted = np.array([[3.0, 4.0, 0.0]])
    actual = np.array([[0.0, 0.0, 0.0]])
    position, _ = pose_drift(predicted, actual)
    assert position[0] == pytest.approx(5.0)


def test_pose_drift_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        pose_drift(np.zeros((4, 3)), np.zeros((5, 3)))


def test_pose_drift_rejects_wrong_pose_width() -> None:
    with pytest.raises(ValueError, match=r"\(T, 3\)"):
        pose_drift(np.zeros((4, 2)), np.zeros((4, 2)))


def test_heading_error_takes_the_short_way_round() -> None:
    predicted = np.array([[0.0, 0.0, np.pi - 0.1]])
    actual = np.array([[0.0, 0.0, -np.pi + 0.1]])
    _, heading = pose_drift(predicted, actual)
    assert heading[0] == pytest.approx(0.2, abs=1e-6)


def test_drift_horizon_counts_leading_good_steps() -> None:
    assert drift_horizon(np.array([0.1, 0.2, 0.9, 0.3]), tolerance=0.5) == 2


def test_drift_horizon_is_zero_when_the_first_step_fails() -> None:
    assert drift_horizon(np.array([5.0, 0.1]), tolerance=0.5) == 0


def test_drift_horizon_returns_full_length_when_always_within_tolerance() -> None:
    assert drift_horizon(np.array([0.1, 0.1, 0.1]), tolerance=0.5) == 3


def test_drift_horizon_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        drift_horizon(np.zeros((3, 2)), tolerance=0.5)


def test_drift_horizon_rejects_non_positive_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance must be positive"):
        drift_horizon(np.array([0.1, 0.2]), tolerance=0.0)
