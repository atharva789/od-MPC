"""Tests for the cmd_vel <-> continuous-action adapter."""

from __future__ import annotations

import numpy as np
import pytest

from od_mpc.actions import (
    CmdVel,
    CmdVelLimits,
    DEFAULT_LIMITS,
    denormalize_cmd_vel,
    from_continuous_actions,
    normalize_cmd_vel,
    to_continuous_actions,
)

pytestmark = pytest.mark.unit


def test_defaults_match_documented_mars_limits() -> None:
    assert DEFAULT_LIMITS.max_linear == pytest.approx(0.4)
    assert DEFAULT_LIMITS.max_yaw == pytest.approx(1.0)


def test_normalize_maps_limits_to_unit_range() -> None:
    commands = np.array([[0.4, 1.0], [-0.4, -1.0], [0.0, 0.0]])
    normalized = normalize_cmd_vel(commands)
    np.testing.assert_allclose(
        normalized, [[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]], atol=1e-6
    )


def test_normalize_clips_beyond_limits() -> None:
    """The simulator clamps the command, so the record must clamp too."""
    normalized = normalize_cmd_vel(np.array([[10.0, -10.0]]))
    np.testing.assert_allclose(normalized, [[1.0, -1.0]])


def test_normalize_output_is_float32() -> None:
    assert normalize_cmd_vel(np.array([[0.1, 0.2]])).dtype == np.float32


def test_roundtrip_within_limits_is_lossless() -> None:
    commands = np.array([[0.13, -0.42], [-0.39, 0.87]])
    recovered = denormalize_cmd_vel(normalize_cmd_vel(commands))
    np.testing.assert_allclose(recovered, commands, atol=1e-6)


def test_roundtrip_via_cmdvel_objects() -> None:
    original = [CmdVel(0.2, -0.5), CmdVel(-0.1, 0.9)]
    decoded = from_continuous_actions(to_continuous_actions(original))
    assert len(decoded) == len(original)
    for got, want in zip(decoded, original):
        assert got.vx == pytest.approx(want.vx, abs=1e-6)
        assert got.wz == pytest.approx(want.wz, abs=1e-6)


def test_custom_limits_change_scaling() -> None:
    limits = CmdVelLimits(max_linear=1.0, max_yaw=2.0)
    np.testing.assert_allclose(
        normalize_cmd_vel(np.array([[0.5, 1.0]]), limits), [[0.5, 0.5]], atol=1e-6
    )


def test_to_continuous_actions_shape() -> None:
    actions = to_continuous_actions([CmdVel(0.1, 0.2)] * 7)
    assert actions.shape == (7, 2)


def test_rejects_wrong_trailing_dimension() -> None:
    with pytest.raises(ValueError, match="trailing dimension 2"):
        normalize_cmd_vel(np.zeros((4, 3)))


def test_rejects_non_finite_commands() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        normalize_cmd_vel(np.array([[np.nan, 0.0]]))


def test_rejects_empty_command_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        to_continuous_actions([])


def test_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        normalize_cmd_vel(np.array([[0.1, 0.1]]), CmdVelLimits(0.0, 1.0))


def test_normalize_does_not_mutate_input() -> None:
    commands = np.array([[10.0, 10.0]])
    before = commands.copy()
    normalize_cmd_vel(commands)
    np.testing.assert_array_equal(commands, before)
