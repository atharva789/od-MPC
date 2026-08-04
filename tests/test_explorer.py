"""Tests for the scripted exploration policy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from od_mpc.data.explorer import (
    ExplorerConfig,
    ExplorerState,
    initial_state,
    step_explorer,
)

pytestmark = pytest.mark.unit

CONFIG = ExplorerConfig()


def _open_scan(n: int = 360, distance: float = 12.0) -> np.ndarray:
    return np.full(n, distance, dtype=np.float64)


def test_open_space_drives_forward() -> None:
    command, _ = step_explorer(_open_scan(), initial_state(), CONFIG)
    assert command.vx > 0.0


def test_speed_respects_configured_ceiling() -> None:
    command, _ = step_explorer(_open_scan(), initial_state(), CONFIG)
    assert command.vx <= CONFIG.max_linear + 1e-9
    assert abs(command.wz) <= CONFIG.max_yaw + 1e-9


def test_wall_ahead_slows_the_robot() -> None:
    """A near obstacle in front must reduce forward speed versus open space."""
    scan = _open_scan()
    scan[:20] = 0.5
    scan[-20:] = 0.5
    blocked, _ = step_explorer(scan, initial_state(), CONFIG)
    clear, _ = step_explorer(_open_scan(), initial_state(), CONFIG)
    assert blocked.vx < clear.vx


def test_boxed_in_triggers_reverse() -> None:
    scan = np.full(360, 0.3, dtype=np.float64)
    command, state = step_explorer(scan, initial_state(), CONFIG)
    assert command.vx < 0.0
    assert state.reverse_remaining == CONFIG.reverse_steps


def test_reverse_manoeuvre_counts_down_and_ends() -> None:
    state = ExplorerState(reverse_remaining=2)
    _, state = step_explorer(_open_scan(), state, CONFIG)
    assert state.reverse_remaining == 1
    _, state = step_explorer(_open_scan(), state, CONFIG)
    assert state.reverse_remaining == 0


def test_turns_toward_the_open_direction() -> None:
    """With space only to the left, yaw must be positive (counter-clockwise)."""
    scan = np.full(360, 0.8, dtype=np.float64)
    scan[80:100] = 12.0  # ~90 degrees CCW
    command, _ = step_explorer(scan, initial_state(), CONFIG)
    assert command.wz > 0.0


def test_heading_is_committed_across_steps() -> None:
    _, state = step_explorer(_open_scan(), initial_state(), CONFIG)
    assert state.commit_remaining == CONFIG.commit_steps
    bearing = state.target_bearing
    _, next_state = step_explorer(_open_scan(), state, CONFIG)
    assert next_state.target_bearing == bearing
    assert next_state.commit_remaining == CONFIG.commit_steps - 1


def test_state_is_not_mutated() -> None:
    state = initial_state()
    step_explorer(_open_scan(), state, CONFIG)
    assert state == ExplorerState()


def test_is_deterministic() -> None:
    scan = _open_scan()
    first, _ = step_explorer(scan, initial_state(), CONFIG)
    second, _ = step_explorer(scan, initial_state(), CONFIG)
    assert first == second


def test_rejects_empty_scan() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        step_explorer(np.array([]), initial_state(), CONFIG)


def test_rejects_non_finite_scan() -> None:
    scan = _open_scan()
    scan[0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        step_explorer(scan, initial_state(), CONFIG)


def test_config_rejects_inverted_clearances() -> None:
    with pytest.raises(ValueError, match="clearance_stop"):
        ExplorerConfig(clearance_stop=2.0, clearance_slow=1.0)


def test_config_rejects_bad_arc() -> None:
    with pytest.raises(ValueError, match="front_arc_deg"):
        ExplorerConfig(front_arc_deg=270.0)


def test_never_emits_non_finite_commands() -> None:
    rng = np.random.default_rng(0)
    state = initial_state()
    for _ in range(200):
        scan = rng.uniform(0.1, 12.0, size=180)
        command, state = step_explorer(scan, state, CONFIG)
        assert math.isfinite(command.vx) and math.isfinite(command.wz)
