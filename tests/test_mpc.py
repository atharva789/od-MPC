"""Tests for the CEM planner and the goal cost.

The planner is exercised against closed-form cost functions rather than a
trained model, so these tests check the search itself: that it converges to a
known optimum, respects action bounds, and refuses to plan through a cost
function that is lying to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from od_mpc.mpc.cost import GoalCostConfig, goal_latent_cost
from od_mpc.mpc.planner import PlannerConfig, plan_actions

pytestmark = pytest.mark.unit

SMALL = PlannerConfig(horizon=4, population=64, elites=8, iterations=6)


def _distance_to(target: np.ndarray):
    """Cost pulling every action toward ``target``."""

    def cost(actions: np.ndarray) -> np.ndarray:
        return np.mean((actions - target) ** 2, axis=(1, 2))

    return cost


# --- planner ----------------------------------------------------------------


def test_converges_toward_a_known_optimum() -> None:
    target = np.array([0.5, -0.25])
    result = plan_actions(_distance_to(target), SMALL, np.random.default_rng(0))
    np.testing.assert_allclose(result.actions, np.broadcast_to(target, (4, 2)), atol=0.1)


def test_reports_the_cost_of_the_returned_sequence() -> None:
    cost_fn = _distance_to(np.array([0.0, 0.0]))
    result = plan_actions(cost_fn, SMALL, np.random.default_rng(1))
    recomputed = cost_fn(result.actions[None])[0]
    assert result.cost == pytest.approx(recomputed, abs=1e-9)


def test_more_iterations_do_not_get_worse() -> None:
    target = np.array([0.8, 0.8])
    few = plan_actions(
        _distance_to(target),
        PlannerConfig(horizon=4, population=64, elites=8, iterations=1),
        np.random.default_rng(2),
    )
    many = plan_actions(
        _distance_to(target),
        PlannerConfig(horizon=4, population=64, elites=8, iterations=8),
        np.random.default_rng(2),
    )
    assert many.cost <= few.cost


def test_actions_stay_within_bounds() -> None:
    """The optimum sits outside the box; the planner must clip to it."""
    result = plan_actions(
        _distance_to(np.array([10.0, -10.0])), SMALL, np.random.default_rng(3)
    )
    assert np.all(result.actions >= -1.0) and np.all(result.actions <= 1.0)


def test_output_shapes() -> None:
    result = plan_actions(_distance_to(np.zeros(2)), SMALL, np.random.default_rng(4))
    assert result.actions.shape == (4, 2)
    assert result.mean.shape == (4, 2)
    assert result.first_action.shape == (2,)


def test_is_reproducible_for_a_given_seed() -> None:
    cost = _distance_to(np.array([0.3, 0.3]))
    first = plan_actions(cost, SMALL, np.random.default_rng(5))
    second = plan_actions(cost, SMALL, np.random.default_rng(5))
    np.testing.assert_array_equal(first.actions, second.actions)


def test_warm_start_is_accepted() -> None:
    warm = np.full((4, 2), 0.5)
    result = plan_actions(
        _distance_to(np.array([0.5, 0.5])), SMALL, np.random.default_rng(6), warm
    )
    assert result.actions.shape == (4, 2)


def test_rejects_wrong_warm_start_shape() -> None:
    with pytest.raises(ValueError, match="warm_start"):
        plan_actions(
            _distance_to(np.zeros(2)), SMALL, np.random.default_rng(7), np.zeros((9, 2))
        )


def test_rejects_badly_shaped_costs() -> None:
    with pytest.raises(ValueError, match="must return"):
        plan_actions(lambda a: np.zeros(3), SMALL, np.random.default_rng(8))


def test_rejects_non_finite_costs() -> None:
    """A diverged rollout must fail loudly, not be selected as an elite."""
    with pytest.raises(ValueError, match="non-finite"):
        plan_actions(
            lambda a: np.full(a.shape[0], np.nan), SMALL, np.random.default_rng(9)
        )


def test_config_rejects_too_many_elites() -> None:
    with pytest.raises(ValueError, match="elites"):
        PlannerConfig(population=10, elites=20)


def test_config_rejects_std_floor_above_start() -> None:
    with pytest.raises(ValueError, match="min_std"):
        PlannerConfig(initial_std=0.1, min_std=0.5)


# --- cost -------------------------------------------------------------------


def test_cost_is_lower_for_latents_nearer_the_goal() -> None:
    goal = np.ones(8)
    near = np.broadcast_to(goal * 0.95, (2, 3, 8)).copy()
    far = np.zeros((2, 3, 8))
    actions = np.zeros((2, 3, 2))
    assert np.all(
        goal_latent_cost(near, goal, actions) < goal_latent_cost(far, goal, actions)
    )


def test_terminal_state_dominates_the_path() -> None:
    """Ending at the goal must beat passing through it and leaving."""
    goal = np.ones(4)
    actions = np.zeros((1, 3, 2))
    arrives = np.stack([np.zeros(4), np.zeros(4), goal])[None]
    departs = np.stack([goal, goal, np.zeros(4)])[None]
    assert goal_latent_cost(arrives, goal, actions) < goal_latent_cost(departs, goal, actions)


def test_action_penalty_prefers_smaller_commands() -> None:
    goal = np.zeros(4)
    latents = np.zeros((2, 3, 4))
    actions = np.stack([np.zeros((3, 2)), np.ones((3, 2))])
    costs = goal_latent_cost(latents, goal, actions)
    assert costs[0] < costs[1]


def test_smoothness_penalty_prefers_steady_commands() -> None:
    goal = np.zeros(4)
    latents = np.zeros((2, 4, 4))
    steady = np.full((4, 2), 0.5)
    jerky = np.array([[0.5, 0.5], [-0.5, -0.5], [0.5, 0.5], [-0.5, -0.5]])
    costs = goal_latent_cost(latents, goal, np.stack([steady, jerky]))
    assert costs[0] < costs[1]


def test_cost_accepts_multi_dimensional_latents() -> None:
    """Packed (K, H, n_spatial, D) latents must work without reshaping."""
    costs = goal_latent_cost(np.zeros((2, 3, 5, 4)), np.zeros((5, 4)), np.zeros((2, 3, 2)))
    assert costs.shape == (2,)


def test_cost_rejects_goal_of_the_wrong_size() -> None:
    with pytest.raises(ValueError, match="goal_latent"):
        goal_latent_cost(np.zeros((2, 3, 8)), np.zeros(5), np.zeros((2, 3, 2)))


def test_cost_rejects_mismatched_actions() -> None:
    with pytest.raises(ValueError, match="actions must be"):
        goal_latent_cost(np.zeros((2, 3, 8)), np.zeros(8), np.zeros((2, 9, 2)))


def test_cost_config_rejects_ignoring_the_goal() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        GoalCostConfig(terminal_weight=0.0, path_weight=0.0)


def test_cost_config_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        GoalCostConfig(terminal_weight=-1.0)


def test_cost_rejects_latents_with_too_few_dimensions() -> None:
    with pytest.raises(ValueError, match="at least"):
        goal_latent_cost(np.zeros((2, 8)), np.zeros(8), np.zeros((2, 3, 2)))


def test_cost_has_no_jerk_term_for_a_single_step_horizon() -> None:
    """A horizon of one has no successive-action difference to penalise."""
    goal = np.zeros(4)
    latents = np.zeros((2, 1, 4))
    steady = np.full((1, 2), 0.5)
    swung = np.full((1, 2), -0.5)
    costs = goal_latent_cost(latents, goal, np.stack([steady, swung]))
    assert costs[0] == pytest.approx(costs[1])


def test_planner_and_cost_compose() -> None:
    """End-to-end: plan through a cost built from goal_latent_cost."""
    goal = np.array([1.0, 0.0, 0.0, 0.0])

    def rollout_cost(actions: np.ndarray) -> np.ndarray:
        # Toy dynamics: the latent is the cumulative action, zero-padded.
        cumulative = np.cumsum(actions, axis=1)
        latents = np.concatenate(
            [cumulative, np.zeros((*cumulative.shape[:2], 2))], axis=2
        )
        return goal_latent_cost(latents, goal, actions)

    result = plan_actions(rollout_cost, SMALL, np.random.default_rng(10))
    assert np.isfinite(result.cost)
    assert result.actions.shape == (4, 2)
