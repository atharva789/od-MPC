"""Cost functions scoring predicted latent rollouts.

The planner needs a scalar per candidate. These build one from a predicted
latent trajectory and a goal latent obtained by encoding a goal image with the
same tokenizer, so goal and prediction live in the same space by construction.

Weighting the terminal state more heavily than the path is the usual choice for
goal reaching: the intermediate states are a means, not the objective. The
action penalty exists to break ties toward smoother commands, which matters on
a real base where a jerky velocity profile is bad for both the hardware and the
next frame's motion blur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GoalCostConfig", "goal_latent_cost"]


@dataclass(frozen=True)
class GoalCostConfig:
    """Weights for :func:`goal_latent_cost`.

    Attributes:
        terminal_weight: Weight on the final predicted latent's distance.
        path_weight: Weight on the mean distance across the rollout.
        action_weight: Penalty on squared action magnitude.
        smoothness_weight: Penalty on squared successive action differences.
    """

    terminal_weight: float = 1.0
    path_weight: float = 0.1
    action_weight: float = 0.01
    smoothness_weight: float = 0.05

    def __post_init__(self) -> None:
        weights = (
            self.terminal_weight,
            self.path_weight,
            self.action_weight,
            self.smoothness_weight,
        )
        if any(w < 0.0 for w in weights):
            raise ValueError(f"Cost weights must be non-negative, got {weights}.")
        if self.terminal_weight == 0.0 and self.path_weight == 0.0:
            raise ValueError(
                "At least one of terminal_weight or path_weight must be "
                "non-zero, or the cost ignores the goal entirely."
            )


def goal_latent_cost(
    predicted_latents: np.ndarray,
    goal_latent: np.ndarray,
    actions: np.ndarray,
    config: GoalCostConfig = GoalCostConfig(),
) -> np.ndarray:
    """Score candidates by distance to a goal latent.

    Args:
        predicted_latents: ``(K, H, ...)`` predicted latents per candidate.
            Trailing dimensions are flattened, so packed or unpacked latents
            both work.
        goal_latent: Goal latent whose trailing dimensions match a single
            predicted step.
        actions: ``(K, H, 2)`` candidate actions being scored.
        config: Cost weights.

    Returns:
        ``(K,)`` costs, lower being better.
    """
    latents = np.asarray(predicted_latents, dtype=np.float64)
    if latents.ndim < 3:
        raise ValueError(
            f"predicted_latents must be at least (K, H, D), got {latents.shape}."
        )

    population, horizon = latents.shape[0], latents.shape[1]
    flat = latents.reshape(population, horizon, -1)

    goal = np.asarray(goal_latent, dtype=np.float64).reshape(-1)
    if goal.size != flat.shape[2]:
        raise ValueError(
            f"goal_latent has {goal.size} elements but each predicted step has "
            f"{flat.shape[2]}."
        )

    action_batch = np.asarray(actions, dtype=np.float64)
    if action_batch.shape != (population, horizon, 2):
        raise ValueError(
            f"actions must be ({population}, {horizon}, 2), "
            f"got {action_batch.shape}."
        )

    # Mean squared error per step keeps the scale independent of latent width,
    # so the weights below do not need retuning when d_bottleneck changes.
    per_step = np.mean((flat - goal[None, None, :]) ** 2, axis=2)

    terminal = per_step[:, -1]
    path = per_step.mean(axis=1)
    effort = np.mean(action_batch**2, axis=(1, 2))

    if horizon > 1:
        jerk = np.mean(np.diff(action_batch, axis=1) ** 2, axis=(1, 2))
    else:
        jerk = np.zeros(population, dtype=np.float64)

    return (
        config.terminal_weight * terminal
        + config.path_weight * path
        + config.action_weight * effort
        + config.smoothness_weight * jerk
    )
