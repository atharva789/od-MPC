"""Cross-entropy-method planning over action sequences.

The planner is deliberately ignorant of what it is planning *through*. It takes
a ``rollout_cost`` callable that scores a batch of candidate action sequences
and returns one cost each. In production that callable runs open-dreamer's
``Dynamics`` and scores the predicted latents; in tests it is a closed-form
function. Keeping the search separate from the model is what makes the search
itself testable without JAX, a checkpoint, or a GPU.

CEM rather than gradient descent through the model: the dynamics model is a
few-step denoiser, so backpropagating through a twenty-step rollout means
differentiating through twenty denoising ladders. Sampling sidesteps that
entirely and parallelises across candidates, which is what the accelerator is
good at anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

__all__ = ["PlannerConfig", "PlanResult", "RolloutCost", "plan_actions"]


class RolloutCost(Protocol):
    """Scores candidate action sequences."""

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        """Score a batch of candidates.

        Args:
            actions: ``(K, H, 2)`` normalised action sequences.

        Returns:
            ``(K,)`` costs; lower is better.
        """
        ...


@dataclass(frozen=True)
class PlannerConfig:
    """Cross-entropy method settings.

    Attributes:
        horizon: Steps planned ahead.
        population: Candidates sampled per iteration.
        elites: Best candidates refitted against each iteration.
        iterations: Refinement rounds.
        initial_std: Starting exploration width in normalised action units.
        min_std: Floor on the std, preventing premature collapse onto one
            candidate before the model has really been consulted.
        momentum: Blend factor retaining the previous mean between iterations.
    """

    horizon: int = 20
    population: int = 256
    elites: int = 32
    iterations: int = 4
    initial_std: float = 0.5
    min_std: float = 0.05
    momentum: float = 0.1

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {self.horizon}.")
        if self.elites < 1 or self.elites > self.population:
            raise ValueError(
                f"elites must be in [1, population], got {self.elites} "
                f"with population {self.population}."
            )
        if self.iterations < 1:
            raise ValueError(f"iterations must be at least 1, got {self.iterations}.")
        if self.initial_std <= 0.0 or self.min_std <= 0.0:
            raise ValueError("Standard deviations must be positive.")
        if self.min_std > self.initial_std:
            raise ValueError(
                f"min_std {self.min_std} exceeds initial_std {self.initial_std}."
            )
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}.")


@dataclass(frozen=True)
class PlanResult:
    """Outcome of a planning call.

    Attributes:
        actions: ``(H, 2)`` best action sequence found.
        cost: Cost of that sequence.
        mean: ``(H, 2)`` final distribution mean, for warm-starting the next
            call.
    """

    actions: np.ndarray
    cost: float
    mean: np.ndarray

    @property
    def first_action(self) -> np.ndarray:
        """The action to execute now, shaped ``(2,)``.

        MPC executes only this and replans, which is what keeps a drifting
        model usable: errors are corrected before they compound.
        """
        return self.actions[0]


def plan_actions(
    rollout_cost: RolloutCost | Callable[[np.ndarray], np.ndarray],
    config: PlannerConfig,
    rng: np.random.Generator,
    warm_start: np.ndarray | None = None,
) -> PlanResult:
    """Search for a low-cost action sequence.

    Args:
        rollout_cost: Scores a ``(K, H, 2)`` batch, returning ``(K,)`` costs.
        config: Planner settings.
        rng: Seeded generator; planning is reproducible given the same one.
        warm_start: Optional ``(H, 2)`` mean to start from, usually the
            previous call's result shifted by one step.

    Returns:
        The best sequence found and the final distribution mean.

    Raises:
        ValueError: If ``warm_start`` has the wrong shape, or the cost function
            returns a badly shaped or non-finite result.
    """
    shape = (config.horizon, 2)

    if warm_start is None:
        mean = np.zeros(shape, dtype=np.float64)
    else:
        mean = np.asarray(warm_start, dtype=np.float64)
        if mean.shape != shape:
            raise ValueError(f"warm_start must be {shape}, got {mean.shape}.")

    std = np.full(shape, config.initial_std, dtype=np.float64)

    best_actions = np.clip(mean, -1.0, 1.0)
    best_cost = np.inf

    for _ in range(config.iterations):
        noise = rng.standard_normal((config.population, *shape))
        candidates = np.clip(mean + std * noise, -1.0, 1.0)

        costs = np.asarray(rollout_cost(candidates), dtype=np.float64)
        if costs.shape != (config.population,):
            raise ValueError(
                f"rollout_cost must return ({config.population},), "
                f"got {costs.shape}."
            )
        if not np.all(np.isfinite(costs)):
            raise ValueError(
                "rollout_cost returned non-finite values; a diverged rollout "
                "would otherwise be silently selected as an elite."
            )

        elite_idx = np.argsort(costs)[: config.elites]
        elites = candidates[elite_idx]

        if costs[elite_idx[0]] < best_cost:
            best_cost = float(costs[elite_idx[0]])
            best_actions = elites[0].copy()

        new_mean = elites.mean(axis=0)
        new_std = np.maximum(elites.std(axis=0), config.min_std)

        mean = config.momentum * mean + (1.0 - config.momentum) * new_mean
        std = new_std

    return PlanResult(
        actions=best_actions,
        cost=best_cost,
        mean=np.clip(mean, -1.0, 1.0),
    )
