"""od-MPC: latent model-predictive control for the MARS robot.

A harness around open-dreamer's world model, pointed at Innate's MARS
simulator. Four pieces:

* :mod:`od_mpc.data` collects navigation trajectories from the simulator.
* :mod:`od_mpc.train` drives open-dreamer's training scripts.
* :mod:`od_mpc.validate` measures whether latent rollouts track ground truth.
* :mod:`od_mpc.mpc` plans action sequences through the trained model.

od-MPC vendors neither open-dreamer nor innate-os. Both are supplied by the
user as local checkouts. See README.md for the licensing rationale.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
