"""Adapter over the real ``VirtualMars`` simulator from innate-os.

innate-os is Apache-2.0 but is not published to PyPI, so it cannot be a normal
dependency. The user supplies a checkout and this module locates the package
inside it. Nothing from innate-os is vendored here.

The import is deferred to :func:`load_virtual_mars` so that importing od-MPC,
running the unit tests, or using :class:`~od_mpc.sim.stub.StubMars` never
requires MuJoCo or a checkout to be present.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["INNATE_OS_ENV_VAR", "driver_package_path", "load_virtual_mars"]

INNATE_OS_ENV_VAR = "INNATE_OS_ROOT"

# Location of the driver package within an innate-os checkout. Recorded in
# docs/SPEC.md alongside the revision this was verified against.
_DRIVER_SUBPATH = Path("ros2_ws/src/mars_bot/mars_sim_driver")


def driver_package_path(innate_os_root: str | Path | None = None) -> Path:
    """Resolve the directory containing the ``mars_sim_driver`` package.

    Args:
        innate_os_root: Path to an innate-os checkout. Falls back to the
            ``INNATE_OS_ROOT`` environment variable.

    Returns:
        The directory to place on ``sys.path``.

    Raises:
        FileNotFoundError: If the checkout or the driver package is missing.
        ValueError: If no root was supplied and the env var is unset.
    """
    root = innate_os_root or os.environ.get(INNATE_OS_ENV_VAR)
    if not root:
        raise ValueError(
            "No innate-os checkout supplied. Pass innate_os_root explicitly or "
            f"set {INNATE_OS_ENV_VAR} to a clone of "
            "https://github.com/innate-inc/innate-os."
        )

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"innate-os root is not a directory: {root_path}")

    package_dir = root_path / _DRIVER_SUBPATH
    if not (package_dir / "mars_sim_driver").is_dir():
        raise FileNotFoundError(
            f"Could not find mars_sim_driver under {package_dir}. Expected the "
            "innate-os layout ros2_ws/src/mars_bot/mars_sim_driver/. If the "
            "upstream layout has moved, update _DRIVER_SUBPATH."
        )
    return package_dir


def load_virtual_mars(innate_os_root: str | Path | None = None) -> type[Any]:
    """Import and return the real ``VirtualMars`` class.

    Args:
        innate_os_root: Path to an innate-os checkout, or ``None`` to read
            ``INNATE_OS_ROOT`` from the environment.

    Returns:
        The ``VirtualMars`` class, which satisfies
        :class:`~od_mpc.sim.protocol.MarsSim`.

    Raises:
        ImportError: If the package is present but cannot be imported, most
            commonly because MuJoCo is not installed.
    """
    package_dir = str(driver_package_path(innate_os_root))
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

    try:
        core = importlib.import_module("mars_sim_driver.core")
    except ImportError as exc:  # pragma: no cover - needs a real checkout
        raise ImportError(
            f"Found mars_sim_driver at {package_dir} but could not import it: "
            f"{exc}. The simulator needs MuJoCo; install the extras from "
            "innate-os's sim/pyproject.toml."
        ) from exc

    return core.VirtualMars
