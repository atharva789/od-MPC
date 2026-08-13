"""Drive open-dreamer's training entry points.

This is a wrapper, not a reimplementation. open-dreamer is All Rights Reserved,
so none of its code can be copied here; and even setting licensing aside,
reimplementing a training loop that already exists would mean maintaining a
divergent copy of someone else's research code.

The pipeline therefore does three things: render od-MPC's settings into config
overrides, invoke the upstream training script as a subprocess, and report
where the checkpoint landed. Everything model-shaped stays upstream.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "OPEN_DREAMER_ENV_VAR",
    "TrainStage",
    "TrainInvocation",
    "build_invocation",
    "run_stage",
]

OPEN_DREAMER_ENV_VAR = "OPEN_DREAMER_ROOT"

_STAGE_SCRIPTS = {
    "tokenizer": Path("scripts/train_tokenizer.py"),
    "dynamics": Path("scripts/train_dynamics.py"),
}


@dataclass(frozen=True)
class TrainStage:
    """One training stage.

    Attributes:
        name: Either ``"tokenizer"`` or ``"dynamics"``.
        overrides: Hydra-style ``key=value`` overrides.
    """

    name: str
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in _STAGE_SCRIPTS:
            raise ValueError(
                f"Unknown stage {self.name!r}; expected one of "
                f"{sorted(_STAGE_SCRIPTS)}."
            )
        for override in self.overrides:
            if "=" not in override:
                raise ValueError(f"Override {override!r} is not in key=value form.")


@dataclass(frozen=True)
class TrainInvocation:
    """A fully resolved command, ready to run or to print.

    Attributes:
        command: Argv list.
        cwd: Directory the command must run from.
    """

    command: tuple[str, ...]
    cwd: Path

    def as_shell(self) -> str:
        """Render the command as a copy-pasteable shell line."""
        return f"cd {shlex.quote(str(self.cwd))} && {shlex.join(self.command)}"


def open_dreamer_root(root: str | Path | None = None) -> Path:
    """Resolve the open-dreamer checkout.

    Args:
        root: Explicit path, or ``None`` to read ``OPEN_DREAMER_ROOT``.

    Returns:
        The checkout directory.

    Raises:
        ValueError: If no path was supplied and the env var is unset.
        FileNotFoundError: If the path is not an open-dreamer checkout.
    """
    resolved = root or os.environ.get(OPEN_DREAMER_ENV_VAR)
    if not resolved:
        raise ValueError(
            "No open-dreamer checkout supplied. Pass root explicitly or set "
            f"{OPEN_DREAMER_ENV_VAR}. od-MPC does not vendor open-dreamer; see "
            "the licensing note in README.md."
        )

    path = Path(resolved).expanduser().resolve()
    if not (path / "dreamer").is_dir():
        raise FileNotFoundError(
            f"{path} does not look like an open-dreamer checkout "
            "(no dreamer/ package)."
        )
    return path


def build_invocation(
    stage: TrainStage,
    root: str | Path | None = None,
    python_executable: str | None = None,
) -> TrainInvocation:
    """Build the command for a training stage without running it.

    Separated from :func:`run_stage` so a dry run can print exactly what would
    execute. Launching a multi-hour GPU job is not something to do by accident.

    Args:
        stage: The stage to build.
        root: open-dreamer checkout, or ``None`` to read the environment.
        python_executable: Interpreter to use; defaults to the current one.

    Returns:
        The resolved invocation.
    """
    checkout = open_dreamer_root(root)
    script = checkout / _STAGE_SCRIPTS[stage.name]
    if not script.is_file():
        raise FileNotFoundError(
            f"Expected {script} in the open-dreamer checkout. If upstream has "
            "renamed its training scripts, update _STAGE_SCRIPTS."
        )

    command = (
        python_executable or sys.executable,
        str(script.relative_to(checkout)),
        *stage.overrides,
    )
    return TrainInvocation(command=command, cwd=checkout)


def run_stage(
    stage: TrainStage,
    root: str | Path | None = None,
    dry_run: bool = True,
) -> TrainInvocation:
    """Build and optionally execute a training stage.

    Defaults to ``dry_run=True``. The caller must opt in to actually spending
    GPU hours.

    Args:
        stage: The stage to run.
        root: open-dreamer checkout, or ``None`` to read the environment.
        dry_run: When true, build the command and return without running it.

    Returns:
        The invocation that was built, whether or not it ran.

    Raises:
        subprocess.CalledProcessError: If the training script exits non-zero.
    """
    invocation = build_invocation(stage, root)
    if dry_run:
        return invocation

    subprocess.run(invocation.command, cwd=invocation.cwd, check=True)
    return invocation
