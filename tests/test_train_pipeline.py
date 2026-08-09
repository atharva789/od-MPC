"""Tests for the open-dreamer training wrapper.

These use a fake checkout rather than a real one: the wrapper's job is to build
a correct command and to fail clearly when the checkout is wrong, and both are
testable without open-dreamer present.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from od_mpc.train.pipeline import (
    OPEN_DREAMER_ENV_VAR,
    TrainStage,
    build_invocation,
    open_dreamer_root,
    run_stage,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def fake_checkout(tmp_path: Path) -> Path:
    """A directory shaped like an open-dreamer checkout."""
    (tmp_path / "dreamer").mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train_tokenizer.py").write_text("")
    (scripts / "train_dynamics.py").write_text("")
    return tmp_path


def test_stage_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        TrainStage(name="policy")


def test_stage_rejects_malformed_overrides() -> None:
    with pytest.raises(ValueError, match="key=value"):
        TrainStage(name="tokenizer", overrides=("just_a_key",))


def test_root_resolution_requires_a_path(monkeypatch) -> None:
    monkeypatch.delenv(OPEN_DREAMER_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="No open-dreamer checkout"):
        open_dreamer_root()


def test_root_resolution_reads_the_environment(monkeypatch, fake_checkout: Path) -> None:
    monkeypatch.setenv(OPEN_DREAMER_ENV_VAR, str(fake_checkout))
    assert open_dreamer_root() == fake_checkout.resolve()


def test_root_rejects_a_directory_that_is_not_a_checkout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not look like"):
        open_dreamer_root(tmp_path)


def test_invocation_targets_the_right_script(fake_checkout: Path) -> None:
    invocation = build_invocation(TrainStage(name="dynamics"), fake_checkout)
    assert invocation.command[1] == "scripts/train_dynamics.py"
    assert invocation.cwd == fake_checkout.resolve()


def test_invocation_passes_overrides_through(fake_checkout: Path) -> None:
    stage = TrainStage(
        name="tokenizer",
        overrides=("dataset.continuous_action_dim=2", "max_steps=1000"),
    )
    command = build_invocation(stage, fake_checkout).command
    assert "dataset.continuous_action_dim=2" in command
    assert "max_steps=1000" in command


def test_invocation_renders_a_shell_line(fake_checkout: Path) -> None:
    shell = build_invocation(TrainStage(name="tokenizer"), fake_checkout).as_shell()
    assert shell.startswith("cd ")
    assert "train_tokenizer.py" in shell


def test_missing_script_is_reported(fake_checkout: Path) -> None:
    (fake_checkout / "scripts" / "train_dynamics.py").unlink()
    with pytest.raises(FileNotFoundError, match="train_dynamics.py"):
        build_invocation(TrainStage(name="dynamics"), fake_checkout)


def test_run_stage_defaults_to_dry_run(fake_checkout: Path) -> None:
    """The default must never launch a GPU job."""
    invocation = run_stage(TrainStage(name="tokenizer"), fake_checkout)
    assert invocation.command[1] == "scripts/train_tokenizer.py"


def test_run_stage_executes_the_script_when_not_a_dry_run(fake_checkout: Path) -> None:
    """The scripts are empty files, so running them as Python is a no-op exit 0."""
    invocation = run_stage(TrainStage(name="tokenizer"), fake_checkout, dry_run=False)
    assert invocation.command[1] == "scripts/train_tokenizer.py"


def test_run_stage_propagates_a_failing_script(fake_checkout: Path) -> None:
    (fake_checkout / "scripts" / "train_dynamics.py").write_text("import sys; sys.exit(1)")
    with pytest.raises(subprocess.CalledProcessError):
        run_stage(TrainStage(name="dynamics"), fake_checkout, dry_run=False)
