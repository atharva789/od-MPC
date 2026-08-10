"""Tests for the innate-os checkout resolution logic in ``od_mpc.sim.adapter``.

These exercise only the pure filesystem/path logic that runs before the real
``mars_sim_driver`` package is imported. They do not require an innate-os
checkout, MuJoCo, or a network connection, and they do not assert anything
about the real ``VirtualMars`` API beyond what is already recorded in
docs/SPEC.md as verified from source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from od_mpc.sim.adapter import (
    INNATE_OS_ENV_VAR,
    driver_package_path,
    load_virtual_mars,
)

pytestmark = pytest.mark.unit


def _make_checkout(root: Path) -> Path:
    """Lay out an innate-os-shaped checkout under ``root`` and return the
    directory that should be placed on ``sys.path``."""
    driver_dir = root / "ros2_ws" / "src" / "mars_bot" / "mars_sim_driver"
    (driver_dir / "mars_sim_driver").mkdir(parents=True)
    return driver_dir


# --- driver_package_path -----------------------------------------------------


def test_driver_package_path_raises_without_root_or_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(INNATE_OS_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match=INNATE_OS_ENV_VAR):
        driver_package_path()


def test_driver_package_path_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        driver_package_path(missing)


def test_driver_package_path_rejects_checkout_without_driver_package(tmp_path: Path) -> None:
    # A real directory, but not shaped like an innate-os checkout.
    with pytest.raises(FileNotFoundError, match="mars_sim_driver"):
        driver_package_path(tmp_path)


def test_driver_package_path_resolves_explicit_root(tmp_path: Path) -> None:
    expected = _make_checkout(tmp_path)
    assert driver_package_path(tmp_path) == expected


def test_driver_package_path_falls_back_to_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _make_checkout(tmp_path)
    monkeypatch.setenv(INNATE_OS_ENV_VAR, str(tmp_path))
    assert driver_package_path() == expected


def test_driver_package_path_explicit_root_overrides_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_root = tmp_path / "env-root"
    env_root.mkdir()
    _make_checkout(env_root)
    monkeypatch.setenv(INNATE_OS_ENV_VAR, str(env_root))

    explicit_root = tmp_path / "explicit-root"
    expected = _make_checkout(explicit_root)

    assert driver_package_path(explicit_root) == expected


# --- load_virtual_mars --------------------------------------------------------


def test_load_virtual_mars_propagates_missing_checkout_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        load_virtual_mars(missing)


def test_load_virtual_mars_wraps_import_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The package directory exists but has no importable ``core`` module, so
    # this exercises the ImportError-wrapping branch without needing MuJoCo
    # or a real mars_sim_driver checkout.
    driver_dir = _make_checkout(tmp_path)

    import sys

    monkeypatch.delitem(sys.modules, "mars_sim_driver", raising=False)
    monkeypatch.delitem(sys.modules, "mars_sim_driver.core", raising=False)

    with pytest.raises(ImportError, match="mars_sim_driver"):
        load_virtual_mars(tmp_path)

    assert str(driver_dir) in sys.path


def test_load_virtual_mars_does_not_duplicate_sys_path_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Calling load_virtual_mars twice (e.g. across two collection runs in the
    # same process) must not grow sys.path without bound: the second call
    # should hit the "already on sys.path" branch and skip the insert.
    driver_dir = _make_checkout(tmp_path)
    package_dir = driver_dir / "mars_sim_driver"
    (package_dir / "__init__.py").write_text("")
    (package_dir / "core.py").write_text("class VirtualMars:\n    pass\n")

    import sys

    monkeypatch.delitem(sys.modules, "mars_sim_driver", raising=False)
    monkeypatch.delitem(sys.modules, "mars_sim_driver.core", raising=False)
    monkeypatch.setattr(sys, "path", [str(driver_dir), *sys.path])
    try:
        load_virtual_mars(tmp_path)
        assert sys.path.count(str(driver_dir)) == 1
    finally:
        sys.modules.pop("mars_sim_driver.core", None)
        sys.modules.pop("mars_sim_driver", None)


def test_load_virtual_mars_returns_the_imported_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stand-in ``mars_sim_driver.core`` module authored entirely by this
    # test (not copied from innate-os) that only needs to exist and expose
    # something named ``VirtualMars`` for load_virtual_mars to import and
    # return it. This exercises the success-path return without asserting
    # anything about the real class beyond what SPEC.md 2.1 already records
    # as verified from source: that ``mars_sim_driver.core`` exports
    # ``VirtualMars``.
    driver_dir = _make_checkout(tmp_path)
    package_dir = driver_dir / "mars_sim_driver"
    (package_dir / "__init__.py").write_text("")
    (package_dir / "core.py").write_text("class VirtualMars:\n    pass\n")

    import sys

    monkeypatch.delitem(sys.modules, "mars_sim_driver", raising=False)
    monkeypatch.delitem(sys.modules, "mars_sim_driver.core", raising=False)
    try:
        result = load_virtual_mars(tmp_path)
        core = sys.modules["mars_sim_driver.core"]
        assert result is core.VirtualMars
    finally:
        sys.modules.pop("mars_sim_driver.core", None)
        sys.modules.pop("mars_sim_driver", None)
