"""Shared fixtures and CLI flags for the integration test suite.

Every test under ``tests/integration/`` carries the ``integration`` marker and
relies on a usable GMAT install. The :func:`gmat_available` fixture skips a
test module when discovery or the gmatpy bootstrap fails, so the suite
degrades to "skipped" rather than "errored" on a developer machine without
GMAT installed. CI primes a cached install before invoking pytest, so the
fixture passes through there.

The ``--regenerate-golden`` CLI flag flips the round-trip tests from compare
mode into write mode: they overwrite the committed CSV goldens with whatever
the current GMAT install produces, then ``pytest.skip`` so the run is loud
about *not* having compared anything. Regenerating goldens is a deliberate
act and only meaningful with a real GMAT install on the same machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gmat_run.errors import GmatLoadError, GmatNotFoundError
from gmat_run.install import locate_gmat
from gmat_run.runtime import bootstrap


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help=(
            "Overwrite the committed integration goldens with the current "
            "GMAT run's output and skip the comparison. Requires a working "
            "GMAT install."
        ),
    )


@pytest.fixture(scope="session")
def regenerate_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--regenerate-golden"))


@pytest.fixture(scope="session")
def gmat_available() -> None:
    """Skip the dependent test if no usable GMAT install is reachable.

    Both halves are checked: discovery (a structurally valid install on disk)
    and bootstrap (gmatpy importable in the current interpreter). The latter
    catches the cross-OS case where a Windows install is mounted into a Linux
    Python — discovery passes, but the .pyd cannot be loaded.

    Session-scoped so the cost (gmatpy module load is not free) is paid once
    per pytest run, not per test. Bootstrap is one-shot-per-process anyway:
    calling it here primes the cache for every subsequent ``Mission.load``.
    """
    try:
        install = locate_gmat()
    except GmatNotFoundError as exc:
        pytest.skip(f"no GMAT install discoverable: {exc}")
    try:
        bootstrap(install)
    except GmatLoadError as exc:
        pytest.skip(f"GMAT install discovered but gmatpy not loadable: {exc}")


@pytest.fixture(scope="session")
def samples_dir(gmat_available: None) -> Path:
    """Path to the GMAT install's ``samples/`` directory."""
    return locate_gmat().root / "samples"


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Where the committed integration golden CSVs live."""
    return Path(__file__).parent / "golden"
