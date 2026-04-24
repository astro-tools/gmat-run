"""Load gmatpy for a resolved GMAT install.

The single public entry point is :func:`bootstrap`, which takes a
:class:`~gmat_run.install.GmatInstall`, ensures the API startup file exists
(generating it on first use via GMAT's ``BuildApiStartupFile.py``), adds the
install's ``bin/`` directory to :data:`sys.path`, imports ``gmatpy``, calls
``gmat.Setup(...)`` on the API startup file, and returns the imported module.

Idempotent within a single interpreter. A second call requesting a different
install raises :class:`~gmat_run.errors.GmatLoadError` — gmatpy cannot be
cleanly reinitialised once loaded.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from types import ModuleType

from gmat_run.errors import GmatLoadError
from gmat_run.install import GmatInstall

__all__ = ["bootstrap"]


_bootstrapped: tuple[GmatInstall, ModuleType] | None = None


def bootstrap(install: GmatInstall) -> ModuleType:
    """Load gmatpy for ``install`` and return the imported module.

    On first call: ensures ``<install.bin_dir>/api_startup_file.txt`` exists
    (generating it via ``<install.api_dir>/BuildApiStartupFile.py`` if
    missing), prepends the bin directory to :data:`sys.path`, imports
    ``gmatpy``, and calls ``gmatpy.Setup(...)`` on the API startup file.

    Subsequent calls with the same install return the cached module handle.
    Calls with a different install raise — gmatpy cannot be reinitialised.

    Raises:
        GmatLoadError: gmatpy could not be imported for this install, the
            API startup file could not be generated, or a second install
            was requested.
    """
    global _bootstrapped

    if _bootstrapped is not None:
        cached_install, cached_module = _bootstrapped
        if cached_install == install:
            return cached_module
        raise GmatLoadError(
            "cannot bootstrap a second GMAT install in one interpreter; "
            f"gmatpy is already bound to {cached_install.root}"
        )

    _ensure_api_startup_file(install)

    bin_dir = str(install.bin_dir)
    if bin_dir not in sys.path:
        # insert(1, ...) matches GMAT's own load_gmat.py template, preserving
        # index 0 (typically the current directory) ahead of the install bin.
        sys.path.insert(1, bin_dir)

    try:
        gmat = importlib.import_module("gmatpy")
    except ImportError as exc:
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise GmatLoadError(
            f"could not import gmatpy from {install.bin_dir}; Python {py} "
            "may not be supported by this GMAT install"
        ) from exc

    startup_file = install.bin_dir / "api_startup_file.txt"
    gmat.Setup(str(startup_file))

    _bootstrapped = (install, gmat)
    return gmat


def _ensure_api_startup_file(install: GmatInstall) -> None:
    startup_file = install.bin_dir / "api_startup_file.txt"
    if startup_file.is_file():
        return

    script = install.api_dir / "BuildApiStartupFile.py"
    # BuildApiStartupFile.py writes api_startup_file.txt to bin/ via an
    # absolute path, so cwd does not affect the real output. When a debug
    # build exists, the script ALSO writes api_startup_file_d.txt via a
    # relative path — a throwaway cwd keeps that stray file out of the
    # caller's directory and the install tree.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(
                [sys.executable, str(script)],
                check=True,
                cwd=tmp,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise GmatLoadError(
                f"failed to generate {startup_file}: "
                f"{script.name} exited with code {exc.returncode}\n"
                f"{exc.stderr}"
            ) from exc
