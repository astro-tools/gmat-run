"""Locate and validate a local GMAT install.

The single public entry point is :func:`locate_gmat`, which returns a
:class:`GmatInstall` describing the resolved install or raises
:class:`~gmat_run.errors.GmatNotFoundError` listing every location it checked.
"""

import glob
import os
import re
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from gmat_run.errors import GmatNotFoundError

__all__ = ["GmatInstall", "locate_gmat"]


# Matches GMAT release tags such as "R2026a" or "R2020a".
_VERSION_RE = re.compile(r"R\d{4}[a-z]")

# Executable names probed on PATH, covering Linux/macOS and Windows installs.
# shutil.which adds the .exe suffix on Windows automatically.
_PATH_BINARY_NAMES: tuple[str, ...] = ("GmatConsole", "GMAT", "gmat")


@dataclass(frozen=True, slots=True)
class GmatInstall:
    """A resolved GMAT install on the local filesystem."""

    root: Path
    bin_dir: Path
    api_dir: Path
    output_dir: Path
    version: str | None


def locate_gmat(gmat_root: str | os.PathLike[str] | None = None) -> GmatInstall:
    """Find a usable GMAT install on the local machine.

    Search order:

    1. The ``gmat_root`` argument, if provided.
    2. The ``GMAT_ROOT`` environment variable, if set.
    3. Platform-standard install locations (Windows ``Program Files``,
       Linux ``~/gmat-*`` and ``/opt/gmat-*``, macOS ``/Applications``).
    4. The ``PATH``, by locating known GMAT executables.

    The first valid candidate wins. An explicit override (argument or environment
    variable) does **not** fall through to later steps if its candidate is invalid —
    user-specified locations fail loudly.

    When multiple platform-glob candidates exist, the lexically greatest basename
    wins (e.g. ``gmat-R2026a`` beats ``gmat-R2024a``).

    Raises:
        GmatNotFoundError: No valid GMAT install was found in any search location.
    """
    attempts: list[tuple[str, Path | None, str]] = []

    if gmat_root is not None:
        path = Path(gmat_root).expanduser()
        result = _validate_install(path)
        if isinstance(result, GmatInstall):
            return result
        attempts.append(("gmat_root argument", path, result))
        raise GmatNotFoundError(attempts)

    env_root = os.environ.get("GMAT_ROOT")
    if env_root:
        path = Path(env_root).expanduser()
        result = _validate_install(path)
        if isinstance(result, GmatInstall):
            return result
        attempts.append(("GMAT_ROOT env var", path, result))
        raise GmatNotFoundError(attempts)
    attempts.append(("GMAT_ROOT env var", None, "not set"))

    glob_candidates = sorted(_platform_install_globs(), key=lambda p: p.name, reverse=True)
    if not glob_candidates:
        attempts.append(("platform install paths", None, "no matches"))
    else:
        for path in glob_candidates:
            result = _validate_install(path)
            if isinstance(result, GmatInstall):
                return result
            attempts.append(("platform install paths", path, result))

    path_candidates = list(_path_install_candidates())
    if not path_candidates:
        attempts.append(("PATH", None, "no GMAT executable found on PATH"))
    else:
        for path in path_candidates:
            result = _validate_install(path)
            if isinstance(result, GmatInstall):
                return result
            attempts.append(("PATH", path, result))

    raise GmatNotFoundError(attempts)


def _validate_install(root: Path) -> GmatInstall | str:
    """Validate ``root`` as a GMAT install.

    Returns a :class:`GmatInstall` on success or a human-readable rejection reason.
    Structural checks only — never imports ``gmatpy`` and never executes anything.
    """
    if not root.exists():
        return "does not exist"
    if not root.is_dir():
        return "not a directory"

    bin_dir = root / "bin"
    if not bin_dir.is_dir():
        return "missing bin/ directory"
    if not (bin_dir / "gmatpy").is_dir():
        return "missing bin/gmatpy/ directory"

    api_dir = root / "api"
    if not (api_dir / "load_gmat.py").is_file():
        return "missing api/load_gmat.py"

    return GmatInstall(
        root=root,
        bin_dir=bin_dir,
        api_dir=api_dir,
        output_dir=root / "output",
        version=_detect_version(root),
    )


def _detect_version(root: Path) -> str | None:
    """Look for a GMAT release tag in the startup file, falling back to the README."""
    for relative in ("bin/gmat_startup_file.txt", "README.txt"):
        path = root / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = _VERSION_RE.search(text)
        if match:
            return match.group(0)
    return None


def _platform_install_globs() -> list[Path]:
    """Expand platform-standard GMAT install glob patterns into existing paths."""
    paths: list[Path] = []
    for pattern in _glob_patterns_for_platform(sys.platform):
        for hit in glob.glob(os.path.expanduser(pattern)):
            paths.append(Path(hit))
    return paths


def _glob_patterns_for_platform(platform: str) -> list[str]:
    """Return the per-platform install-path glob patterns.

    Split out so tests can monkeypatch it without touching ``sys.platform``.
    """
    if platform == "win32":
        return [r"C:\Program Files\GMAT*", r"C:\Program Files (x86)\GMAT*"]
    if platform == "darwin":
        return ["/Applications/GMAT*"]
    return ["~/gmat-*", "/opt/gmat-*"]


def _path_install_candidates() -> Iterator[Path]:
    """Find candidate install roots by locating GMAT executables on PATH.

    Assumes the binary lives at ``<root>/bin/<name>``, which holds for every GMAT
    distribution shipped to date.
    """
    seen: set[Path] = set()
    for name in _PATH_BINARY_NAMES:
        binary = shutil.which(name)
        if binary is None:
            continue
        root = Path(binary).resolve().parent.parent
        if root not in seen:
            seen.add(root)
            yield root
