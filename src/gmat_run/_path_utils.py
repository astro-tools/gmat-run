"""Path-normalisation helper shared across the public path-shaped API.

Every public surface that accepts a filesystem path argument funnels it
through :func:`resolve_user_path` so callers see the same behaviour
everywhere: ``~`` is expanded, relative paths are resolved against the
caller's CWD at submit time, and absolute paths are passed through
unchanged. The boundary normalisation kills GMAT's install-time
``OUTPUT_PATH``-relative footgun for workspace arguments and gives stored
path attributes (:attr:`Mission.script_path`, :attr:`Results.output_dir`,
``write_oem`` return values) a canonical absolute form.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_user_path(path: str | os.PathLike[str]) -> Path:
    """Return ``Path(path)`` with ``~`` expanded and resolved to absolute.

    Always returns an absolute :class:`~pathlib.Path`. ``.resolve()`` on an
    already-absolute path is a no-op, so callers who pre-resolved are
    unaffected. Symlinks are resolved as a side effect — that's the
    documented contract of :meth:`Path.resolve`.
    """
    return Path(path).expanduser().resolve()
