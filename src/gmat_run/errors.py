"""Typed exception hierarchy for gmat-run.

Every exception the library raises inherits from :class:`GmatError`, so
downstream code can branch on failure mode without string-parsing GMAT's log
output. Leaf classes carry the payload relevant to their failure mode as
attributes (``attempts``, ``log``, ``path``, ``value``), never only as the
formatted message.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any


class GmatError(Exception):
    """Base class for every exception raised by gmat-run."""


class GmatNotFoundError(GmatError):
    """Raised when no usable GMAT install can be located.

    The ``attempts`` attribute records each search step that ran, the path that was
    probed (if any), and the reason it was rejected — so callers can present a full
    diagnostic without re-running discovery.
    """

    def __init__(self, attempts: Sequence[tuple[str, Path | None, str]]) -> None:
        self.attempts: tuple[tuple[str, Path | None, str], ...] = tuple(attempts)
        super().__init__(self._render(self.attempts))

    @staticmethod
    def _render(attempts: Sequence[tuple[str, Path | None, str]]) -> str:
        if not attempts:
            return "No GMAT install found and no search steps were attempted."
        lines = ["No usable GMAT install found. Search results:"]
        for step, path, reason in attempts:
            location = str(path) if path is not None else "(none)"
            lines.append(f"  - {step}: {location}: {reason}")
        return "\n".join(lines)


class GmatLoadError(GmatError):
    """Raised when gmatpy cannot be loaded for a resolved GMAT install.

    Covers unsupported Python versions (gmatpy ships per-Python-minor shared
    libraries), missing shared libraries, failed generation of the API startup
    file, and attempts to bootstrap a second install in one interpreter. The
    triggering ``ImportError`` or ``CalledProcessError`` is chained via
    ``__cause__``.
    """


class GmatRunError(GmatError):
    """Raised when a GMAT mission sequence fails at run time.

    The ``log`` attribute carries GMAT's own stderr / log content captured during
    the run, so callers can surface GMAT's diagnostic verbatim without re-reading
    the log file. The optional ``path`` attribute carries the offending directory
    or file when the failure is wired to a specific filesystem location (e.g. a
    non-writable ``working_dir`` or a colliding output file); ``None`` for engine
    failures that are not path-scoped.
    """

    def __init__(self, message: str, log: str, *, path: Path | None = None) -> None:
        self.log = log
        self.path = path
        super().__init__(message)


class GmatOutputParseError(GmatError):
    """Raised when a GMAT output file cannot be parsed into a DataFrame.

    The ``path`` attribute points to the offending file, so callers can inspect
    it or surface it in error messages without re-deriving the path.
    """

    def __init__(self, message: str, path: Path) -> None:
        self.path = path
        super().__init__(message)


class GmatFieldError(GmatError):
    """Raised when a dotted-path field access fails.

    Covers unknown paths, type mismatches on write, and unresolvable parent
    objects. The ``path`` attribute carries the offending dotted-path key; the
    ``value`` attribute carries the value the caller attempted to write (or
    ``None`` for a read), verbatim.
    """

    def __init__(self, message: str, path: str, value: Any = None) -> None:
        self.path = path
        self.value = value
        super().__init__(message)


class GmatTimeoutError(GmatRunError):
    """Raised when ``Mission.run(timeout=...)`` exceeds its wall-clock cap.

    A subclass of :class:`GmatRunError` so callers that already branch on
    "the run failed" keep working — narrower handlers can catch this first
    to distinguish a timeout from an engine-side failure.

    Carries the ``requested_timeout`` (the cap the caller asked for, in
    seconds) and ``elapsed`` (the wall-clock time the parent observed before
    killing the child). The inherited ``log`` carries any partial GMAT log
    content readable from the workspace at kill time, or ``""`` if nothing
    was salvageable. ``path`` is unset — a timeout is not wired to a
    specific filesystem location.
    """

    def __init__(
        self,
        message: str,
        log: str,
        *,
        requested_timeout: float,
        elapsed: float,
    ) -> None:
        self.requested_timeout = requested_timeout
        self.elapsed = elapsed
        super().__init__(message, log=log)
