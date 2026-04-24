"""Typed exception hierarchy for gmat-run."""

from collections.abc import Sequence
from pathlib import Path


class GmatRunError(Exception):
    """Base class for all gmat-run errors."""


class GmatNotFoundError(GmatRunError):
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
