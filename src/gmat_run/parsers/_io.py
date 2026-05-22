"""Shared file-reading helper for the output-file parsers.

Every parser in this package opens its input the same way — UTF-8 with an
optional BOM stripped and universal-newline translation — and must surface a
read failure as a typed :class:`~gmat_run.errors.GmatOutputParseError` rather
than leaking a raw ``OSError`` / ``UnicodeDecodeError`` to the caller.
"""

from __future__ import annotations

from pathlib import Path

from gmat_run.errors import GmatOutputParseError

__all__ = ["read_text_lines"]


def read_text_lines(path: Path) -> list[str]:
    """Read ``path`` as UTF-8 text and return its universal-newline-split lines.

    ``utf-8-sig`` strips an optional BOM; ``newline=None`` gives universal-
    newline translation so CRLF and LF inputs split identically.

    Raises:
        GmatOutputParseError: ``path`` is missing or unreadable, or its bytes
            are not valid UTF-8 text (e.g. a binary ephemeris). The underlying
            ``OSError`` / ``UnicodeDecodeError`` is wrapped so callers see the
            typed error the rest of the parser surface raises, never a raw one.
    """
    try:
        with path.open(encoding="utf-8-sig", newline=None) as fh:
            return fh.read().splitlines()
    except OSError as exc:
        raise GmatOutputParseError(f"could not read '{path}': {exc}", path) from exc
    except UnicodeDecodeError as exc:
        raise GmatOutputParseError(
            f"'{path}' is not valid UTF-8 text (a binary file?): {exc}", path
        ) from exc
