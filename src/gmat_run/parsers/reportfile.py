"""Parse a GMAT ``ReportFile`` output into a :class:`pandas.DataFrame`.

GMAT writes ``ReportFile`` as whitespace-aligned text: a header row whose tokens
are resource-qualified field names (e.g. ``Sat.Earth.SMA``), followed by data
rows with the same column layout. Column separators are runs of two or more
whitespace characters; values never contain two consecutive whitespace
characters, so a ``\\s{2,}`` split cleanly handles both the numeric columns and
the space-bearing ``UTCGregorian`` epoch format
(``26 Nov 2026 12:00:00.000``).

GMAT re-emits the header line at mission-sequence segment boundaries; those
repeats are skipped. Duplicate *data* rows are preserved — a faithful
round-trip is part of the acceptance contract, and distinct report events at
the same epoch are legitimate.

Epoch columns are intentionally not promoted here — ``UTCGregorian`` stays
``str``, and ``*ModJulian`` stays ``float`` — because epoch detection belongs to
the datetime-parsing layer (see issue #7).
"""

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from gmat_run.errors import GmatOutputParseError

__all__ = ["parse"]

# Two-or-more whitespace: the column separator GMAT uses. A single space can
# appear *inside* a value (``UTCGregorian``), so \s+ would shred epoch columns.
_COLUMN_SEP = re.compile(r"\s{2,}")


def parse(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Parse a GMAT ``ReportFile`` into a :class:`pandas.DataFrame`.

    Column names are taken verbatim from the header row (dots preserved, e.g.
    ``Sat.Earth.SMA``). Each column is coerced to ``int64`` or ``float64`` if
    every value parses as numeric; otherwise the column stays ``object``/``str``.

    Args:
        path: Path to the ``ReportFile`` on disk.

    Returns:
        A DataFrame with one row per report event and one column per header
        field, in header order.

    Raises:
        GmatOutputParseError: The file is empty, or a data row's column count
            does not match the header.
    """
    path = Path(path)

    # utf-8-sig strips an optional UTF-8 BOM; newline=None gives universal-newline
    # translation so CRLF and LF files produce identical splits.
    with path.open(encoding="utf-8-sig", newline=None) as fh:
        lines = fh.read().splitlines()

    header_lineno, header_line = _find_header(lines, path)
    header_stripped = header_line.strip()
    column_names = _COLUMN_SEP.split(header_stripped)

    rows: list[list[str]] = []
    for offset, raw_line in enumerate(lines[header_lineno:], start=header_lineno + 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped == header_stripped:
            continue
        tokens = _COLUMN_SEP.split(stripped)
        if len(tokens) != len(column_names):
            raise GmatOutputParseError(
                f"expected {len(column_names)} columns, got {len(tokens)} on line {offset}",
                path,
            )
        rows.append(tokens)

    df = pd.DataFrame(rows, columns=column_names)
    for column in df.columns:
        df[column] = _coerce_numeric(df[column])
    return df


def _find_header(lines: list[str], path: Path) -> tuple[int, str]:
    """Return ``(index, line)`` of the first non-blank line in ``lines``.

    Index is 0-based over the file so the caller can slice ``lines[index:]``
    and enumerate from there.
    """
    for index, line in enumerate(lines):
        if line.strip():
            return index, line
    raise GmatOutputParseError("file is empty", path)


def _coerce_numeric(series: "pd.Series[Any]") -> "pd.Series[Any]":
    """Return ``series`` as numeric if every value parses; otherwise unchanged.

    ``pd.to_numeric`` picks the narrowest dtype that fits: ``int64`` when every
    value is an integer literal, ``float64`` otherwise (including any scientific
    notation or decimal point). One non-numeric value leaves the column as
    strings.
    """
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series
