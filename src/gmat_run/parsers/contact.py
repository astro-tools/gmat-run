"""Parse a GMAT ``ContactLocator`` output file into a :class:`pandas.DataFrame`.

GMAT's ``ContactLocator.ReportFormat`` field selects one of six text formats.
The structure splits cleanly in two:

* ``Legacy`` (the default in stock R2026a samples) emits per-observer blocks:
  a ``Target:`` line, then for each observer a ``Observer: <name>`` header,
  a ``Start Time / Stop Time / Duration`` table, a blank gap, and a
  ``Number of events : N`` total. Time format is always
  ``UTCGregorian`` and the column header reads ``(UTC)``.
* The five non-Legacy formats (``ContactRangeReport``,
  ``SiteViewMaxElevationReport``, ``SiteViewMaxElevationRangeReport``,
  ``AzimuthElevationRangeReport``, ``AzimuthElevationRangeRangeRateReport``)
  share a single tabular layout: a ``Target:`` line, a column-header row,
  a row of ``---`` dashes, then chronologically-merged data rows. Time format
  is one of ``UTCGregorian`` (header ``(UTCGregorian)``), ``UTCMJD``
  (numeric, header ``(UTC-MJD)``), or ``ISOYD`` (header ``(ISO-YD)``,
  e.g. ``2010-009T16:00:00.000``).

The two ``AzimuthElevationRange*`` variants emit one row per
``IntervalStepSize`` tick within each pass, with a leading ``Pass Number``
column that groups ticks. Their grain is per (Pass, tick); every other
variant is per event. All variants resolve to a single :class:`pandas.DataFrame`
with an ``Observer`` column — analyses iterate via ``df.groupby("Observer")``
or ``df.groupby(["Observer", "Pass"])``.

Time columns are eagerly parsed to ``datetime64[ns]`` (``promote_epochs`` is
not used — its ten-suffix table does not include ``ISO-YD``); ``Duration``
is ``timedelta64[ns]``; numeric value columns are ``float64``; ``Pass`` is
``int64``; ``Observer`` is ``object``. ``df.attrs`` carries ``target``,
``report_format``, ``time_scale``, ``epoch_scales``, and ``event_counts``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd

from gmat_run.errors import GmatOutputParseError

__all__ = ["parse"]


# Two-or-more whitespace: column separator on every variant. A single space
# can sit inside a value (``UTCGregorian`` → ``"09 Jan 2010 16:00:00.000"``),
# so ``\s+`` would shred those columns.
_COLUMN_SEP: Final = re.compile(r"\s{2,}")

# A line of dashes (with optional inline whitespace) separates header from
# data in every non-Legacy variant. Tokenised, each token is all-dashes.
_DASH_TOKEN: Final = re.compile(r"^-+$")

# GMAT Gregorian: ``09 Jan 2010 16:00:00.000``. Locale-sensitive ``%b``;
# C/en_US is the assumed locale on every CI runner and dev workstation,
# matching the existing reportfile / ephemeris parsers.
_GREGORIAN_FORMAT: Final = "%d %b %Y %H:%M:%S.%f"
# ISO ordinal date: ``2010-009T16:00:00.000`` (year + day-of-year).
_ISOYD_FORMAT: Final = "%Y-%jT%H:%M:%S.%f"
# GMAT's MJD epoch: JD 2430000.0 = 1941-01-05 12:00:00 UTC. Same constant
# as ``parsers.epoch._GMAT_MJD_EPOCH``.
_GMAT_MJD_EPOCH: Final = pd.Timestamp("1941-01-05 12:00:00")


# Time scale is always UTC for ``ContactLocator`` regardless of variant —
# Legacy is locked there per the docs, and the non-Legacy ``ReportTimeFormat``
# enum (``UTCGregorian / UTCMJD / ISOYD``) admits only UTC values.
_TIME_SCALE: Final = "UTC"


# Header parens-token → (format-kind, ``pandas.to_datetime`` strategy).
# Format kinds drive the per-column converter selection below.
_TIME_TOKEN_TO_KIND: Final[dict[str, str]] = {
    "UTC": "gregorian",  # Legacy
    "UTCGregorian": "gregorian",
    "UTC-MJD": "modjulian",
    "ISO-YD": "isoyd",
}


# Canonical column metadata. Keyed by the **base** header token (without the
# ``(…)`` parens suffix). Each entry names the column we surface and the
# kind of value coercion to apply.
@dataclass(frozen=True)
class _Column:
    """Schema entry for one column of a non-Legacy tabular variant."""

    name: str
    kind: str  # "str" | "int" | "float" | "duration" | "time"


_TABULAR_COLUMNS: Final[dict[str, _Column]] = {
    "Observer": _Column("Observer", "str"),
    "Pass Number": _Column("Pass", "int"),
    "Duration": _Column("Duration", "duration"),
    "Start Time": _Column("Start", "time"),
    "Stop Time": _Column("Stop", "time"),
    "Time": _Column("Time", "time"),
    "Start Range": _Column("StartRange", "float"),
    "Stop Range": _Column("StopRange", "float"),
    "Range": _Column("Range", "float"),
    "Range Rate": _Column("RangeRate", "float"),
    "Azimuth": _Column("Azimuth", "float"),
    "Elevation": _Column("Elevation", "float"),
    "Maximum Elevation": _Column("MaxElevation", "float"),
    "Max Elevation Time": _Column("MaxElevationTime", "time"),
}


# Column-name fingerprint → ReportFormat. Built once from the canonical names
# in ``_TABULAR_COLUMNS``; column order within the file does not affect the
# match. Legacy is detected structurally, not by fingerprint.
_FORMAT_FINGERPRINTS: Final[dict[frozenset[str], str]] = {
    frozenset(["Observer", "Duration", "Start", "Stop", "StartRange", "StopRange"]): (
        "ContactRangeReport"
    ),
    frozenset(
        ["Observer", "Start", "Stop", "Duration", "MaxElevation", "MaxElevationTime"]
    ): "SiteViewMaxElevationReport",
    frozenset(
        [
            "Observer",
            "Start",
            "Stop",
            "Duration",
            "MaxElevation",
            "MaxElevationTime",
            "StartRange",
            "StopRange",
        ]
    ): "SiteViewMaxElevationRangeReport",
    frozenset(["Pass", "Observer", "Time", "Azimuth", "Elevation", "Range"]): (
        "AzimuthElevationRangeReport"
    ),
    frozenset(
        ["Pass", "Observer", "Time", "Azimuth", "Elevation", "Range", "RangeRate"]
    ): "AzimuthElevationRangeRangeRateReport",
}


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def parse(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Parse a GMAT ``ContactLocator`` report into a :class:`pandas.DataFrame`.

    The returned schema depends on the underlying ``ContactLocator.ReportFormat``:

    * ``Legacy`` → ``Observer``, ``Start``, ``Stop``, ``Duration``.
    * ``ContactRangeReport`` → ``Observer``, ``Duration``, ``Start``, ``Stop``,
      ``StartRange``, ``StopRange``.
    * ``SiteViewMaxElevationReport`` → ``Observer``, ``Start``, ``Stop``,
      ``Duration``, ``MaxElevation``, ``MaxElevationTime``.
    * ``SiteViewMaxElevationRangeReport`` → ``Observer``, ``Start``, ``Stop``,
      ``Duration``, ``MaxElevation``, ``MaxElevationTime``, ``StartRange``,
      ``StopRange``.
    * ``AzimuthElevationRangeReport`` → ``Pass``, ``Observer``, ``Time``,
      ``Azimuth``, ``Elevation``, ``Range``.
    * ``AzimuthElevationRangeRangeRateReport`` → as above plus ``RangeRate``.

    Time columns are ``datetime64[ns]``; ``Duration`` is ``timedelta64[ns]``;
    ``Pass`` is ``int64``; ``Observer`` is ``object``; numeric value columns
    (ranges, angles, rates) are ``float64``.

    ``df.attrs``:

    * ``target`` — single ``str``, the ``Target`` resource name from the file.
    * ``report_format`` — variant name as GMAT spells it (e.g. ``"Legacy"``).
    * ``time_scale`` — always ``"UTC"`` in current GMAT releases.
    * ``epoch_scales`` — ``{datetime_column: "UTC"}`` for every parsed time
      column.
    * ``event_counts`` — ``{observer: int}``. Lifted from Legacy's
      ``Number of events : N`` lines; derived for the per-event variants via
      a row-count groupby on ``Observer``; derived for the AzEl variants via a
      ``Pass`` ``nunique`` per observer.

    Args:
        path: Path to the ``ContactLocator`` text report on disk.

    Returns:
        A DataFrame with one row per event (or per (Pass, tick) for the AzEl
        variants), in source order.

    Raises:
        GmatOutputParseError: The file is empty, missing a ``Target:`` line,
            an empty Legacy file (no ``Observer:`` blocks), the column-header
            fingerprint matches no known ``ReportFormat``, a data row has the
            wrong column count, an epoch value is malformed, or a numeric value
            does not parse.
    """
    path = Path(path)

    # ``utf-8-sig`` strips an optional UTF-8 BOM; ``newline=None`` activates
    # universal-newline translation so CRLF and LF files produce identical
    # token streams.
    with path.open(encoding="utf-8-sig", newline=None) as fh:
        lines = fh.read().splitlines()

    if not any(line.strip() for line in lines):
        raise GmatOutputParseError("file is empty", path)

    target, body_lines = _read_target(lines, path)

    # Legacy is the only variant whose first content line after ``Target:``
    # begins with ``Observer:``. Everything else opens with the column header.
    if _is_legacy(body_lines):
        df = _parse_legacy(body_lines, path)
    else:
        df = _parse_tabular(body_lines, path)

    df.attrs["target"] = target
    df.attrs["time_scale"] = _TIME_SCALE
    return df


# ----------------------------------------------------------------------------
# Shared front-matter parsing
# ----------------------------------------------------------------------------


def _read_target(lines: list[str], path: Path) -> tuple[str, list[str]]:
    """Consume the leading ``Target: <name>`` line; return ``(name, remainder)``.

    The remainder is the slice of ``lines`` that begins immediately after the
    ``Target:`` line — leading blank lines are kept so downstream logic can
    decide whether to skip them.
    """
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        if not stripped.startswith("Target:"):
            raise GmatOutputParseError(
                f"line {index + 1}: expected 'Target: <name>', got {stripped!r}",
                path,
            )
        target = stripped[len("Target:") :].strip()
        if not target:
            raise GmatOutputParseError(
                f"line {index + 1}: 'Target:' is missing a target name",
                path,
            )
        return target, lines[index + 1 :]
    raise GmatOutputParseError(  # pragma: no cover — guarded by parse()'s empty check
        "missing 'Target:' line", path
    )


def _is_legacy(body: list[str]) -> bool:
    """Return ``True`` when the first non-blank body line is ``Observer: …``.

    Legacy is the only variant that splits the file into per-observer blocks;
    the other five drop straight into the column header. Detection is purely
    structural — the column-name fingerprint is consulted only once we know we
    are in the tabular path.
    """
    for raw in body:
        stripped = raw.strip()
        if not stripped:
            continue
        return stripped.startswith("Observer:")
    return False


# ----------------------------------------------------------------------------
# Legacy parser
# ----------------------------------------------------------------------------


@dataclass
class _LegacyBlock:
    """Per-observer block accumulator for the Legacy format."""

    observer: str
    rows: list[tuple[str, str, str]] = field(default_factory=list)
    declared_count: int | None = None


def _parse_legacy(body: list[str], path: Path) -> pd.DataFrame:
    """Walk the per-observer blocks of a Legacy file into a tall DataFrame."""
    blocks: list[_LegacyBlock] = []
    current: _LegacyBlock | None = None
    seen_header = False

    for index, raw in enumerate(body, start=1):
        stripped = raw.strip()
        if not stripped:
            continue

        if stripped.startswith("Observer:"):
            observer = stripped[len("Observer:") :].strip()
            if not observer:
                raise GmatOutputParseError(f"line {index}: 'Observer:' is missing a name", path)
            current = _LegacyBlock(observer=observer)
            blocks.append(current)
            seen_header = False
            continue

        if current is None:
            raise GmatOutputParseError(
                f"line {index}: content before first 'Observer:' line: {stripped!r}",
                path,
            )

        if stripped.startswith("Number of events"):
            current.declared_count = _parse_count(stripped, index, path)
            current = None
            continue

        if not seen_header:
            # Only structural check on the header — Legacy has no fingerprint.
            seen_header = True
            continue

        tokens = _COLUMN_SEP.split(stripped)
        if len(tokens) != 3:
            raise GmatOutputParseError(
                f"line {index}: expected 3 columns (Start, Stop, Duration), "
                f"got {len(tokens)}: {stripped!r}",
                path,
            )
        current.rows.append((tokens[0], tokens[1], tokens[2]))

    if not blocks:  # pragma: no cover — guarded by _is_legacy precondition
        raise GmatOutputParseError("no 'Observer:' blocks in Legacy file", path)

    observers = [block.observer for block in blocks]
    starts: list[str] = []
    stops: list[str] = []
    durations: list[str] = []
    observer_col: list[str] = []
    for block in blocks:
        if block.declared_count is not None and block.declared_count != len(block.rows):
            raise GmatOutputParseError(
                f"Observer {block.observer!r}: declared {block.declared_count} "
                f"events but parsed {len(block.rows)}",
                path,
            )
        for start, stop, duration in block.rows:
            observer_col.append(block.observer)
            starts.append(start)
            stops.append(stop)
            durations.append(duration)

    df = pd.DataFrame(
        {
            "Observer": pd.Series(observer_col, dtype="object", name="Observer"),
            "Start": _convert_time_column(starts, "gregorian", "Start", path),
            "Stop": _convert_time_column(stops, "gregorian", "Stop", path),
            "Duration": _convert_duration(durations, "Duration", path),
        }
    )

    df.attrs["report_format"] = "Legacy"
    df.attrs["epoch_scales"] = {"Start": _TIME_SCALE, "Stop": _TIME_SCALE}
    df.attrs["event_counts"] = {
        block.observer: block.declared_count
        if block.declared_count is not None
        else len(block.rows)
        for block in blocks
    }
    df.attrs["observers"] = tuple(observers)
    return df


def _parse_count(line: str, lineno: int, path: Path) -> int:
    """Parse ``Number of events : N`` to ``int``."""
    _, _, tail = line.partition(":")
    text = tail.strip()
    try:
        return int(text)
    except ValueError as exc:
        raise GmatOutputParseError(
            f"line {lineno}: malformed 'Number of events' value {text!r}", path
        ) from exc


# ----------------------------------------------------------------------------
# Tabular parser (the five non-Legacy variants)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _TabularSchema:
    """Per-variant column layout resolved from the file's header."""

    report_format: str
    columns: tuple[_Column, ...]
    # Per-time-column format kind ("gregorian" / "modjulian" / "isoyd"),
    # keyed by canonical column name.
    time_kinds: dict[str, str]


def _parse_tabular(body: list[str], path: Path) -> pd.DataFrame:
    """Parse one of the five tabular variants into a DataFrame."""
    schema, data_rows = _read_tabular_header(body, path)

    columns_by_name: list[list[str]] = [[] for _ in schema.columns]
    expected_token_count = len(schema.columns)
    for lineno, line in data_rows:
        tokens = _COLUMN_SEP.split(line.strip())
        if len(tokens) != expected_token_count:
            raise GmatOutputParseError(
                f"line {lineno}: expected {expected_token_count} columns, "
                f"got {len(tokens)}: {line.strip()!r}",
                path,
            )
        for column_idx, token in enumerate(tokens):
            columns_by_name[column_idx].append(token)

    # Build typed columns one by one. ``DataFrame({col: series})`` would also
    # work; doing it column-by-column keeps the per-column failure case cleanly
    # attributed to the offending column name.
    df_dict: dict[str, pd.Series] = {}
    for column, raw_values in zip(schema.columns, columns_by_name, strict=True):
        df_dict[column.name] = _coerce_column(column, raw_values, schema, path)

    df = pd.DataFrame(df_dict)

    df.attrs["report_format"] = schema.report_format
    df.attrs["epoch_scales"] = {
        column.name: _TIME_SCALE for column in schema.columns if column.kind == "time"
    }
    df.attrs["observers"] = tuple(_unique_in_order(df["Observer"].tolist()))
    df.attrs["event_counts"] = _derive_event_counts(df, schema)
    return df


def _read_tabular_header(
    body: list[str], path: Path
) -> tuple[_TabularSchema, list[tuple[int, str]]]:
    """Consume the column header + dashes line; return ``(schema, data_rows)``.

    ``data_rows`` is a list of ``(lineno, raw_line)`` for downstream parsing —
    blank lines between Pass groups (AzEl variants) are stripped at this layer
    so the data parser does not need a state machine.
    """
    iterator = iter(enumerate(body, start=1))

    header_line: str | None = None
    for lineno, raw in iterator:
        stripped = raw.strip()
        if not stripped:
            continue
        header_line = stripped
        header_lineno = lineno
        break
    if header_line is None:
        raise GmatOutputParseError("missing column header line", path)

    header_tokens = _COLUMN_SEP.split(header_line)
    schema = _resolve_schema(header_tokens, header_lineno, path)

    # Dash-divider line: must follow the header on every non-Legacy variant we
    # have seen. Tolerate intervening blanks just in case.
    saw_divider = False
    for lineno, raw in iterator:
        stripped = raw.strip()
        if not stripped:
            continue
        tokens = _COLUMN_SEP.split(stripped)
        if all(_DASH_TOKEN.match(token) for token in tokens):
            if len(tokens) != len(schema.columns):
                raise GmatOutputParseError(
                    f"line {lineno}: dashes divider has {len(tokens)} tokens, "
                    f"expected {len(schema.columns)}",
                    path,
                )
            saw_divider = True
            break
        raise GmatOutputParseError(
            f"line {lineno}: expected '---' divider after header, got {stripped!r}",
            path,
        )
    if not saw_divider:
        raise GmatOutputParseError("missing '---' divider after header", path)

    data_rows: list[tuple[int, str]] = []
    for lineno, raw in iterator:
        stripped = raw.strip()
        if not stripped:
            continue
        data_rows.append((lineno, stripped))
    return schema, data_rows


def _resolve_schema(header_tokens: list[str], lineno: int, path: Path) -> _TabularSchema:
    """Map raw header tokens to a :class:`_TabularSchema` via fingerprint."""
    columns: list[_Column] = []
    time_kinds: dict[str, str] = {}
    for token in header_tokens:
        base, paren = _split_header_token(token)
        meta = _TABULAR_COLUMNS.get(base)
        if meta is None:
            raise GmatOutputParseError(f"line {lineno}: unknown column header {token!r}", path)
        columns.append(meta)
        if meta.kind == "time":
            if paren is None:
                raise GmatOutputParseError(
                    f"line {lineno}: time column {token!r} has no '(scale)' suffix",
                    path,
                )
            kind = _TIME_TOKEN_TO_KIND.get(paren)
            if kind is None:
                raise GmatOutputParseError(
                    f"line {lineno}: unrecognised time format token "
                    f"({paren!r}) in column {token!r}; supported: "
                    f"{sorted(_TIME_TOKEN_TO_KIND)}",
                    path,
                )
            time_kinds[meta.name] = kind

    fingerprint = frozenset(column.name for column in columns)
    report_format = _FORMAT_FINGERPRINTS.get(fingerprint)
    if report_format is None:
        raise GmatOutputParseError(
            f"line {lineno}: column set {sorted(fingerprint)} matches no "
            f"known ContactLocator ReportFormat",
            path,
        )
    return _TabularSchema(
        report_format=report_format,
        columns=tuple(columns),
        time_kinds=time_kinds,
    )


def _split_header_token(token: str) -> tuple[str, str | None]:
    """Split ``"Start Time (UTCGregorian)"`` into ``("Start Time", "UTCGregorian")``.

    Tokens without a parenthesised suffix return ``(token, None)``.
    """
    match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", token)
    if match is None:
        return token.strip(), None
    return match.group(1).strip(), match.group(2).strip()


# ----------------------------------------------------------------------------
# Per-column coercion
# ----------------------------------------------------------------------------


def _coerce_column(
    column: _Column,
    raw_values: list[str],
    schema: _TabularSchema,
    path: Path,
) -> pd.Series:
    if column.kind == "str":
        return pd.Series(raw_values, dtype="object", name=column.name)
    if column.kind == "int":
        try:
            ints = [int(v) for v in raw_values]
        except ValueError as exc:
            raise GmatOutputParseError(
                f"column {column.name!r}: non-integer value: {exc}", path
            ) from exc
        return pd.Series(ints, dtype="int64", name=column.name)
    if column.kind == "float":
        return _convert_float(raw_values, column.name, path)
    if column.kind == "duration":
        return _convert_duration(raw_values, column.name, path)
    if column.kind == "time":
        time_kind = schema.time_kinds[column.name]
        return _convert_time_column(raw_values, time_kind, column.name, path)
    raise GmatOutputParseError(  # pragma: no cover — exhaustive on _Column.kind
        f"internal error: unhandled column kind {column.kind!r}", path
    )


def _convert_float(raw_values: list[str], column_name: str, path: Path) -> pd.Series:
    try:
        series = pd.to_numeric(pd.Series(raw_values, name=column_name)).astype("float64")
    except (ValueError, TypeError) as exc:
        raise GmatOutputParseError(
            f"column {column_name!r}: non-numeric value: {exc}", path
        ) from exc
    return series


def _convert_duration(raw_values: list[str], column_name: str, path: Path) -> pd.Series:
    try:
        seconds = pd.to_numeric(pd.Series(raw_values, dtype="object"))
    except (ValueError, TypeError) as exc:
        raise GmatOutputParseError(
            f"column {column_name!r}: non-numeric seconds: {exc}", path
        ) from exc
    return pd.Series(pd.to_timedelta(seconds, unit="s"), name=column_name).astype("timedelta64[ns]")


def _convert_time_column(
    raw_values: list[str], kind: str, column_name: str, path: Path
) -> pd.Series:
    if kind == "gregorian":
        try:
            parsed = pd.to_datetime(raw_values, format=_GREGORIAN_FORMAT)
        except (ValueError, TypeError) as exc:
            raise GmatOutputParseError(
                f"column {column_name!r}: malformed Gregorian epoch: {exc}", path
            ) from exc
        return pd.Series(parsed, name=column_name).astype("datetime64[ns]")
    if kind == "isoyd":
        try:
            parsed = pd.to_datetime(raw_values, format=_ISOYD_FORMAT)
        except (ValueError, TypeError) as exc:
            raise GmatOutputParseError(
                f"column {column_name!r}: malformed ISO-YD epoch: {exc}", path
            ) from exc
        return pd.Series(parsed, name=column_name).astype("datetime64[ns]")
    if kind == "modjulian":
        try:
            mjd = pd.to_numeric(pd.Series(raw_values, dtype="object"))
        except (ValueError, TypeError) as exc:
            raise GmatOutputParseError(
                f"column {column_name!r}: non-numeric ModJulian value: {exc}", path
            ) from exc
        try:
            absolute = _GMAT_MJD_EPOCH + pd.to_timedelta(mjd, unit="D")
        except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime) as exc:
            raise GmatOutputParseError(
                f"column {column_name!r}: ModJulian value outside the "
                f"representable datetime64[ns] range (~1677..2262): {exc}",
                path,
            ) from exc
        return pd.Series(absolute, name=column_name).astype("datetime64[ns]")
    raise GmatOutputParseError(  # pragma: no cover — exhaustive on _TIME_TOKEN_TO_KIND
        f"column {column_name!r}: internal error: unhandled time kind {kind!r}",
        path,
    )


def _derive_event_counts(df: pd.DataFrame, schema: _TabularSchema) -> dict[str, int]:
    """``{observer: events}`` for the tabular variants.

    Per-event variants count one row per event. AzEl variants count one row
    per (Pass, tick), so collapse to ``Pass`` ``nunique`` per observer to
    recover the event count (= the number of distinct passes the observer saw).
    """
    if "Pass" in {column.name for column in schema.columns}:
        return {
            str(observer): int(passes)
            for observer, passes in df.groupby("Observer")["Pass"].nunique().items()
        }
    return {str(observer): int(count) for observer, count in df.groupby("Observer").size().items()}


def _unique_in_order(values: list[str]) -> list[str]:
    """Stable de-duplication preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
