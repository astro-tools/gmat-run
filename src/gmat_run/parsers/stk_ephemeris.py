"""Parse a GMAT ``EphemerisFile`` (STK-TimePosVel text format) into a DataFrame.

GMAT's second text-ephemeris writer (selected with
``EphemerisFile.FileFormat = STK-TimePosVel``) emits the legacy STK ephemeris
format: a one-line ``stk.v.X.Y`` version banner, optional ``# WrittenBy …``
comment lines, then a ``BEGIN Ephemeris`` block of whitespace-separated
``KEY VALUE`` metadata followed by an ``EphemerisTimePosVel`` data section
with seven columns per record (offset-from-epoch in seconds + X, Y, Z, VX,
VY, VZ), terminated by ``END Ephemeris``.

This module parses that single FileFormat. The CCSDS-OEM text flavour is
handled by :mod:`gmat_run.parsers.ephemeris`; SPK (binary), Code-500
(binary), and CCSDS-AEM (attitude-only) are explicitly out of scope.
``EphemerisTimePosVelAcc`` (with acceleration) and ``EphemerisTimePos``
(position-only) data-section variants are also rejected — surfacing the
format mismatch is more useful than guessing intent.

Records' offset-from-epoch is converted to absolute UTC datetimes by adding
the ``ScenarioEpoch`` value from the meta block. The ``.e`` file does not
carry an epoch-format declaration, so the parser assumes UTC: GMAT's default
``EpochFormat`` for the STK writer, and the only scale that gives the
caller a sensible default. Missions that pinned a non-UTC ``EpochFormat``
on the ``EphemerisFile`` resource will see ``Epoch`` values labelled
``"UTC"`` that are actually in the chosen scale; ``df.attrs["scenario_epoch"]``
preserves the raw text so consumers can re-interpret if needed.

Header variants observed across GMAT releases (older writers omit
``InterpolationMethod`` / ``InterpolationSamplesM1``, padding differs,
``DistanceUnit`` may or may not be present) are tolerated; only the
mandatory ``ScenarioEpoch`` and ``EphemerisTimePosVel`` markers are
required.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd

from gmat_run.errors import GmatOutputParseError

__all__ = ["parse"]

# GMAT's default ScenarioEpoch text shape, e.g. ``01 Jan 2000 11:59:28.000``.
# ``%b`` is locale sensitive; this matches the C/en_US locale active on every
# CI runner and dev workstation.
_GREGORIAN_FORMAT: Final = "%d %b %Y %H:%M:%S.%f"

# Mandatory state-vector columns per the STK ephemeris specification.
_STATE_COLUMNS: Final = ["X", "Y", "Z", "VX", "VY", "VZ"]

# Total record column count: offset-from-epoch + six state components.
_RECORD_COLUMNS: Final = 1 + len(_STATE_COLUMNS)

# Meta keys we surface as flat attrs when the file declares them. STK uses a
# whitespace-separated ``KEY VALUE`` convention, not the OEM ``KEY = VALUE``.
_META_TO_ATTR: Final = {
    "CentralBody": "central_body",
    "CoordinateSystem": "coordinate_system",
    "InterpolationMethod": "interpolation",
    "InterpolationSamplesM1": "interpolation_degree",
    "DistanceUnit": "distance_unit",
}

# Section markers.
_BEGIN_EPHEMERIS: Final = "BEGIN Ephemeris"
_END_EPHEMERIS: Final = "END Ephemeris"
_DATA_SECTION_HEADER: Final = "EphemerisTimePosVel"

# Other ``Ephemeris*`` data-section headers GMAT can emit. We reject these
# explicitly so the user gets a typed error rather than silent column-count
# confusion.
_UNSUPPORTED_DATA_SECTIONS: Final = (
    "EphemerisTimePos",
    "EphemerisTimePosVelAcc",
)

# stk.v.X.Y banner — the version number is captured but not validated.
_VERSION_BANNER_RE: Final = re.compile(r"^stk\.v\.\d+\.\d+\s*$")

# UTC is the only scale we infer for the absolute Epoch column. See module
# docstring for the rationale.
_DEFAULT_TIME_SCALE: Final = "UTC"


@dataclass
class _Header:
    """Pre-``BEGIN Ephemeris`` content: version banner plus comment lines."""

    version: str = ""
    comments: list[str] = field(default_factory=list)


def parse(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Parse an STK-TimePosVel ephemeris file into a :class:`pandas.DataFrame`.

    Args:
        path: Path to the ``.e`` file on disk.

    Returns:
        A DataFrame with one row per state record. Columns are ``Epoch`` (a
        ``datetime64[ns]`` column with the offset added to ``ScenarioEpoch``)
        plus ``X``, ``Y``, ``Z``, ``VX``, ``VY``, ``VZ`` (all ``float64``).

        Metadata surfaces on ``df.attrs``:

        * ``df.attrs["epoch_scales"] = {"Epoch": "UTC"}`` — see module
          docstring for the UTC default rationale.
        * Flat keys (``central_body``, ``coordinate_system``,
          ``interpolation``, ``interpolation_degree``, ``distance_unit``,
          ``time_scale``) for whichever meta keys the file declared.
        * ``df.attrs["scenario_epoch"]`` carries the raw ``ScenarioEpoch``
          text so callers can re-interpret if their mission used a non-UTC
          ``EpochFormat``.
        * ``df.attrs["file_header"]`` carries the version banner and any
          ``# WrittenBy …`` comment lines from above ``BEGIN Ephemeris``.

    Raises:
        GmatOutputParseError: The file is empty, missing a ``stk.v.X.Y``
            banner, missing ``BEGIN Ephemeris`` / ``ScenarioEpoch`` /
            ``EphemerisTimePosVel``, declares an unsupported data section
            (``EphemerisTimePos``, ``EphemerisTimePosVelAcc``), or contains a
            malformed meta line, record column count, offset, state value, or
            ``ScenarioEpoch``.
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline=None) as fh:
        lines = fh.read().splitlines()

    if not any(line.strip() for line in lines):
        raise GmatOutputParseError("file is empty", path)

    header, meta, rows, data_start_lineno = _split(lines, path)

    if "ScenarioEpoch" not in meta:
        raise GmatOutputParseError(
            "missing ScenarioEpoch in BEGIN Ephemeris meta block",
            path,
        )

    scenario_epoch_raw = meta["ScenarioEpoch"]
    scenario_epoch = _parse_scenario_epoch(scenario_epoch_raw, path)

    df = _records_to_frame(rows, scenario_epoch, data_start_lineno, path)

    df.attrs["epoch_scales"] = {"Epoch": _DEFAULT_TIME_SCALE}
    df.attrs["time_scale"] = _DEFAULT_TIME_SCALE
    df.attrs["scenario_epoch"] = scenario_epoch_raw
    for meta_key, attr_key in _META_TO_ATTR.items():
        if meta_key in meta:
            df.attrs[attr_key] = _coerce_attr(meta_key, meta[meta_key])

    df.attrs["file_header"] = {
        "version": header.version,
        "comments": list(header.comments),
    }
    return df


# --- file structure ---------------------------------------------------------


def _split(lines: list[str], path: Path) -> tuple[_Header, dict[str, str], list[list[str]], int]:
    """Walk ``lines`` once, returning header, meta, data rows, and data lineno.

    Pre-``BEGIN Ephemeris``: version banner + ``# …`` comments. Inside the
    ``BEGIN Ephemeris`` block but before ``EphemerisTimePosVel``: meta lines.
    After ``EphemerisTimePosVel`` and before ``END Ephemeris`` (or EOF): data
    rows.
    """
    header = _Header()
    meta: dict[str, str] = {}
    rows: list[list[str]] = []
    data_start_lineno = 0

    in_ephemeris = False
    in_data = False
    seen_end = False

    for index, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        if not in_ephemeris:
            # Pre-``BEGIN Ephemeris``: version banner first, then optional
            # ``# …`` comments. Anything else is a malformed header.
            if line == _BEGIN_EPHEMERIS:
                if not header.version:
                    raise GmatOutputParseError(
                        f"line {index}: BEGIN Ephemeris before stk.v.X.Y banner",
                        path,
                    )
                in_ephemeris = True
                continue
            if line.startswith("#"):
                header.comments.append(line)
                continue
            if _VERSION_BANNER_RE.match(line):
                if header.version:
                    raise GmatOutputParseError(f"line {index}: duplicate stk.v.X.Y banner", path)
                header.version = line
                continue
            raise GmatOutputParseError(
                f"line {index}: expected stk.v.X.Y banner, '# …' comment, or "
                f"'BEGIN Ephemeris', got {line!r}",
                path,
            )

        if seen_end:
            raise GmatOutputParseError(f"line {index}: content after END Ephemeris: {line!r}", path)

        if line == _END_EPHEMERIS:
            seen_end = True
            in_data = False
            continue

        if not in_data:
            # Inside the ephemeris block, before the data-section header:
            # meta ``KEY VALUE`` lines, or the data-section marker itself.
            if line == _DATA_SECTION_HEADER:
                in_data = True
                data_start_lineno = index + 1
                continue
            if line in _UNSUPPORTED_DATA_SECTIONS:
                raise GmatOutputParseError(
                    f"line {index}: unsupported data section {line!r}; "
                    "only EphemerisTimePosVel is parsed",
                    path,
                )
            key, value = _parse_meta_line(line, index, path)
            meta[key] = value
            continue

        # Data row.
        rows.append(_split_record(line, index, path))

    if not header.version:
        raise GmatOutputParseError("missing stk.v.X.Y version banner; not an STK ephemeris?", path)
    if not in_ephemeris:
        raise GmatOutputParseError("missing BEGIN Ephemeris block", path)
    if not data_start_lineno:
        raise GmatOutputParseError("missing EphemerisTimePosVel data section", path)

    return header, meta, rows, data_start_lineno


def _parse_meta_line(line: str, lineno: int, path: Path) -> tuple[str, str]:
    """Split a whitespace-separated ``KEY VALUE`` meta line.

    STK keys are alphanumeric tokens; values can contain spaces (the
    ``ScenarioEpoch`` Gregorian text does). Split on the first run of
    whitespace.
    """
    tokens = line.split(maxsplit=1)
    if len(tokens) != 2:
        raise GmatOutputParseError(f"line {lineno}: expected 'KEY VALUE', got {line!r}", path)
    key, value = tokens
    return key, value.strip()


def _split_record(line: str, lineno: int, path: Path) -> list[str]:
    """Tokenize a data record line; validate column count up front."""
    tokens = line.split()
    if len(tokens) != _RECORD_COLUMNS:
        raise GmatOutputParseError(
            f"line {lineno}: expected {_RECORD_COLUMNS} columns "
            f"(offset + 6 state components), got {len(tokens)}",
            path,
        )
    return tokens


# --- records → frame --------------------------------------------------------


def _records_to_frame(
    rows: list[list[str]],
    scenario_epoch: pd.Timestamp,
    data_start_lineno: int,
    path: Path,
) -> pd.DataFrame:
    """Convert offset-plus-state records into a typed DataFrame."""
    if not rows:
        return pd.DataFrame(
            {
                "Epoch": pd.Series(dtype="datetime64[ns]"),
                **{c: pd.Series(dtype="float64") for c in _STATE_COLUMNS},
            }
        )

    offsets_raw = [row[0] for row in rows]
    states = [row[1:] for row in rows]

    try:
        offsets = pd.to_numeric(pd.Series(offsets_raw))
    except (ValueError, TypeError) as exc:
        raise GmatOutputParseError(
            f"non-numeric offset in data section starting at line {data_start_lineno}: {exc}",
            path,
        ) from exc
    # Force the addition result to datetime64[ns] explicitly so the column
    # dtype is stable across pandas releases.
    epochs_series = (scenario_epoch + pd.to_timedelta(offsets, unit="s")).astype("datetime64[ns]")

    state_frame = pd.DataFrame(states, columns=_STATE_COLUMNS)
    for column in _STATE_COLUMNS:
        try:
            state_frame[column] = pd.to_numeric(state_frame[column])
        except (ValueError, TypeError) as exc:
            raise GmatOutputParseError(
                f"non-numeric value in state column {column!r} of data "
                f"section starting at line {data_start_lineno}: {exc}",
                path,
            ) from exc
        state_frame[column] = state_frame[column].astype("float64")

    df = pd.DataFrame({"Epoch": epochs_series})
    for column in _STATE_COLUMNS:
        df[column] = state_frame[column].to_numpy()
    return df


# --- metadata helpers -------------------------------------------------------


def _parse_scenario_epoch(value: str, path: Path) -> pd.Timestamp:
    """Convert a raw ``ScenarioEpoch`` value to a timestamp.

    GMAT's STK writer emits ``ScenarioEpoch`` in whatever ``EpochFormat`` was
    configured. The ``.e`` file does not say which one. Detect Gregorian (an
    alphabetic month token) vs ModJulian (a single numeric value) by shape;
    raise a typed error on anything else.
    """
    text = value.strip()
    if not text:
        raise GmatOutputParseError("ScenarioEpoch value is empty", path)

    if any(ch.isalpha() for ch in text):
        try:
            return pd.Timestamp(pd.to_datetime(text, format=_GREGORIAN_FORMAT))
        except (ValueError, TypeError) as exc:
            raise GmatOutputParseError(
                f"malformed Gregorian ScenarioEpoch {text!r}: {exc}", path
            ) from exc

    # Numeric → ModJulian. GMAT's MJD epoch is JD 2430000.0 = 1941-01-05 12:00.
    try:
        days = float(text)
    except ValueError as exc:
        raise GmatOutputParseError(
            f"unparseable ScenarioEpoch {text!r}: not Gregorian, not numeric",
            path,
        ) from exc
    try:
        absolute = pd.Timestamp("1941-01-05 12:00:00") + pd.to_timedelta(days, unit="D")
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime) as exc:
        raise GmatOutputParseError(
            f"ScenarioEpoch ModJulian value {text!r} outside the "
            f"representable datetime64[ns] range (~1677..2262): {exc}",
            path,
        ) from exc
    return pd.Timestamp(absolute)


def _coerce_attr(meta_key: str, value: str) -> Any:
    """Light type coercion for surfaced attrs. Mirrors the OEM convention."""
    if meta_key == "InterpolationSamplesM1":
        try:
            return int(value)
        except ValueError:
            return value
    return value


# --- format detection -------------------------------------------------------


def is_stk_ephemeris(path: str | os.PathLike[str]) -> bool:
    """Return ``True`` if ``path`` looks like an STK-TimePosVel ephemeris.

    Sniffs the file's first non-blank, non-comment line for an ``stk.v.X.Y``
    banner. Used by :mod:`gmat_run.results` to dispatch to the right parser
    without relying on file extension.
    """
    try:
        with Path(path).open(encoding="utf-8-sig", newline=None) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                return bool(_VERSION_BANNER_RE.match(line))
    except OSError:
        return False
    return False
