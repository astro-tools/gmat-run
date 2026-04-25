"""Parse a GMAT ``EphemerisFile`` (CCSDS-OEM text format) into a DataFrame.

GMAT's default text-ephemeris writer emits CCSDS Orbit Data Messages in the OEM
flavour (CCSDS 502.0-B-2): a free-form file header (``CCSDS_OEM_VERS``,
``CREATION_DATE``, ``ORIGINATOR``), followed by one or more segments, each a
``META_START`` … ``META_STOP`` block of ``KEY = VALUE`` metadata followed by
ISO-8601-epoch state records ``epoch X Y Z VX VY VZ`` separated by blank lines.

This module parses that single FileFormat. SPK (binary), Code-500 (binary),
STK-TimePosVel, and CCSDS-AEM (attitude-only) are explicitly out of scope and
will fail with :class:`~gmat_run.errors.GmatOutputParseError`.

Multi-segment files are concatenated into one DataFrame in segment order. Per-
segment metadata is preserved on ``df.attrs["segments"]`` as a list of dicts.
When every segment shares the same value for a given metadata key, the key is
also surfaced flat on ``df.attrs`` (``coordinate_system``, ``central_body``,
``time_scale``, ``interpolation``, ``interpolation_degree``, ``object_name``)
so single-segment files — by far the common case — read cleanly.

Optional CCSDS-OEM features deferred for a later release:

* covariance blocks (``COVARIANCE_START`` … ``COVARIANCE_STOP``) are skipped
  with the data lines parsed only on the assumption of seven columns;
* acceleration columns past the mandatory six state components are rejected
  rather than silently dropped — surfacing the format mismatch is more useful
  than guessing intent;
* multiple ``OBJECT_NAME``/``OBJECT_ID`` per file (each typically in its own
  segment) is supported via the per-segment metadata, not via a multi-index.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd

from gmat_run.errors import GmatOutputParseError

__all__ = ["parse"]

# CCSDS-OEM data records use ISO-8601 epochs (``2026-01-01T12:00:00.000``)
# regardless of the ``EpochFormat`` field on the GMAT ``EphemerisFile`` resource
# — that field affects ``START_TIME``/``STOP_TIME`` meta values, not the data
# block. ``%f`` swallows any number of fractional-second digits.
_ISO8601_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%f"

# Mandatory state-vector columns per CCSDS 502.0-B-2 §5.2.4.1.
_STATE_COLUMNS: Final = ["X", "Y", "Z", "VX", "VY", "VZ"]

# Total record column count: epoch + six state components.
_RECORD_COLUMNS: Final = 1 + len(_STATE_COLUMNS)

# Metadata keys we surface as flat attrs when all segments agree.
_META_TO_ATTR: Final = {
    "OBJECT_NAME": "object_name",
    "CENTER_NAME": "central_body",
    "REF_FRAME": "coordinate_system",
    "TIME_SYSTEM": "time_scale",
    "INTERPOLATION": "interpolation",
    "INTERPOLATION_DEGREE": "interpolation_degree",
}

# Lines beginning with these tokens are CCSDS-OEM section delimiters.
_META_START: Final = "META_START"
_META_STOP: Final = "META_STOP"
_COVARIANCE_START: Final = "COVARIANCE_START"
_COVARIANCE_STOP: Final = "COVARIANCE_STOP"

# CCSDS allows free-form COMMENT lines anywhere; we ignore them.
_COMMENT_PREFIX: Final = "COMMENT"

# KEY = VALUE inside meta blocks. Values may contain whitespace
# (``ORIGINATOR     = GMAT USER``) so we split on the first ``=`` only.
_META_KV: Final = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


@dataclass
class _Segment:
    """One CCSDS-OEM segment: meta block plus its data records."""

    meta: dict[str, str] = field(default_factory=dict)
    rows: list[list[str]] = field(default_factory=list)
    # Lineno where the data block starts, for error messages.
    data_start_lineno: int = 0


def parse(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Parse a CCSDS-OEM ephemeris file into a :class:`pandas.DataFrame`.

    Args:
        path: Path to the ``.oem`` (or ``.eph``) file on disk.

    Returns:
        A DataFrame with one row per state record. Columns are ``Epoch`` (a
        ``datetime64[ns]`` column) plus ``X``, ``Y``, ``Z``, ``VX``, ``VY``,
        ``VZ`` (all ``float64``). Segments are concatenated in file order.

        Metadata surfaces on ``df.attrs``:

        * ``df.attrs["epoch_scales"] = {"Epoch": time_scale}`` mirrors the
          convention from :mod:`gmat_run.parsers.epoch`.
        * Flat keys (``object_name``, ``central_body``, ``coordinate_system``,
          ``time_scale``, ``interpolation``, ``interpolation_degree``) are set
          when every segment agrees.
        * ``df.attrs["segments"]`` lists the full per-segment metadata dict
          (only present when more than one segment was parsed).
        * ``df.attrs["file_header"]`` carries the pre-segment header keys
          (``CCSDS_OEM_VERS``, ``CREATION_DATE``, ``ORIGINATOR``).

    Raises:
        GmatOutputParseError: The file is empty, no ``META_START`` block was
            found, a meta line is malformed, a record's column count is wrong,
            or an epoch / state value cannot be parsed.
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline=None) as fh:
        lines = fh.read().splitlines()

    if not any(line.strip() for line in lines):
        raise GmatOutputParseError("file is empty", path)

    file_header, segments = _split(lines, path)
    if not segments:
        raise GmatOutputParseError("no META_START block found; not a CCSDS-OEM ephemeris?", path)

    frames: list[pd.DataFrame] = []
    metas: list[dict[str, str]] = []
    for segment in segments:
        df = _segment_to_frame(segment, path)
        frames.append(df)
        metas.append(segment.meta)

    result = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Tag epoch scale per the epoch.py convention. Use the first segment's
    # TIME_SYSTEM — _consensus already validates that all segments agree on
    # any flat-attr key (and TIME_SYSTEM is one of those).
    time_scale = metas[0].get("TIME_SYSTEM", "")
    if time_scale:
        result.attrs["epoch_scales"] = {"Epoch": time_scale}

    flat = _consensus(metas)
    result.attrs.update(flat)
    if len(metas) > 1:
        result.attrs["segments"] = [dict(m) for m in metas]
    result.attrs["file_header"] = dict(file_header)
    return result


# --- file structure ---------------------------------------------------------


def _split(lines: list[str], path: Path) -> tuple[dict[str, str], list[_Segment]]:
    """Walk ``lines`` once, returning the file header and segment list.

    File header is everything before the first ``META_START``; subsequent
    segments are bracketed by ``META_START``/``META_STOP`` (meta) and run from
    ``META_STOP`` to the next ``META_START`` (or EOF) for data.
    """
    file_header: dict[str, str] = {}
    segments: list[_Segment] = []
    current: _Segment | None = None
    in_meta = False
    in_covariance = False

    for index, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(_COMMENT_PREFIX):
            continue

        if line == _META_START:
            current = _Segment()
            in_meta = True
            continue
        if line == _META_STOP:
            if current is None or not in_meta:
                raise GmatOutputParseError(
                    f"line {index}: META_STOP without matching META_START", path
                )
            in_meta = False
            current.data_start_lineno = index + 1
            segments.append(current)
            continue
        if line == _COVARIANCE_START:
            in_covariance = True
            continue
        if line == _COVARIANCE_STOP:
            in_covariance = False
            continue

        if in_covariance:
            # Covariance blocks are not exposed in v0.2 — silently consumed.
            continue

        if in_meta:
            assert current is not None  # for mypy; guaranteed by META_START
            key, value = _parse_meta_line(line, index, path)
            current.meta[key] = value
            continue

        if current is None:
            # Pre-segment header lines (CCSDS_OEM_VERS, CREATION_DATE, …).
            key, value = _parse_meta_line(line, index, path)
            file_header[key] = value
            continue

        # Data record inside the current segment.
        current.rows.append(_split_record(line, index, path))

    if in_meta:
        raise GmatOutputParseError("unterminated META_START block (no matching META_STOP)", path)
    if in_covariance:
        raise GmatOutputParseError(
            "unterminated COVARIANCE_START block (no matching COVARIANCE_STOP)",
            path,
        )
    return file_header, segments


def _parse_meta_line(line: str, lineno: int, path: Path) -> tuple[str, str]:
    """Split a ``KEY = VALUE`` line. Surface a clear error on garbage."""
    match = _META_KV.match(line)
    if match is None:
        raise GmatOutputParseError(f"line {lineno}: expected 'KEY = VALUE', got {line!r}", path)
    return match.group(1), match.group(2)


def _split_record(line: str, lineno: int, path: Path) -> list[str]:
    """Tokenize a data record line; validate column count up front."""
    tokens = line.split()
    if len(tokens) != _RECORD_COLUMNS:
        raise GmatOutputParseError(
            f"line {lineno}: expected {_RECORD_COLUMNS} columns "
            f"(epoch + 6 state components), got {len(tokens)}",
            path,
        )
    return tokens


# --- segment → frame --------------------------------------------------------


def _segment_to_frame(segment: _Segment, path: Path) -> pd.DataFrame:
    """Convert one segment's records into a typed DataFrame."""
    if not segment.rows:
        # An empty segment is allowed per spec but exotic in practice; surface
        # an empty DataFrame with the expected dtypes so downstream concat
        # doesn't trip on missing columns.
        return pd.DataFrame(
            {
                "Epoch": pd.Series(dtype="datetime64[ns]"),
                **{c: pd.Series(dtype="float64") for c in _STATE_COLUMNS},
            }
        )

    epochs_raw = [row[0] for row in segment.rows]
    states = [row[1:] for row in segment.rows]

    try:
        parsed = pd.to_datetime(epochs_raw, format=_ISO8601_FORMAT)
    except (ValueError, TypeError) as exc:
        raise GmatOutputParseError(
            f"malformed ISO-8601 epoch in segment starting at line "
            f"{segment.data_start_lineno}: {exc}",
            path,
        ) from exc
    # pandas 2.x returns datetime64[us] for this format; normalise to [ns]
    # so we match what reportfile.parse / epoch.promote_epochs produce.
    epochs = pd.DatetimeIndex(parsed.astype("datetime64[ns]"))

    state_frame = pd.DataFrame(states, columns=_STATE_COLUMNS)
    for column in _STATE_COLUMNS:
        try:
            state_frame[column] = pd.to_numeric(state_frame[column])
        except (ValueError, TypeError) as exc:
            raise GmatOutputParseError(
                f"non-numeric value in state column {column!r} of segment "
                f"starting at line {segment.data_start_lineno}: {exc}",
                path,
            ) from exc
        # CCSDS-OEM mandates all six components are real-valued; force float64
        # even if every row of a particular column happens to be an integer.
        state_frame[column] = state_frame[column].astype("float64")

    df = pd.DataFrame({"Epoch": pd.Series(epochs)})
    for column in _STATE_COLUMNS:
        df[column] = state_frame[column].to_numpy()
    return df


# --- metadata helpers -------------------------------------------------------


def _consensus(metas: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Surface meta keys as flat ``df.attrs`` only when every segment agrees.

    Coerces ``INTERPOLATION_DEGREE`` to ``int`` so callers can do arithmetic
    without restringifying. All other values pass through as ``str``.
    """
    metas_list = list(metas)
    if not metas_list:
        return {}
    flat: dict[str, Any] = {}
    for meta_key, attr_key in _META_TO_ATTR.items():
        values = {m.get(meta_key) for m in metas_list}
        if len(values) != 1:
            continue
        (value,) = values
        if value is None:
            continue
        flat[attr_key] = _coerce_attr(meta_key, value)
    return flat


def _coerce_attr(meta_key: str, value: str) -> Any:
    if meta_key == "INTERPOLATION_DEGREE":
        try:
            return int(value)
        except ValueError:
            # Fall back to the raw string; consumer can decide how to react.
            return value
    return value
