"""Parse a CCSDS-AEM (Attitude Ephemeris Message) file into a DataFrame.

CCSDS-AEM (CCSDS 504.0-B-1) is the attitude analogue of CCSDS-OEM: a free-form
file header (``CCSDS_AEM_VERS``, ``CREATION_DATE``, ``ORIGINATOR``), then one
or more segments, each a ``META_START`` … ``META_STOP`` block of
``KEY = VALUE`` metadata followed by a ``DATA_START`` … ``DATA_STOP`` block
of ISO-8601-epoch attitude records. Unlike OEM, the data block is explicitly
bracketed and the record shape varies with the segment's ``ATTITUDE_TYPE``.

GMAT R2026a does **not** emit AEM via :guilabel:`EphemerisFile` — its writer
supports CCSDS-OEM, SPK, Code-500, and STK-TimePosVel only. AEM in GMAT is
exclusively a *reader* format consumed by ``Spacecraft.AttitudeFileName``;
this parser exists to surface those input files (see
:attr:`gmat_run.Mission.attitude_inputs`) and to handle third-party AEM.

Supported ``ATTITUDE_TYPE`` values:

* ``QUATERNION`` — four components per record. Column order matches the file
  (no reordering); ``df.attrs["quaternion_type"]`` (``"LAST"`` / ``"FIRST"``)
  tells the caller which column is the scalar.
* ``EULER_ANGLE`` — three angles per record; ``df.attrs["euler_rot_seq"]``
  (e.g. ``"321"``) carries the rotation sequence.

The rate/derivative/spin variants (``QUATERNION/DERIVATIVE``,
``QUATERNION/RATE``, ``EULER_ANGLE/RATE``, ``SPIN``, ``SPIN/NUTATION``) are
rejected with a typed :class:`~gmat_run.errors.GmatOutputParseError` rather
than silently dropping or guessing extra columns. The same is true of a
multi-segment file mixing two different ``ATTITUDE_TYPE`` values — concat
across heterogeneous column shapes is meaningless.

Multi-segment files of a single ``ATTITUDE_TYPE`` are concatenated in segment
order. Per-segment metadata is preserved on ``df.attrs["segments"]`` as a list
of dicts. When every segment shares the same value for a given key, the key
is also surfaced flat on ``df.attrs`` so single-segment files — by far the
common case — read cleanly.
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

__all__ = ["is_aem_ephemeris", "parse"]

# CCSDS-AEM data records use ISO-8601 epochs (``2000-01-01T11:59:28.000``).
# ``%f`` swallows any number of fractional-second digits.
_ISO8601_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%f"

# Section markers.
_META_START: Final = "META_START"
_META_STOP: Final = "META_STOP"
_DATA_START: Final = "DATA_START"
_DATA_STOP: Final = "DATA_STOP"

# CCSDS allows COMMENT lines anywhere; ignored.
_COMMENT_PREFIX: Final = "COMMENT"

# KEY = VALUE inside meta blocks. Values may contain whitespace; split on the
# first ``=`` only.
_META_KV: Final = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")

# Banner that identifies an AEM file. Used by :func:`is_aem_ephemeris`.
_AEM_VERS_RE: Final = re.compile(r"^\s*CCSDS_AEM_VERS\s*=")

# Per-segment metadata keys promoted to flat ``df.attrs`` when every segment
# in the file agrees on the value. ``ATTITUDE_TYPE`` is included so callers
# can branch without iterating ``segments``.
_META_TO_ATTR: Final = {
    "OBJECT_NAME": "object_name",
    "OBJECT_ID": "object_id",
    "CENTER_NAME": "center_name",
    "REF_FRAME_A": "ref_frame_a",
    "REF_FRAME_B": "ref_frame_b",
    "ATTITUDE_DIR": "attitude_dir",
    "TIME_SYSTEM": "time_scale",
    "ATTITUDE_TYPE": "attitude_type",
    "QUATERNION_TYPE": "quaternion_type",
    "EULER_ROT_SEQ": "euler_rot_seq",
    "INTERPOLATION_METHOD": "interpolation",
    "INTERPOLATION_DEGREE": "interpolation_degree",
}

# Supported ATTITUDE_TYPE values. The keys map to the data columns the parser
# emits for that type (in source order — no reordering of QUATERNION components).
_ATTITUDE_COLUMNS: Final = {
    "QUATERNION": ["Q1", "Q2", "Q3", "Q4"],
    "EULER_ANGLE": ["EulerAngle1", "EulerAngle2", "EulerAngle3"],
}

# ATTITUDE_TYPE values the spec allows but this parser rejects rather than
# guess column shape. Surfacing the format mismatch is more useful than
# silently dropping rate / derivative / spin information.
_REJECTED_ATTITUDE_TYPES: Final = (
    "QUATERNION/DERIVATIVE",
    "QUATERNION/RATE",
    "EULER_ANGLE/RATE",
    "SPIN",
    "SPIN/NUTATION",
)


@dataclass
class _Segment:
    """One CCSDS-AEM segment: meta block plus its data records."""

    meta: dict[str, str] = field(default_factory=dict)
    rows: list[list[str]] = field(default_factory=list)
    # Lineno where the data block starts, for error messages.
    data_start_lineno: int = 0


def parse(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Parse a CCSDS-AEM attitude ephemeris file into a :class:`pandas.DataFrame`.

    Args:
        path: Path to the ``.aem`` file on disk.

    Returns:
        A DataFrame with one row per attitude record. Columns depend on the
        file's ``ATTITUDE_TYPE``:

        * ``QUATERNION`` → ``Epoch`` plus ``Q1``, ``Q2``, ``Q3``, ``Q4`` in
            source order; ``df.attrs["quaternion_type"]`` (``"LAST"`` /
            ``"FIRST"``) names the scalar component.
        * ``EULER_ANGLE`` → ``Epoch`` plus ``EulerAngle1``, ``EulerAngle2``,
            ``EulerAngle3``; ``df.attrs["euler_rot_seq"]`` carries the rotation
            sequence (``"321"`` etc.).

        Common metadata surfaces on ``df.attrs``:

        * ``df.attrs["epoch_scales"] = {"Epoch": time_scale}`` mirrors the
            convention from :mod:`gmat_run.parsers.epoch`.
        * Flat keys (``attitude_type``, ``object_name``, ``object_id``,
            ``center_name``, ``ref_frame_a``, ``ref_frame_b``, ``attitude_dir``,
            ``time_scale``, ``interpolation``, ``interpolation_degree``,
            plus the type-specific ``quaternion_type`` / ``euler_rot_seq``)
            are set when every segment agrees on them.
        * ``df.attrs["segments"]`` lists the full per-segment metadata dict
            (only present when more than one segment was parsed).
        * ``df.attrs["file_header"]`` carries the pre-segment header keys
            (``CCSDS_AEM_VERS``, ``CREATION_DATE``, ``ORIGINATOR``).

    Raises:
        GmatOutputParseError: The file is empty, no ``META_START`` block was
            found, a meta line is malformed, ``ATTITUDE_TYPE`` is missing or
            unsupported, the ``DATA_START``/``DATA_STOP`` brackets are
            unbalanced, segments declare different ``ATTITUDE_TYPE``s, a
            record's column count is wrong, or an epoch / numeric value
            cannot be parsed.
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline=None) as fh:
        lines = fh.read().splitlines()

    if not any(line.strip() for line in lines):
        raise GmatOutputParseError("file is empty", path)

    file_header, segments = _split(lines, path)
    if not segments:
        raise GmatOutputParseError("no META_START block found; not a CCSDS-AEM ephemeris?", path)

    # All segments must agree on ATTITUDE_TYPE; mixing column shapes in one
    # frame is meaningless.
    types = {seg.meta.get("ATTITUDE_TYPE", "") for seg in segments}
    if len(types) != 1:
        raise GmatOutputParseError(
            f"segments declare different ATTITUDE_TYPE values {sorted(types)}; "
            "cannot concatenate heterogeneous attitude data",
            path,
        )
    (attitude_type,) = types
    columns = _resolve_columns(attitude_type, path)

    frames: list[pd.DataFrame] = []
    metas: list[dict[str, str]] = []
    for segment in segments:
        df = _segment_to_frame(segment, columns, path)
        frames.append(df)
        metas.append(segment.meta)

    result = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

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
    segments are bracketed by ``META_START``/``META_STOP`` (meta) and then
    ``DATA_START``/``DATA_STOP`` (records).
    """
    file_header: dict[str, str] = {}
    segments: list[_Segment] = []
    current: _Segment | None = None
    in_meta = False
    in_data = False

    for index, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(_COMMENT_PREFIX):
            continue

        if line == _META_START:
            if in_meta or in_data:
                raise GmatOutputParseError(f"line {index}: META_START inside an open block", path)
            current = _Segment()
            in_meta = True
            continue
        if line == _META_STOP:
            if current is None or not in_meta:
                raise GmatOutputParseError(
                    f"line {index}: META_STOP without matching META_START", path
                )
            in_meta = False
            continue
        if line == _DATA_START:
            if current is None or in_meta:
                raise GmatOutputParseError(
                    f"line {index}: DATA_START outside of a closed META block", path
                )
            in_data = True
            current.data_start_lineno = index + 1
            continue
        if line == _DATA_STOP:
            if current is None or not in_data:
                raise GmatOutputParseError(
                    f"line {index}: DATA_STOP without matching DATA_START", path
                )
            in_data = False
            segments.append(current)
            current = None
            continue

        if in_meta:
            assert current is not None  # for mypy; guaranteed by META_START
            key, value = _parse_meta_line(line, index, path)
            current.meta[key] = value
            continue

        if in_data:
            assert current is not None  # for mypy; guaranteed by DATA_START
            current.rows.append(line.split())
            continue

        if current is None:
            # Pre-segment header lines (CCSDS_AEM_VERS, CREATION_DATE, …).
            key, value = _parse_meta_line(line, index, path)
            file_header[key] = value
            continue

        # Between META_STOP and DATA_START, or after DATA_STOP and before the
        # next META_START. The spec permits no content here.
        raise GmatOutputParseError(
            f"line {index}: unexpected content between segment blocks: {line!r}", path
        )

    if in_meta:
        raise GmatOutputParseError("unterminated META_START block (no matching META_STOP)", path)
    if in_data:
        raise GmatOutputParseError("unterminated DATA_START block (no matching DATA_STOP)", path)
    return file_header, segments


def _parse_meta_line(line: str, lineno: int, path: Path) -> tuple[str, str]:
    """Split a ``KEY = VALUE`` line. Surface a clear error on garbage."""
    match = _META_KV.match(line)
    if match is None:
        raise GmatOutputParseError(f"line {lineno}: expected 'KEY = VALUE', got {line!r}", path)
    return match.group(1), match.group(2)


def _resolve_columns(attitude_type: str, path: Path) -> list[str]:
    """Return the data-column names for ``attitude_type`` or raise."""
    if not attitude_type:
        raise GmatOutputParseError("missing ATTITUDE_TYPE in META block", path)
    if attitude_type in _REJECTED_ATTITUDE_TYPES:
        raise GmatOutputParseError(
            f"ATTITUDE_TYPE {attitude_type!r} is not supported by this parser; "
            "only QUATERNION and EULER_ANGLE are handled",
            path,
        )
    columns = _ATTITUDE_COLUMNS.get(attitude_type)
    if columns is None:
        raise GmatOutputParseError(
            f"unrecognised ATTITUDE_TYPE {attitude_type!r}; "
            f"expected one of {sorted(_ATTITUDE_COLUMNS)}",
            path,
        )
    return list(columns)


# --- segment → frame --------------------------------------------------------


def _segment_to_frame(segment: _Segment, columns: list[str], path: Path) -> pd.DataFrame:
    """Convert one segment's records into a typed DataFrame."""
    expected_record_columns = 1 + len(columns)
    empty_frame = pd.DataFrame(
        {
            "Epoch": pd.Series(dtype="datetime64[ns]"),
            **{c: pd.Series(dtype="float64") for c in columns},
        }
    )
    if not segment.rows:
        return empty_frame

    for offset, row in enumerate(segment.rows):
        if len(row) != expected_record_columns:
            raise GmatOutputParseError(
                f"line {segment.data_start_lineno + offset}: expected "
                f"{expected_record_columns} columns (epoch + {len(columns)} "
                f"attitude components), got {len(row)}",
                path,
            )

    epochs_raw = [row[0] for row in segment.rows]
    values = [row[1:] for row in segment.rows]

    try:
        parsed = pd.to_datetime(epochs_raw, format=_ISO8601_FORMAT)
    except (ValueError, TypeError) as exc:
        raise GmatOutputParseError(
            f"malformed ISO-8601 epoch in segment starting at line "
            f"{segment.data_start_lineno}: {exc}",
            path,
        ) from exc
    epochs = pd.DatetimeIndex(parsed.astype("datetime64[ns]"))

    value_frame = pd.DataFrame(values, columns=columns)
    for column in columns:
        try:
            value_frame[column] = pd.to_numeric(value_frame[column])
        except (ValueError, TypeError) as exc:
            raise GmatOutputParseError(
                f"non-numeric value in column {column!r} of segment "
                f"starting at line {segment.data_start_lineno}: {exc}",
                path,
            ) from exc
        value_frame[column] = value_frame[column].astype("float64")

    df = pd.DataFrame({"Epoch": pd.Series(epochs)})
    for column in columns:
        df[column] = value_frame[column].to_numpy()
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
            return value
    return value


# --- format detection -------------------------------------------------------


def is_aem_ephemeris(path: str | os.PathLike[str]) -> bool:
    """Return ``True`` if ``path`` looks like a CCSDS-AEM file.

    Sniffs the first non-blank, non-comment line for a ``CCSDS_AEM_VERS = …``
    header. Cheap enough to run on every candidate file when classifying a
    directory's worth of attitude artefacts.
    """
    try:
        with Path(path).open(encoding="utf-8-sig", newline=None) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith(_COMMENT_PREFIX):
                    continue
                return bool(_AEM_VERS_RE.match(line))
    except OSError:
        return False
    return False
