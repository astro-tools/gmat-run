"""Promote known GMAT epoch columns in a DataFrame to ``datetime64[ns]``.

GMAT emits epochs in one of ten ``{scale}{format}`` combinations — five scales
(A1, TAI, UTC, TT, TDB) crossed with two formats (``Gregorian``,
``ModJulian``). :func:`promote_epochs` recognises all ten by exact match on
the column name's last dotted segment, converts each to ``datetime64[ns]``,
and records the time scale on ``df.attrs["epoch_scales"]`` so downstream code
can branch on it without re-parsing the column name.

No leap-second-correct time-scale *conversion* happens by default: a
``TAIModJulian`` column becomes a ``datetime64[ns]`` representing the TAI
instant, labelled ``"TAI"``. Pass ``convert_to=`` to convert every promoted
column to a single target scale in one call — that path delegates to
:mod:`gmat_run.time` and inherits the ``[astropy]`` extra requirement; see
:func:`promote_epochs` for the details. For ad-hoc conversion of an
already-promoted DataFrame, call :func:`gmat_run.time.convert_column`
directly.

Columns whose name ends in ``Gregorian`` or ``ModJulian`` but are not one of
the ten recognised names trigger a ``UserWarning`` and are left untouched —
this catches typos and any hypothetical future GMAT epoch format without
silently mis-promoting data.
"""

import re
import warnings
from pathlib import Path
from typing import Final

import pandas as pd

from gmat_run.errors import GmatOutputParseError

__all__ = ["promote_epochs"]


# MJD 0 in GMAT's convention: JD 2430000.0 = 1941-01-05 12:00:00.
# Cross-check: GMAT MJD 21545.0 must map to J2000 (2000-01-01 12:00:00),
# and it does — see ``tests/test_parsers_epoch.py``.
_GMAT_MJD_EPOCH: Final = pd.Timestamp("1941-01-05 12:00:00")

# GMAT Gregorian format, e.g. ``26 Nov 2026 12:00:00.000``. ``%b`` is locale
# sensitive; this assumes the C/en_US locale active on every CI runner and
# virtually every dev workstation. If a non-English locale is ever a real
# target we would swap this for a manual month-name lookup.
_GREGORIAN_FORMAT: Final = "%d %b %Y %H:%M:%S.%f"

# Suffix → time scale. Suffix matches the column name's last dotted segment
# (or the whole name if there is no dot).
_GREGORIAN_SUFFIXES: Final[dict[str, str]] = {
    "A1Gregorian": "A1",
    "TAIGregorian": "TAI",
    "UTCGregorian": "UTC",
    "TTGregorian": "TT",
    "TDBGregorian": "TDB",
}
_MODJULIAN_SUFFIXES: Final[dict[str, str]] = {
    "A1ModJulian": "A1",
    "TAIModJulian": "TAI",
    "UTCModJulian": "UTC",
    "TTModJulian": "TT",
    "TDBModJulian": "TDB",
}

# Suffix shape of "epoch-looking but unrecognised" columns — triggers the
# unknown-format warning. Matches the last dotted segment.
_UNKNOWN_EPOCH_SUFFIX_RE: Final = re.compile(r".+(?:Gregorian|ModJulian)$")


def promote_epochs(df: pd.DataFrame, *, convert_to: str | None = None) -> pd.DataFrame:
    """Promote recognised epoch columns in ``df`` to ``datetime64[ns]`` in place.

    The DataFrame is mutated and returned so callers can chain. Time scales for
    promoted columns are recorded under ``df.attrs["epoch_scales"]`` as a
    ``{column_name: scale_string}`` dict. ``scale_string`` is one of
    ``"A1"``, ``"TAI"``, ``"UTC"``, ``"TT"``, ``"TDB"``.

    Idempotent: calling this twice on the same frame is a no-op on the second
    pass, because promoted columns are already ``datetime64[ns]`` and are
    skipped.

    Args:
        df: DataFrame whose columns follow GMAT's ``{resource}.{field}``
            naming convention. Columns whose last segment is one of the ten
            recognised epoch suffixes are promoted.
        convert_to: If set, every column listed in ``df.attrs["epoch_scales"]``
            after promotion is converted from its detected scale to
            ``convert_to`` and ``df.attrs["epoch_scales"]`` is updated
            accordingly. Must be one of ``"A1"``, ``"TAI"``, ``"UTC"``,
            ``"TT"``, ``"TDB"``. Conversion is delegated to
            :func:`gmat_run.time.convert_column`, which is gated behind the
            ``[astropy]`` extra.

    Returns:
        ``df`` itself, mutated in place.

    Raises:
        GmatOutputParseError: A recognised epoch column contains values that
            cannot be parsed (malformed Gregorian text, non-numeric ModJulian,
            or a ModJulian value that overflows ``datetime64[ns]``'s range).
        ValueError: ``convert_to`` is set to a value that is not one of the
            five recognised GMAT scales, or one of the recorded source scales
            on ``df`` is not recognised (e.g. a CCSDS ``TIME_SYSTEM`` of
            ``UT1`` or ``GPS``).
        ImportError: ``convert_to`` is set, a non-trivial conversion is
            required, and ``astropy`` is not installed. The message points at
            the ``gmat-run[astropy]`` extra.
    """
    for column in df.columns:
        suffix = _suffix(str(column))
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            # Already promoted (idempotence). Tag the name-derived scale only
            # when nothing recorded one yet — a caller that built the frame by
            # hand still gets consistent attrs — but never overwrite a scale an
            # earlier pass or convert_column already recorded: the column name
            # keeps its original suffix after a conversion, so re-deriving from
            # it would silently mislabel converted data.
            scale = _GREGORIAN_SUFFIXES.get(suffix) or _MODJULIAN_SUFFIXES.get(suffix)
            if scale is not None and column not in df.attrs.get("epoch_scales", {}):
                _tag_scale(df, column, scale)
            continue
        if suffix in _GREGORIAN_SUFFIXES:
            df[column] = _convert_gregorian(df[column], column)
            _tag_scale(df, column, _GREGORIAN_SUFFIXES[suffix])
        elif suffix in _MODJULIAN_SUFFIXES:
            df[column] = _convert_modjulian(df[column], column)
            _tag_scale(df, column, _MODJULIAN_SUFFIXES[suffix])
        elif _UNKNOWN_EPOCH_SUFFIX_RE.match(suffix):
            warnings.warn(
                f"Column {column!r} has an unrecognised epoch suffix {suffix!r}; "
                "leaving it unchanged. Known suffixes: "
                f"{sorted(_GREGORIAN_SUFFIXES) + sorted(_MODJULIAN_SUFFIXES)}.",
                UserWarning,
                stacklevel=2,
            )
    if convert_to is not None:
        _convert_all_to(df, convert_to)
    return df


def _convert_all_to(df: pd.DataFrame, target: str) -> None:
    """Convert every entry in ``df.attrs["epoch_scales"]`` to ``target`` in place.

    Imports :mod:`gmat_run.time` lazily so the default ``promote_epochs`` path
    stays astropy-free. ``convert_column`` validates ``target`` and each
    source scale, and surfaces the ``[astropy]`` install hint when needed.
    """
    from gmat_run.time import convert_column

    scales: dict[str, str] = df.attrs.get("epoch_scales", {})
    # Snapshot the keys: ``convert_column`` mutates this dict as it goes.
    for column in list(scales):
        convert_column(df, column, target)


def _suffix(column: str) -> str:
    """Return the portion of ``column`` after the last ``.``, or the whole name."""
    _, _, tail = column.rpartition(".")
    return tail or column


def _tag_scale(df: pd.DataFrame, column: str, scale: str) -> None:
    """Record ``column``'s time scale on ``df.attrs["epoch_scales"]``."""
    scales = df.attrs.setdefault("epoch_scales", {})
    scales[column] = scale


def _convert_gregorian(series: "pd.Series[str]", column: str) -> "pd.Series[pd.Timestamp]":
    """Convert a GMAT Gregorian string column to ``datetime64[ns]``."""
    try:
        parsed = pd.to_datetime(series, format=_GREGORIAN_FORMAT)
    except (ValueError, TypeError) as exc:
        raise GmatOutputParseError(
            f"column {column!r} has malformed Gregorian epoch value: {exc}",
            _synthetic_path(column),
        ) from exc
    # pandas 2.x returns datetime64[us] for this format; normalise to the [ns]
    # precision promised by the issue's acceptance criterion.
    return parsed.astype("datetime64[ns]")


def _convert_modjulian(series: "pd.Series[float]", column: str) -> "pd.Series[pd.Timestamp]":
    """Convert a GMAT ModJulian numeric column to ``datetime64[ns]``."""
    if not pd.api.types.is_numeric_dtype(series):
        raise GmatOutputParseError(
            f"column {column!r} is a ModJulian epoch but its values are "
            f"non-numeric (dtype={series.dtype})",
            _synthetic_path(column),
        )
    try:
        return _GMAT_MJD_EPOCH + pd.to_timedelta(series, unit="D")
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime) as exc:
        raise GmatOutputParseError(
            f"column {column!r} has a ModJulian value outside the "
            f"representable datetime64[ns] range (~1677..2262): {exc}",
            _synthetic_path(column),
        ) from exc


def _synthetic_path(column: str) -> Path:
    """Placeholder ``path`` for errors that don't know the file location.

    ``promote_epochs`` is called from ``reportfile.parse`` which knows the
    path, but the function itself is reusable on any in-memory DataFrame; when
    there is no file, we surface the offending column name instead.
    """
    return Path(f"<column:{column}>")
