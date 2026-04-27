"""Leap-second-correct conversion between GMAT time scales.

GMAT uses five time scales — A1, TAI, UTC, TT, TDB — and emits epoch columns
labelled with each one. :func:`gmat_run.parsers.epoch.promote_epochs` turns
those columns into ``datetime64[ns]`` representing the *labelled* instant in
its native scale; this module handles the conversion *between* scales when a
caller wants every epoch column on a common axis.

All conversion goes through :class:`astropy.time.Time`, which owns the IERS
leap-second table. The library does not bundle a leap-second table of its own.

A1 is GMAT-specific — astropy does not recognise it. Per the GMAT Mathematical
Specification, A1 leads TAI by a fixed 0.0343817 s; this module routes A1
through TAI by applying that offset before/after the astropy conversion.

This module is gated behind the ``[astropy]`` extra. Importing the module
without astropy installed is fine; calling :func:`convert` or
:func:`convert_column` raises :class:`ImportError` with an
``install with gmat-run[astropy]`` hint.
"""

from typing import TYPE_CHECKING, Any, Final

import pandas as pd

if TYPE_CHECKING:
    pass

__all__ = ["convert", "convert_column"]


# A1 leads TAI by exactly 0.0343817 s (GMAT Mathematical Specification §2.1).
# Exactly representable in datetime64[ns] (= 34_381_700 ns).
_A1_TAI_OFFSET: Final = pd.Timedelta("0.0343817s")

# The five GMAT time scales. Order matches the GMAT MathSpec §2.1 listing.
_GMAT_SCALES: Final = ("A1", "TAI", "UTC", "TT", "TDB")

# Mapping from GMAT scale label to astropy scale name. A1 is special-cased
# (astropy has no A1) and routed through TAI; not present in this map.
_ASTROPY_SCALE: Final[dict[str, str]] = {
    "TAI": "tai",
    "UTC": "utc",
    "TT": "tt",
    "TDB": "tdb",
}


def convert(
    series: "pd.Series[pd.Timestamp]",
    from_scale: str,
    to_scale: str,
) -> "pd.Series[pd.Timestamp]":
    """Convert an epoch ``Series`` from ``from_scale`` to ``to_scale``.

    Both scales must be one of ``"A1"``, ``"TAI"``, ``"UTC"``, ``"TT"``,
    ``"TDB"``. Same-scale conversion returns a copy of ``series`` without
    importing astropy.

    Args:
        series: ``datetime64[ns]`` ``Series`` representing the labelled
            instant in ``from_scale`` (the dtype produced by
            :func:`gmat_run.parsers.epoch.promote_epochs`).
        from_scale: Source GMAT time scale.
        to_scale: Target GMAT time scale.

    Returns:
        A new ``Series`` of the same dtype, index, and name, with values
        representing the same physical instant in ``to_scale``.

    Raises:
        ValueError: ``from_scale`` or ``to_scale`` is not one of the five
            recognised GMAT scales, or ``series`` is not ``datetime64[ns]``.
        ImportError: ``astropy`` is not installed and a non-trivial
            conversion is requested. The message points at the
            ``[astropy]`` extra.
    """
    _require_known_scale("from_scale", from_scale)
    _require_known_scale("to_scale", to_scale)
    _require_datetime_dtype(series)

    if from_scale == to_scale:
        return series.copy()

    # A1 ↔ TAI is a pure constant offset (no leap seconds, no astropy).
    # Routing it through astropy.time.Time would truncate at microsecond
    # precision via the .datetime64 accessor; doing it in pandas keeps the
    # full ns of the input.
    if {from_scale, to_scale} <= {"A1", "TAI"}:
        sign = -1 if from_scale == "A1" else 1
        return pd.Series(
            series.to_numpy() + sign * _A1_TAI_OFFSET.to_numpy(),
            index=series.index,
            name=series.name,
        )

    Time = _import_astropy_time()

    # A1 input → shift to TAI, then let astropy handle the rest.
    values = series.to_numpy()
    if from_scale == "A1":
        values = values - _A1_TAI_OFFSET.to_numpy()
        astropy_from = "tai"
    else:
        astropy_from = _ASTROPY_SCALE[from_scale]

    # A1 output → ask astropy for TAI, then apply the offset.
    astropy_to = "tai" if to_scale == "A1" else _ASTROPY_SCALE[to_scale]

    t = Time(values, format="datetime64", scale=astropy_from)
    converted_values = getattr(t, astropy_to).datetime64

    if to_scale == "A1":
        converted_values = converted_values + _A1_TAI_OFFSET.to_numpy()

    return pd.Series(converted_values, index=series.index, name=series.name)


def convert_column(df: pd.DataFrame, column: str, to_scale: str) -> pd.DataFrame:
    """Convert ``df[column]`` to ``to_scale`` and update ``df.attrs``.

    The source scale is read from ``df.attrs["epoch_scales"][column]`` —
    populated by :func:`gmat_run.parsers.epoch.promote_epochs`. After
    conversion ``df.attrs["epoch_scales"][column]`` is updated to
    ``to_scale``.

    Idempotent when the source and target scales are equal: no astropy
    import, no data copy beyond the in-place attrs touch.

    Args:
        df: DataFrame whose ``df.attrs["epoch_scales"]`` records the source
            scale for ``column``.
        column: Column name. Must be present in ``df`` and in
            ``df.attrs["epoch_scales"]``.
        to_scale: Target GMAT time scale.

    Returns:
        ``df`` itself, mutated in place.

    Raises:
        ValueError: ``column`` is not in ``df``, the source scale is not
            recorded in ``df.attrs["epoch_scales"]``, or either scale is
            not one of the five recognised GMAT scales.
        ImportError: ``astropy`` is not installed and a non-trivial
            conversion is requested.
    """
    if column not in df.columns:
        raise ValueError(f"column {column!r} not in DataFrame")
    scales: dict[str, str] = df.attrs.get("epoch_scales", {})
    if column not in scales:
        raise ValueError(
            f"column {column!r} has no recorded source scale in "
            f"df.attrs['epoch_scales']; call promote_epochs first or set the "
            f"scale manually"
        )
    from_scale = scales[column]
    df[column] = convert(df[column], from_scale, to_scale)
    scales[column] = to_scale
    return df


def _require_known_scale(arg_name: str, scale: str) -> None:
    if scale not in _GMAT_SCALES:
        raise ValueError(
            f"{arg_name}={scale!r} is not a recognised GMAT time scale; "
            f"valid scales are {list(_GMAT_SCALES)}"
        )


def _require_datetime_dtype(series: pd.Series) -> None:
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError(
            f"expected a datetime64 Series (the dtype produced by "
            f"promote_epochs); got dtype={series.dtype}"
        )


def _import_astropy_time() -> Any:
    """Import :class:`astropy.time.Time`, with a friendly error on failure.

    astropy carries IERS leap-second data and is a heavy optional
    dependency. Importing here (rather than at module top) keeps
    ``import gmat_run.time`` cheap for callers who only want the
    same-scale fast path.
    """
    try:
        from astropy.time import Time
    except ImportError as exc:
        raise ImportError(
            "Time-scale conversion requires astropy: install with `pip install gmat-run[astropy]`"
        ) from exc
    return Time
