"""Unit tests for :mod:`gmat_run.time` time-scale conversion."""

import itertools
import sys

import numpy as np
import pandas as pd
import pytest

from gmat_run.time import convert, convert_column

# A representative epoch well inside the datetime64[ns] safe range and after
# the latest historical leap second (2017-01-01).
_EPOCH = pd.Timestamp("2026-01-15 12:00:00")

_SCALES = ("A1", "TAI", "UTC", "TT", "TDB")


# --- round-trip --------------------------------------------------------------


def test_roundtrip_through_all_scales_within_one_nanosecond() -> None:
    """A1 → TAI → UTC → TT → TDB → A1 returns the input within 1 ns."""
    s = pd.Series([_EPOCH, _EPOCH + pd.Timedelta(days=1)], name="Sat.A1ModJulian")

    cur = s
    chain = ["A1", "TAI", "UTC", "TT", "TDB", "A1"]
    for from_scale, to_scale in itertools.pairwise(chain):
        cur = convert(cur, from_scale, to_scale)

    delta = (cur - s).abs()
    assert (delta <= pd.Timedelta("1ns")).all(), delta.tolist()


@pytest.mark.parametrize("scale", _SCALES)
def test_same_scale_returns_copy(scale: str) -> None:
    s = pd.Series([_EPOCH], name="t")
    out = convert(s, scale, scale)
    assert out.equals(s)
    # Mutating the result must not affect the input — it is a copy.
    assert out is not s


def test_same_scale_does_not_import_astropy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same-scale shortcut takes the no-astropy path."""
    monkeypatch.setitem(sys.modules, "astropy", None)
    monkeypatch.setitem(sys.modules, "astropy.time", None)
    s = pd.Series([_EPOCH], name="t")
    out = convert(s, "UTC", "UTC")
    assert out.equals(s)


# --- A1 offset ---------------------------------------------------------------

# A1 leads TAI by exactly 0.0343817 s = 34_381_700 ns. Tests compare in
# integer nanoseconds because pd.Timedelta.total_seconds() truncates at
# microsecond precision and would mask the trailing 700 ns of the offset.
_A1_OFFSET_NS = 34_381_700


def test_a1_to_tai_applies_fixed_offset() -> None:
    """A1 - 34381700 ns == TAI for a single epoch."""
    s = pd.Series([_EPOCH], name="t")
    tai = convert(s, "A1", "TAI")
    delta = s.iloc[0] - tai.iloc[0]
    assert delta.value == _A1_OFFSET_NS


def test_tai_to_a1_applies_fixed_offset() -> None:
    s = pd.Series([_EPOCH], name="t")
    a1 = convert(s, "TAI", "A1")
    delta = a1.iloc[0] - s.iloc[0]
    assert delta.value == _A1_OFFSET_NS


def test_a1_to_utc_routes_through_astropy() -> None:
    """A1 → UTC = (A1 - 0.0343817 s) → UTC; ~37 s offset in 2026 plus the A1 shift."""
    s = pd.Series([_EPOCH], name="t")
    utc = convert(s, "A1", "UTC")
    # A1 → TAI subtracts 0.0343817 s; TAI → UTC subtracts 37 s in 2026.
    # So A1 → UTC subtracts ~37.0343817 s; check at second precision.
    delta = (s.iloc[0] - utc.iloc[0]).total_seconds()
    assert delta == pytest.approx(37.0343817, abs=1e-3)


# --- leap-second boundary ----------------------------------------------------


def test_utc_to_tai_jumps_one_second_across_2017_leap() -> None:
    """The 2017-01-01 leap second adds 1 s to TAI-UTC across the boundary.

    UTC 2016-12-31 23:59:59 → TAI offset 36 s.
    UTC 2017-01-01 00:00:01 → TAI offset 37 s.
    """
    before = pd.Series([pd.Timestamp("2016-12-31 23:59:59")], name="t")
    after = pd.Series([pd.Timestamp("2017-01-01 00:00:01")], name="t")

    tai_before = convert(before, "UTC", "TAI")
    tai_after = convert(after, "UTC", "TAI")

    offset_before = (tai_before.iloc[0] - before.iloc[0]).total_seconds()
    offset_after = (tai_after.iloc[0] - after.iloc[0]).total_seconds()

    assert offset_before == pytest.approx(36.0, abs=1e-9)
    assert offset_after == pytest.approx(37.0, abs=1e-9)
    assert offset_after - offset_before == pytest.approx(1.0, abs=1e-9)


# --- ImportError plumbing ----------------------------------------------------


def test_missing_astropy_raises_friendly_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-trivial conversion with astropy missing surfaces the install hint."""
    # Block the import: drop any cached astropy modules and make resolution
    # return None so `from astropy.time import Time` raises ImportError.
    monkeypatch.setitem(sys.modules, "astropy", None)
    monkeypatch.setitem(sys.modules, "astropy.time", None)

    s = pd.Series([_EPOCH], name="t")
    with pytest.raises(ImportError, match=r"astropy.*\[astropy\]"):
        convert(s, "UTC", "TAI")


# --- input validation --------------------------------------------------------


@pytest.mark.parametrize("bad", ["a1", "GPS", "", "ut1"])
def test_unknown_from_scale_raises_value_error(bad: str) -> None:
    s = pd.Series([_EPOCH], name="t")
    with pytest.raises(ValueError, match=r"from_scale"):
        convert(s, bad, "TAI")


@pytest.mark.parametrize("bad", ["a1", "GPS", "", "ut1"])
def test_unknown_to_scale_raises_value_error(bad: str) -> None:
    s = pd.Series([_EPOCH], name="t")
    with pytest.raises(ValueError, match=r"to_scale"):
        convert(s, "TAI", bad)


def test_non_datetime_dtype_raises_value_error() -> None:
    s = pd.Series([1.0, 2.0], name="t")
    with pytest.raises(ValueError, match=r"datetime64"):
        convert(s, "TAI", "UTC")


# --- convert_column ----------------------------------------------------------


def test_convert_column_updates_dataframe_and_attrs() -> None:
    df = pd.DataFrame({"Sat.UTCGregorian": [_EPOCH]})
    df.attrs["epoch_scales"] = {"Sat.UTCGregorian": "UTC"}

    out = convert_column(df, "Sat.UTCGregorian", "TAI")

    assert out is df  # mutates in place, returns self
    assert df.attrs["epoch_scales"]["Sat.UTCGregorian"] == "TAI"
    delta = (df["Sat.UTCGregorian"].iloc[0] - _EPOCH).total_seconds()
    # 2026 is past the last leap second, so TAI-UTC = 37 s.
    assert delta == pytest.approx(37.0, abs=1e-9)


def test_convert_column_same_scale_is_noop_on_attrs() -> None:
    df = pd.DataFrame({"Sat.UTCGregorian": [_EPOCH]})
    df.attrs["epoch_scales"] = {"Sat.UTCGregorian": "UTC"}

    convert_column(df, "Sat.UTCGregorian", "UTC")

    assert df.attrs["epoch_scales"]["Sat.UTCGregorian"] == "UTC"
    assert df["Sat.UTCGregorian"].iloc[0] == _EPOCH


def test_convert_column_missing_column_raises() -> None:
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match=r"not in DataFrame"):
        convert_column(df, "missing", "TAI")


def test_convert_column_missing_recorded_scale_raises() -> None:
    df = pd.DataFrame({"Sat.UTCGregorian": [_EPOCH]})
    # No epoch_scales attr — promote_epochs was not called.
    with pytest.raises(ValueError, match=r"no recorded source scale"):
        convert_column(df, "Sat.UTCGregorian", "TAI")


# --- regression sanity -------------------------------------------------------


def test_convert_preserves_index_and_name() -> None:
    idx = pd.Index([10, 20, 30], name="row")
    s = pd.Series(
        [_EPOCH, _EPOCH + pd.Timedelta(seconds=1), _EPOCH + pd.Timedelta(seconds=2)],
        index=idx,
        name="Sat.UTCGregorian",
    )
    out = convert(s, "UTC", "TAI")
    assert out.index.equals(idx)
    assert out.name == "Sat.UTCGregorian"
    assert len(out) == 3


def test_convert_preserves_dtype_datetime64ns() -> None:
    s = pd.Series([_EPOCH], name="t")
    out = convert(s, "UTC", "TAI")
    assert out.dtype == np.dtype("datetime64[ns]")
