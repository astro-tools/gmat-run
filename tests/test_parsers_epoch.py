"""Unit tests for :func:`gmat_run.parsers.epoch.promote_epochs`.

Tests build DataFrames in memory — no file I/O, no ReportFile involvement.
End-to-end tests through ``reportfile.parse`` live in
``test_parsers_reportfile.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from gmat_run.errors import GmatOutputParseError
from gmat_run.parsers.epoch import promote_epochs

# Scale suffixes the parser recognises. Parametrising over these catches any
# table-entry-missing or typo regression in one place.
_GREGORIAN_SUFFIXES = [
    ("A1Gregorian", "A1"),
    ("TAIGregorian", "TAI"),
    ("UTCGregorian", "UTC"),
    ("TTGregorian", "TT"),
    ("TDBGregorian", "TDB"),
]
_MODJULIAN_SUFFIXES = [
    ("A1ModJulian", "A1"),
    ("TAIModJulian", "TAI"),
    ("UTCModJulian", "UTC"),
    ("TTModJulian", "TT"),
    ("TDBModJulian", "TDB"),
]


# --- Gregorian ---------------------------------------------------------------


@pytest.mark.parametrize(("suffix", "scale"), _GREGORIAN_SUFFIXES)
def test_gregorian_each_scale_becomes_datetime64_ns(suffix: str, scale: str) -> None:
    col = f"Sat.{suffix}"
    df = pd.DataFrame({col: ["26 Nov 2026 12:00:00.000", "01 Jan 2000 11:59:28.000"]})
    promote_epochs(df)
    assert df[col].dtype == np.dtype("datetime64[ns]")
    assert df[col].iloc[0] == pd.Timestamp("2026-11-26 12:00:00")
    assert df[col].iloc[1] == pd.Timestamp("2000-01-01 11:59:28")
    assert df.attrs["epoch_scales"][col] == scale


def test_gregorian_malformed_raises(tmp_path: object) -> None:
    df = pd.DataFrame({"Sat.UTCGregorian": ["26 Nov 2026 12:00:00.000", "not a date"]})
    with pytest.raises(GmatOutputParseError) as excinfo:
        promote_epochs(df)
    assert "Sat.UTCGregorian" in str(excinfo.value)


def test_gregorian_works_on_pandas_string_dtype() -> None:
    """reportfile.parse leaves Gregorian as StringDtype; the promoter must accept it."""
    col = "Sat.UTCGregorian"
    series = pd.Series(["26 Nov 2026 12:00:00.000"], dtype="string")
    df = pd.DataFrame({col: series})
    promote_epochs(df)
    assert df[col].dtype == np.dtype("datetime64[ns]")


# --- ModJulian ---------------------------------------------------------------


@pytest.mark.parametrize(("suffix", "scale"), _MODJULIAN_SUFFIXES)
def test_modjulian_each_scale_becomes_datetime64_ns(suffix: str, scale: str) -> None:
    col = f"Sat.{suffix}"
    df = pd.DataFrame({col: [21545.0, 21545.5]})
    promote_epochs(df)
    assert df[col].dtype == np.dtype("datetime64[ns]")
    # MJD 21545.0 is J2000 by construction: 2000-01-01 12:00:00.
    assert df[col].iloc[0] == pd.Timestamp("2000-01-01 12:00:00")
    # Fractional-day round-trip: 0.5 day == 12h.
    assert df[col].iloc[1] == pd.Timestamp("2000-01-02 00:00:00")
    assert df.attrs["epoch_scales"][col] == scale


def test_modjulian_subsecond_precision() -> None:
    """MJD values round-trip well below millisecond resolution.

    Exact second-level equality isn't achievable with float64: 1/86400 day
    isn't exactly representable, so the conversion accumulates ~100 ns of
    float-rounding error. The acceptance contract is ``datetime64[ns]``
    precision — this test pins us to within 1 µs, which rules out any bug
    that drops seconds or reorders the epoch.
    """
    one_second_in_mjd = 1.0 / 86400.0
    col = "Sat.TAIModJulian"
    df = pd.DataFrame({col: [21545.0 + one_second_in_mjd]})
    promote_epochs(df)
    expected = pd.Timestamp("2000-01-01 12:00:01")
    assert abs(df[col].iloc[0] - expected) < pd.Timedelta(microseconds=1)


def test_modjulian_non_numeric_raises() -> None:
    df = pd.DataFrame({"Sat.TAIModJulian": ["21545.0"]})  # string, not float
    with pytest.raises(GmatOutputParseError) as excinfo:
        promote_epochs(df)
    assert "non-numeric" in str(excinfo.value)
    assert "Sat.TAIModJulian" in str(excinfo.value)


def test_modjulian_overflow_raises() -> None:
    """A value outside datetime64[ns]'s 1677..2262 range must fail typed."""
    df = pd.DataFrame({"Sat.TAIModJulian": [1.0e9]})  # ~2.7M years, well past 2262
    with pytest.raises(GmatOutputParseError) as excinfo:
        promote_epochs(df)
    assert "Sat.TAIModJulian" in str(excinfo.value)


# --- non-epoch columns -------------------------------------------------------


def test_non_epoch_columns_are_left_alone() -> None:
    df = pd.DataFrame(
        {
            "Sat.Earth.SMA": [6578.136, 6578.137],
            "Sat.Earth.ECC": [1e-5, 1.5e-5],
            "Sat.TotalMass": [1300000, 1300000],
        }
    )
    original = df.copy(deep=True)
    promote_epochs(df)
    pd.testing.assert_frame_equal(df, original)
    # Nothing was promoted, so no scale tags should have been added.
    assert "epoch_scales" not in df.attrs


def test_non_epoch_name_does_not_warn() -> None:
    """``Sat.Earth.SMA`` must not warn; only *Gregorian/*ModJulian suffixes do."""
    df = pd.DataFrame({"Sat.Earth.SMA": [1.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning → test failure
        promote_epochs(df)


# --- unknown-epoch warning ---------------------------------------------------


@pytest.mark.parametrize(
    "column",
    ["Sat.FooGregorian", "Sat.SomeFormatModJulian", "Foo.BarGregorian", "PlainModJulian"],
)
def test_unknown_epoch_suffix_warns_and_leaves_column_alone(column: str) -> None:
    df = pd.DataFrame({column: ["1 Jan 2000 00:00:00.000"]})
    original = df[column].copy()
    with pytest.warns(UserWarning, match="unrecognised epoch suffix"):
        promote_epochs(df)
    # Column is untouched.
    pd.testing.assert_series_equal(df[column], original)
    # Scale was not recorded.
    assert column not in df.attrs.get("epoch_scales", {})


@pytest.mark.parametrize(
    "column",
    ["Sat.utcgregorian", "Sat.modjulian", "Sat.Gregorian", "Sat.ModJulian"],
)
def test_suffix_warning_is_case_sensitive_and_requires_prefix(column: str) -> None:
    """The warning regex is strict: suffix must be exactly ``*Gregorian``/``*ModJulian``.

    Lowercased names and bare ``Gregorian``/``ModJulian`` suffixes (no scale
    prefix) do not match the heuristic and therefore do not warn. Silence is
    correct here — we don't want to be noisy about plain unrelated columns.
    """
    df = pd.DataFrame({column: ["1 Jan 2000 00:00:00.000"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        promote_epochs(df)


# --- attrs structure ---------------------------------------------------------


def test_multiple_epoch_columns_all_get_tagged() -> None:
    df = pd.DataFrame(
        {
            "Sat.UTCGregorian": ["01 Jan 2000 12:00:00.000"],
            "Sat.TAIModJulian": [21545.0],
            "Sat.Earth.SMA": [6578.136],
        }
    )
    promote_epochs(df)
    assert df.attrs["epoch_scales"] == {
        "Sat.UTCGregorian": "UTC",
        "Sat.TAIModJulian": "TAI",
    }


def test_promote_is_idempotent() -> None:
    """Calling promote_epochs twice on the same frame is a no-op on pass 2."""
    df = pd.DataFrame(
        {
            "Sat.UTCGregorian": ["01 Jan 2000 12:00:00.000"],
            "Sat.TAIModJulian": [21545.0],
        }
    )
    promote_epochs(df)
    snapshot = df.copy(deep=True)
    attrs_snapshot = dict(df.attrs["epoch_scales"])
    promote_epochs(df)
    pd.testing.assert_frame_equal(df, snapshot)
    assert df.attrs["epoch_scales"] == attrs_snapshot


def test_returns_same_frame_for_chaining() -> None:
    df = pd.DataFrame({"Sat.UTCGregorian": ["01 Jan 2000 12:00:00.000"]})
    result = promote_epochs(df)
    assert result is df
