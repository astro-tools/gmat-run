"""Unit tests for :func:`gmat_run.parsers.contact.parse`.

Inline ``tmp_path`` fixtures keep encoding and structure visible at each call
site, mirroring :mod:`tests.test_parsers_ephemeris`. Edge cases (empty files,
malformed headers, encoding variants, ReportTimeFormat axes) are covered here;
shape pinning against full-byte real-GMAT captures lives in
:mod:`tests.test_parsers_contact_goldens`.
"""

# ruff: noqa: E501  -- ContactLocator's text format pads every column for a
# right-justified emit; even the smallest real-format fixture exceeds 100
# columns. Trimming widths here would diverge from what the parser sees in
# CI integration runs and weaken the unit-test signal.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmat_run.errors import GmatOutputParseError
from gmat_run.parsers.contact import parse

# --- helpers -----------------------------------------------------------------


def _write(path: Path, content: str, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Write ``content`` verbatim with the requested encoding and line ending."""
    path.write_bytes(content.replace("\n", newline).encode(encoding))
    return path


# --- minimal Legacy fixtures -------------------------------------------------

# A two-observer Legacy file mirroring the byte shape of GMAT's output.
_LEGACY_TWO_OBSERVERS = """\
Target: Sat

Observer: AthGS
Start Time (UTC)            Stop Time (UTC)               Duration (s)
09 Jan 2010 20:36:24.626    09 Jan 2010 23:05:18.684      8934.0587546
10 Jan 2010 04:31:56.141    10 Jan 2010 07:00:47.587      8931.4459472


Number of events : 2


Observer: HreGS
Start Time (UTC)            Stop Time (UTC)               Duration (s)
09 Jan 2010 16:00:00.000    09 Jan 2010 16:50:53.222      3053.2220792


Number of events : 1


"""

_LEGACY_EMPTY_OBSERVER = """\
Target: Sat

Observer: AthGS
Start Time (UTC)            Stop Time (UTC)               Duration (s)


Number of events : 0


"""


# --- minimal Tabular fixtures ------------------------------------------------

_CONTACT_RANGE = """\
Target: Sat

Observer     Duration (s)     Start Time (UTCGregorian)     Stop Time (UTCGregorian)     Start Range (km)     Stop Range (km)
--------     ------------     -------------------------     ------------------------     ----------------     ---------------
   HreGS      3053.222079      09 Jan 2010 16:00:00.000     09 Jan 2010 16:50:53.222         13571.316577        14857.240352
   AthGS      8934.058755      09 Jan 2010 20:36:24.626     09 Jan 2010 23:05:18.684         14861.913473        14860.735274
"""

_AZELRANGE_ISO = """\
Target: Sat

Pass Number     Observer     Time (ISO-YD)             Azimuth (deg)     Elevation (deg)     Range (km)
-----------     --------     ---------------------     -------------     ---------------     ------------
          1        HreGS     2010-009T16:00:00.000        173.586003           17.825309     13571.316577
          1        HreGS     2010-009T16:01:00.000        172.612105           17.759523     13577.416988

          2        AthGS     2010-009T20:36:24.626        249.687221            5.000000     14861.913473
"""

_AZELRANGE_RR_MJD = """\
Target: Sat

Pass Number     Observer     Time (UTC-MJD)     Azimuth (deg)     Elevation (deg)     Range (km)       Range Rate (km/s)
-----------     --------     --------------     -------------     ---------------     ------------     -----------------
          1        HreGS     25206.16666667        173.586003           17.825309     13571.316577              0.094483
          1        HreGS     25206.16805556        171.639909           17.684536     13584.378772              0.123194
"""


# --- target-line front matter -------------------------------------------------


def test_empty_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GmatOutputParseError, match="empty"):
        parse(_write(tmp_path / "empty.txt", ""))


def test_whitespace_only_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GmatOutputParseError, match="empty"):
        parse(_write(tmp_path / "blank.txt", "\n   \n\n"))


def test_missing_target_line_raises(tmp_path: Path) -> None:
    content = "Observer: AthGS\nStart Time (UTC)  Stop Time (UTC)  Duration (s)\n"
    with pytest.raises(GmatOutputParseError, match="Target"):
        parse(_write(tmp_path / "no_target.txt", content))


def test_target_with_empty_name_raises(tmp_path: Path) -> None:
    with pytest.raises(GmatOutputParseError, match="missing a target name"):
        parse(_write(tmp_path / "empty_target.txt", "Target: \n"))


def test_target_carries_to_attrs(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "legacy.txt", _LEGACY_TWO_OBSERVERS))
    assert df.attrs["target"] == "Sat"


def test_time_scale_attr_is_always_utc(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "legacy.txt", _LEGACY_TWO_OBSERVERS))
    assert df.attrs["time_scale"] == "UTC"


# --- Legacy: happy path -------------------------------------------------------


class TestLegacy:
    @pytest.fixture
    def df(self, tmp_path: Path) -> pd.DataFrame:
        return parse(_write(tmp_path / "legacy.txt", _LEGACY_TWO_OBSERVERS))

    def test_report_format(self, df: pd.DataFrame) -> None:
        assert df.attrs["report_format"] == "Legacy"

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == ["Observer", "Start", "Stop", "Duration"]

    def test_dtypes(self, df: pd.DataFrame) -> None:
        assert df["Observer"].dtype == np.dtype("object")
        assert df["Start"].dtype == np.dtype("datetime64[ns]")
        assert df["Stop"].dtype == np.dtype("datetime64[ns]")
        assert df["Duration"].dtype == np.dtype("timedelta64[ns]")

    def test_row_count_aggregates_observers(self, df: pd.DataFrame) -> None:
        assert len(df) == 3  # 2 events from AthGS + 1 from HreGS

    def test_observer_column_groups_blocks(self, df: pd.DataFrame) -> None:
        assert df["Observer"].tolist() == ["AthGS", "AthGS", "HreGS"]

    def test_event_counts_attr(self, df: pd.DataFrame) -> None:
        assert df.attrs["event_counts"] == {"AthGS": 2, "HreGS": 1}

    def test_observers_attr_preserves_block_order(self, df: pd.DataFrame) -> None:
        assert df.attrs["observers"] == ("AthGS", "HreGS")

    def test_epoch_scales_attr(self, df: pd.DataFrame) -> None:
        assert df.attrs["epoch_scales"] == {"Start": "UTC", "Stop": "UTC"}

    def test_first_row_values(self, df: pd.DataFrame) -> None:
        assert df["Start"].iloc[0] == pd.Timestamp("2010-01-09 20:36:24.626")
        assert df["Stop"].iloc[0] == pd.Timestamp("2010-01-09 23:05:18.684")
        assert df["Duration"].iloc[0] == pd.Timedelta(seconds=8934.0587546)

    def test_observer_with_zero_events_returns_empty_frame(self, tmp_path: Path) -> None:
        df = parse(_write(tmp_path / "empty_obs.txt", _LEGACY_EMPTY_OBSERVER))
        assert df.attrs["report_format"] == "Legacy"
        assert len(df) == 0
        # Schema is preserved even on a zero-event frame.
        assert list(df.columns) == ["Observer", "Start", "Stop", "Duration"]
        assert df.attrs["event_counts"] == {"AthGS": 0}
        assert df.attrs["observers"] == ("AthGS",)


# --- Legacy: error paths ------------------------------------------------------


def test_legacy_no_observer_blocks_raises(tmp_path: Path) -> None:
    # Header only — nothing follows the Target line.
    with pytest.raises(GmatOutputParseError, match="missing column header"):
        # Without an Observer: line the format detector treats it as tabular,
        # which then fails on missing header. That's correct behaviour: a file
        # with only "Target: Sat" is malformed regardless of intended variant.
        parse(_write(tmp_path / "target_only.txt", "Target: Sat\n"))


def test_legacy_observer_with_no_name_raises(tmp_path: Path) -> None:
    content = (
        "Target: Sat\n"
        "\n"
        "Observer: \n"
        "Start Time (UTC)  Stop Time (UTC)  Duration (s)\n"
        "Number of events : 0\n"
    )
    with pytest.raises(GmatOutputParseError, match=r"Observer.*missing a name"):
        parse(_write(tmp_path / "anon_observer.txt", content))


def test_legacy_count_mismatch_raises(tmp_path: Path) -> None:
    bad = _LEGACY_TWO_OBSERVERS.replace("Number of events : 2", "Number of events : 99")
    with pytest.raises(GmatOutputParseError, match="declared 99 events but parsed 2"):
        parse(_write(tmp_path / "mismatch.txt", bad))


def test_legacy_malformed_event_count_raises(tmp_path: Path) -> None:
    bad = _LEGACY_TWO_OBSERVERS.replace("Number of events : 2", "Number of events : NOPE")
    with pytest.raises(GmatOutputParseError, match=r"malformed.*Number of events"):
        parse(_write(tmp_path / "bad_count.txt", bad))


def test_legacy_data_row_wrong_column_count_raises(tmp_path: Path) -> None:
    # Drop the Duration column from one row — leaves 2 tokens instead of 3.
    bad = _LEGACY_TWO_OBSERVERS.replace(
        "10 Jan 2010 04:31:56.141    10 Jan 2010 07:00:47.587      8931.4459472",
        "10 Jan 2010 04:31:56.141    10 Jan 2010 07:00:47.587",
    )
    with pytest.raises(GmatOutputParseError, match="expected 3 columns"):
        parse(_write(tmp_path / "short_row.txt", bad))


def test_legacy_malformed_gregorian_value_raises(tmp_path: Path) -> None:
    bad = _LEGACY_TWO_OBSERVERS.replace("09 Jan 2010 20:36:24.626", "ALPHA BETA GAMMA")
    with pytest.raises(GmatOutputParseError, match="malformed Gregorian"):
        parse(_write(tmp_path / "bad_time.txt", bad))


# --- Tabular: ContactRangeReport ---------------------------------------------


class TestContactRangeReport:
    @pytest.fixture
    def df(self, tmp_path: Path) -> pd.DataFrame:
        return parse(_write(tmp_path / "cr.txt", _CONTACT_RANGE))

    def test_report_format(self, df: pd.DataFrame) -> None:
        assert df.attrs["report_format"] == "ContactRangeReport"

    def test_columns_in_source_order(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == [
            "Observer",
            "Duration",
            "Start",
            "Stop",
            "StartRange",
            "StopRange",
        ]

    def test_dtypes(self, df: pd.DataFrame) -> None:
        assert df["Observer"].dtype == np.dtype("object")
        assert df["Duration"].dtype == np.dtype("timedelta64[ns]")
        assert df["Start"].dtype == np.dtype("datetime64[ns]")
        assert df["Stop"].dtype == np.dtype("datetime64[ns]")
        assert df["StartRange"].dtype == np.float64
        assert df["StopRange"].dtype == np.float64

    def test_event_counts_derived_per_observer(self, df: pd.DataFrame) -> None:
        # Two events: one HreGS, one AthGS.
        assert df.attrs["event_counts"] == {"HreGS": 1, "AthGS": 1}

    def test_observers_attr_preserves_first_seen_order(self, df: pd.DataFrame) -> None:
        assert df.attrs["observers"] == ("HreGS", "AthGS")

    def test_first_row_values(self, df: pd.DataFrame) -> None:
        assert df["Observer"].iloc[0] == "HreGS"
        assert df["Start"].iloc[0] == pd.Timestamp("2010-01-09 16:00:00")
        assert df["StartRange"].iloc[0] == pytest.approx(13571.316577)


# --- Tabular: AzEl variants ((Pass, tick) grain) ------------------------------


class TestAzimuthElevationRangeReport:
    @pytest.fixture
    def df(self, tmp_path: Path) -> pd.DataFrame:
        return parse(_write(tmp_path / "az.txt", _AZELRANGE_ISO))

    def test_report_format(self, df: pd.DataFrame) -> None:
        assert df.attrs["report_format"] == "AzimuthElevationRangeReport"

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == [
            "Pass",
            "Observer",
            "Time",
            "Azimuth",
            "Elevation",
            "Range",
        ]

    def test_pass_column_is_int(self, df: pd.DataFrame) -> None:
        assert df["Pass"].dtype == np.int64
        assert df["Pass"].tolist() == [1, 1, 2]

    def test_isoyd_time_parsed(self, df: pd.DataFrame) -> None:
        assert df["Time"].dtype == np.dtype("datetime64[ns]")
        # Day-of-year 9 of 2010 = 2010-01-09.
        assert df["Time"].iloc[0] == pd.Timestamp("2010-01-09 16:00:00")

    def test_event_counts_uses_pass_nunique(self, df: pd.DataFrame) -> None:
        # HreGS sees one Pass (#1), AthGS sees one Pass (#2). Multiple ticks
        # within a Pass do not inflate the event count.
        assert df.attrs["event_counts"] == {"HreGS": 1, "AthGS": 1}

    def test_blank_lines_between_passes_skipped(self, df: pd.DataFrame) -> None:
        # 2 ticks in Pass 1 + 1 tick in Pass 2 = 3 rows; the blank line that
        # separates passes in the source must not become an extra row.
        assert len(df) == 3


class TestAzimuthElevationRangeRangeRateReport:
    @pytest.fixture
    def df(self, tmp_path: Path) -> pd.DataFrame:
        return parse(_write(tmp_path / "azrr.txt", _AZELRANGE_RR_MJD))

    def test_report_format(self, df: pd.DataFrame) -> None:
        assert df.attrs["report_format"] == "AzimuthElevationRangeRangeRateReport"

    def test_range_rate_column_present(self, df: pd.DataFrame) -> None:
        assert "RangeRate" in df.columns
        assert df["RangeRate"].dtype == np.float64
        assert df["RangeRate"].iloc[0] == pytest.approx(0.094483)

    def test_utc_mjd_time_parsed(self, df: pd.DataFrame) -> None:
        assert df["Time"].dtype == np.dtype("datetime64[ns]")
        # MJD 25206.16666667 = 2010-01-09 16:00:00 UTC ± floating-point drift.
        assert (
            df["Time"].iloc[0] - pd.Timestamp("2010-01-09 16:00:00")
        ).total_seconds() == pytest.approx(0.0, abs=1e-3)


# --- Tabular: error paths ----------------------------------------------------


def test_tabular_missing_dashes_divider_raises(tmp_path: Path) -> None:
    bad = _CONTACT_RANGE.replace(
        "--------     ------------     -------------------------     "
        "------------------------     ----------------     ---------------\n",
        "",
    )
    with pytest.raises(GmatOutputParseError, match="divider"):
        parse(_write(tmp_path / "no_dashes.txt", bad))


def test_tabular_unknown_column_header_raises(tmp_path: Path) -> None:
    bad = _CONTACT_RANGE.replace("Start Range (km)", "Lunar Coffee Pressure (mP)")
    with pytest.raises(GmatOutputParseError, match="unknown column header"):
        parse(_write(tmp_path / "unknown_col.txt", bad))


def test_tabular_unknown_fingerprint_raises(tmp_path: Path) -> None:
    # Drop the StopRange column from both header and a row to produce a
    # column-set that matches no known fingerprint.
    bad = _CONTACT_RANGE.replace("     Stop Range (km)", "")
    bad = bad.replace("     ---------------", "")
    bad = bad.replace("        14857.240352", "")
    bad = bad.replace("        14860.735274", "")
    with pytest.raises(GmatOutputParseError, match="matches no known"):
        parse(_write(tmp_path / "bad_fingerprint.txt", bad))


def test_tabular_data_row_wrong_column_count_raises(tmp_path: Path) -> None:
    bad = _CONTACT_RANGE.replace("        14857.240352", "")
    with pytest.raises(GmatOutputParseError, match="expected 6 columns"):
        parse(_write(tmp_path / "short_row.txt", bad))


def test_tabular_malformed_isoyd_value_raises(tmp_path: Path) -> None:
    bad = _AZELRANGE_ISO.replace("2010-009T16:00:00.000", "FOOBARZQ")
    with pytest.raises(GmatOutputParseError, match="malformed ISO-YD"):
        parse(_write(tmp_path / "bad_iso.txt", bad))


def test_tabular_malformed_modjulian_value_raises(tmp_path: Path) -> None:
    bad = _AZELRANGE_RR_MJD.replace("25206.16666667", "NOTNUM")
    with pytest.raises(GmatOutputParseError, match="non-numeric ModJulian"):
        parse(_write(tmp_path / "bad_mjd.txt", bad))


def test_tabular_unrecognised_time_token_raises(tmp_path: Path) -> None:
    bad = _AZELRANGE_ISO.replace("(ISO-YD)", "(BIZARRO)")
    with pytest.raises(GmatOutputParseError, match="unrecognised time format"):
        parse(_write(tmp_path / "bad_token.txt", bad))


def test_tabular_non_integer_pass_raises(tmp_path: Path) -> None:
    bad = _AZELRANGE_ISO.replace("          1        HreGS", "       FOOX        HreGS")
    with pytest.raises(GmatOutputParseError, match="non-integer"):
        parse(_write(tmp_path / "bad_pass.txt", bad))


# --- encoding / line-ending variants -----------------------------------------


def test_crlf_line_endings_accepted(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "crlf.txt", _LEGACY_TWO_OBSERVERS, newline="\r\n"))
    assert len(df) == 3


def test_utf8_bom_stripped(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "bom.txt", _LEGACY_TWO_OBSERVERS, encoding="utf-8-sig"))
    assert df.attrs["target"] == "Sat"
    assert df.attrs["report_format"] == "Legacy"


def test_leading_blank_lines_before_target_accepted(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "leading.txt", "\n\n   \n" + _LEGACY_TWO_OBSERVERS))
    assert df.attrs["target"] == "Sat"


def test_legacy_stray_row_between_blocks_raises(tmp_path: Path) -> None:
    # A data row sneaking in between the previous block's "Number of events"
    # line and the next "Observer:" header — current is None at that point.
    bad = (
        "Target: Sat\n"
        "\n"
        "Observer: AthGS\n"
        "Start Time (UTC)            Stop Time (UTC)               Duration (s)\n"
        "09 Jan 2010 20:36:24.626    09 Jan 2010 23:05:18.684      8934.0587546\n"
        "\n"
        "Number of events : 1\n"
        "\n"
        "STRAY-LINE\n"
        "\n"
        "Observer: HreGS\n"
        "Start Time (UTC)            Stop Time (UTC)               Duration (s)\n"
        "Number of events : 0\n"
    )
    with pytest.raises(GmatOutputParseError, match="content before first 'Observer:'"):
        parse(_write(tmp_path / "stray.txt", bad))


def test_tabular_blank_lines_between_header_and_dashes_accepted(tmp_path: Path) -> None:
    # The dashes-loop is tolerant of blank lines after the header before the
    # divider — exercise that tolerance.
    bad = _CONTACT_RANGE.replace(
        "Stop Range (km)\n--------",
        "Stop Range (km)\n\n   \n--------",
    )
    df = parse(_write(tmp_path / "blank_gap.txt", bad))
    assert df.attrs["report_format"] == "ContactRangeReport"


def test_tabular_eof_after_header_no_dashes_raises(tmp_path: Path) -> None:
    # Header line then EOF (no dashes, no data).
    bad = (
        "Target: Sat\n"
        "\n"
        "Observer     Duration (s)     Start Time (UTCGregorian)     "
        "Stop Time (UTCGregorian)     Start Range (km)     Stop Range (km)\n"
    )
    with pytest.raises(GmatOutputParseError, match="missing '---' divider"):
        parse(_write(tmp_path / "no_divider_eof.txt", bad))


def test_tabular_time_column_missing_scale_token_raises(tmp_path: Path) -> None:
    # Strip the parens suffix off a time column.
    bad = _CONTACT_RANGE.replace("Start Time (UTCGregorian)", "Start Time")
    with pytest.raises(GmatOutputParseError, match="no '\\(scale\\)' suffix"):
        parse(_write(tmp_path / "no_scale.txt", bad))


def test_tabular_non_numeric_float_value_raises(tmp_path: Path) -> None:
    bad = _CONTACT_RANGE.replace("13571.316577", "NOTANUMBER")
    with pytest.raises(GmatOutputParseError, match="non-numeric value"):
        parse(_write(tmp_path / "bad_float.txt", bad))


def test_tabular_non_numeric_duration_raises(tmp_path: Path) -> None:
    bad = _CONTACT_RANGE.replace("3053.222079", "NOPE")
    with pytest.raises(GmatOutputParseError, match="non-numeric seconds"):
        parse(_write(tmp_path / "bad_dur.txt", bad))


def test_tabular_dashes_divider_wrong_token_count_raises(tmp_path: Path) -> None:
    # Drop one dash-group from the divider line; header still has 6 columns.
    bad = _CONTACT_RANGE.replace(
        "--------     ------------     -------------------------     "
        "------------------------     ----------------     ---------------",
        "--------     ------------     -------------------------     "
        "------------------------     ----------------",
    )
    with pytest.raises(GmatOutputParseError, match="dashes divider has 5 tokens"):
        parse(_write(tmp_path / "short_divider.txt", bad))


def test_tabular_modjulian_out_of_range_raises(tmp_path: Path) -> None:
    # ``datetime64[ns]`` covers ~1677..2262. MJD value far outside that range
    # overflows during ``GMAT_MJD_EPOCH + to_timedelta``.
    bad = _AZELRANGE_RR_MJD.replace("25206.16666667", "9999999999.0")
    with pytest.raises(GmatOutputParseError, match="outside the representable"):
        parse(_write(tmp_path / "huge_mjd.txt", bad))
