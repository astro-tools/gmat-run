"""Golden-file tests for :func:`gmat_run.parsers.contact.parse`.

Where :mod:`tests.test_parsers_contact` builds inputs inline to exercise edge
cases at full control, these tests parse real ``ContactLocator`` output
captured from the in-repo ``Ex_ContactLocatorAllFormats.script`` and pin the
exact structure of the resulting DataFrames. A format drift in a future GMAT
release (new column, padding change, fingerprint mismatch) fails these tests
loudly.

Each fixture's source resource is named in its test class's docstring.
Regenerate after a GMAT upgrade by running the source script via
``GmatConsole --run`` and copying the resulting output files into
``tests/fixtures/contact/`` under the names this module expects. The two
``AzimuthElevationRange*`` fixtures are trimmed to the first three passes to
keep the committed fixture small while still exercising multi-pass /
multi-observer / blank-line-separator handling.

Tests run without any GMAT install — parsing is pure pandas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmat_run.parsers.contact import parse

FIXTURES = Path(__file__).parent / "fixtures" / "contact"


# ---------------------------------------------------------------------------
# Legacy — Ex_ContactLocatorAllFormats / LegacyCL
# ---------------------------------------------------------------------------


class TestLegacy:
    """``LegacyCL`` (default ReportFormat) → 4 cols, 10 rows across 3 observers."""

    FIXTURE = FIXTURES / "Ex_ContactLocatorAllFormats__Legacy.txt"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_report_format(self, df: pd.DataFrame) -> None:
        assert df.attrs["report_format"] == "Legacy"

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == ["Observer", "Start", "Stop", "Duration"]

    def test_dtypes(self, df: pd.DataFrame) -> None:
        assert df["Observer"].dtype == np.dtype("object")
        assert df["Start"].dtype == np.dtype("datetime64[ns]")
        assert df["Stop"].dtype == np.dtype("datetime64[ns]")
        assert df["Duration"].dtype == np.dtype("timedelta64[ns]")

    def test_row_count(self, df: pd.DataFrame) -> None:
        assert len(df) == 10

    def test_observer_block_order(self, df: pd.DataFrame) -> None:
        assert df.attrs["observers"] == ("AthGS", "HreGS", "GSFC")

    def test_event_counts(self, df: pd.DataFrame) -> None:
        assert df.attrs["event_counts"] == {"AthGS": 3, "HreGS": 4, "GSFC": 3}

    def test_first_row(self, df: pd.DataFrame) -> None:
        assert df["Observer"].iloc[0] == "AthGS"
        assert df["Start"].iloc[0] == pd.Timestamp("2010-01-09 20:36:24.626")
        assert df["Stop"].iloc[0] == pd.Timestamp("2010-01-09 23:05:18.684")
        assert df["Duration"].iloc[0] == pd.Timedelta(seconds=8934.0587546)

    def test_last_row(self, df: pd.DataFrame) -> None:
        assert df["Observer"].iloc[-1] == "GSFC"
        assert df["Start"].iloc[-1] == pd.Timestamp("2010-01-10 12:39:39.031")


# ---------------------------------------------------------------------------
# ContactRangeReport — single tabular layout, range columns
# ---------------------------------------------------------------------------


class TestContactRangeReport:
    """``ContactRangeCL`` → 6 cols, 10 rows chronologically merged."""

    FIXTURE = FIXTURES / "Ex_ContactLocatorAllFormats__ContactRange.txt"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_report_format(self, df: pd.DataFrame) -> None:
        assert df.attrs["report_format"] == "ContactRangeReport"

    def test_columns(self, df: pd.DataFrame) -> None:
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
        assert df["StartRange"].dtype == np.float64
        assert df["StopRange"].dtype == np.float64

    def test_row_count(self, df: pd.DataFrame) -> None:
        assert len(df) == 10

    def test_event_counts_match_legacy_totals(self, df: pd.DataFrame) -> None:
        assert df.attrs["event_counts"] == {"AthGS": 3, "HreGS": 4, "GSFC": 3}

    def test_first_row(self, df: pd.DataFrame) -> None:
        assert df["Observer"].iloc[0] == "HreGS"
        assert df["StartRange"].iloc[0] == pytest.approx(13571.316577)
        assert df["StopRange"].iloc[0] == pytest.approx(14857.240352)

    def test_chronological_observer_interleave(self, df: pd.DataFrame) -> None:
        # Distinct chronologically-merged stream — observers are NOT block-sorted.
        # Confirms the parser preserves source row order rather than re-sorting.
        assert df["Observer"].tolist() == [
            "HreGS",
            "AthGS",
            "GSFC",
            "HreGS",
            "AthGS",
            "GSFC",
            "HreGS",
            "AthGS",
            "GSFC",
            "HreGS",
        ]


# ---------------------------------------------------------------------------
# SiteViewMaxElevationReport — adds MaxElevation + MaxElevationTime
# ---------------------------------------------------------------------------


class TestSiteViewMaxElevationReport:
    """``MaxElevCL`` → 6 cols, 10 rows; adds MaxElevation/MaxElevationTime."""

    FIXTURE = FIXTURES / "Ex_ContactLocatorAllFormats__MaxElev.txt"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_report_format(self, df: pd.DataFrame) -> None:
        assert df.attrs["report_format"] == "SiteViewMaxElevationReport"

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == [
            "Observer",
            "Start",
            "Stop",
            "Duration",
            "MaxElevation",
            "MaxElevationTime",
        ]

    def test_max_elevation_dtypes(self, df: pd.DataFrame) -> None:
        assert df["MaxElevation"].dtype == np.float64
        assert df["MaxElevationTime"].dtype == np.dtype("datetime64[ns]")

    def test_epoch_scales_includes_max_elev_time(self, df: pd.DataFrame) -> None:
        assert df.attrs["epoch_scales"] == {
            "Start": "UTC",
            "Stop": "UTC",
            "MaxElevationTime": "UTC",
        }

    def test_first_row(self, df: pd.DataFrame) -> None:
        assert df["Observer"].iloc[0] == "HreGS"
        assert df["MaxElevation"].iloc[0] == pytest.approx(17.825309)
        assert df["MaxElevationTime"].iloc[0] == pd.Timestamp("2010-01-09 16:00:00")


# ---------------------------------------------------------------------------
# SiteViewMaxElevationRangeReport — superset of MaxElev + range
# ---------------------------------------------------------------------------


class TestSiteViewMaxElevationRangeReport:
    """``MaxElevRangeCL`` → 8 cols (union of MaxElev + ContactRange columns)."""

    FIXTURE = FIXTURES / "Ex_ContactLocatorAllFormats__MaxElevRange.txt"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_report_format(self, df: pd.DataFrame) -> None:
        assert df.attrs["report_format"] == "SiteViewMaxElevationRangeReport"

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == [
            "Observer",
            "Start",
            "Stop",
            "Duration",
            "MaxElevation",
            "MaxElevationTime",
            "StartRange",
            "StopRange",
        ]

    def test_first_row(self, df: pd.DataFrame) -> None:
        assert df["Observer"].iloc[0] == "HreGS"
        assert df["MaxElevation"].iloc[0] == pytest.approx(17.825309)
        assert df["StartRange"].iloc[0] == pytest.approx(13571.316577)
        assert df["StopRange"].iloc[0] == pytest.approx(14857.240352)


# ---------------------------------------------------------------------------
# AzimuthElevationRangeReport - (Pass, tick) grain, ISO-YD time format
# ---------------------------------------------------------------------------


class TestAzimuthElevationRangeReport:
    """``AzElRangeCL`` (ReportTimeFormat=ISOYD) → first 3 passes, ~340 ticks."""

    FIXTURE = FIXTURES / "Ex_ContactLocatorAllFormats__AzElRange.txt"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

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

    def test_pass_int_dtype(self, df: pd.DataFrame) -> None:
        assert df["Pass"].dtype == np.int64
        assert sorted(df["Pass"].unique().tolist()) == [1, 2, 3]

    def test_event_counts_use_pass_nunique(self, df: pd.DataFrame) -> None:
        # Each observer sees one pass in the trimmed fixture.
        assert df.attrs["event_counts"] == {"HreGS": 1, "AthGS": 1, "GSFC": 1}

    def test_isoyd_first_row_parses_to_2010_01_09(self, df: pd.DataFrame) -> None:
        # 2010-009T16:00:00.000 → 2010-01-09 16:00:00 UTC.
        assert df["Time"].iloc[0] == pd.Timestamp("2010-01-09 16:00:00")

    def test_observer_changes_at_pass_boundary(self, df: pd.DataFrame) -> None:
        # The trimmed fixture spans Pass 1 (HreGS), Pass 2 (AthGS), Pass 3 (GSFC).
        # Confirms the Pass column groups distinct observers and no rows leak
        # between passes despite the blank-line separators.
        first_per_pass = df.drop_duplicates(subset="Pass")[["Pass", "Observer"]]
        assert first_per_pass["Observer"].tolist() == ["HreGS", "AthGS", "GSFC"]


# ---------------------------------------------------------------------------
# AzimuthElevationRangeRangeRateReport — adds RangeRate, UTC-MJD time format
# ---------------------------------------------------------------------------


class TestAzimuthElevationRangeRangeRateReport:
    """``AzElRangeRRCL`` (ReportTimeFormat=UTCMJD) → adds RangeRate."""

    FIXTURE = FIXTURES / "Ex_ContactLocatorAllFormats__AzElRangeRR.txt"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_report_format(self, df: pd.DataFrame) -> None:
        assert df.attrs["report_format"] == "AzimuthElevationRangeRangeRateReport"

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == [
            "Pass",
            "Observer",
            "Time",
            "Azimuth",
            "Elevation",
            "Range",
            "RangeRate",
        ]

    def test_range_rate_first_row(self, df: pd.DataFrame) -> None:
        assert df["RangeRate"].iloc[0] == pytest.approx(0.094483)

    def test_utcmjd_time_close_to_iso(self, df: pd.DataFrame) -> None:
        # MJD 25206.16666667 = 2010-01-09 16:00:00 UTC ± float drift.
        first = df["Time"].iloc[0]
        assert (first - pd.Timestamp("2010-01-09 16:00:00")).total_seconds() == (
            pytest.approx(0.0, abs=1e-3)
        )
