"""Golden-file tests for :func:`gmat_run.parsers.reportfile.parse`.

Unlike :mod:`tests.test_parsers_reportfile` — which builds inputs inline via
``tmp_path`` to exercise edge cases at full control — these tests parse real
``ReportFile`` output captured from stock GMAT R2026a ``Ex_*.script`` samples
and pin the exact structure of the resulting DataFrames. A format drift in a
future GMAT release (new column on a stock resource, padding change, epoch
format change, new segment boundary) fails these tests loudly.

Each fixture's source script is named in its test class's docstring.
Regenerate after a GMAT upgrade by running each source script via
``GmatConsole --run`` and copying the resulting output file into
``tests/fixtures/reportfile/`` under the name this module expects.

Tests run without any GMAT install — parsing is pure pandas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmat_run.parsers.reportfile import parse

FIXTURES = Path(__file__).parent / "fixtures" / "reportfile"


# ---------------------------------------------------------------------------
# Ex_TLE_Propagation — UTCGregorian + Earth Cartesian state
# ---------------------------------------------------------------------------


class TestExTLEPropagation:
    """Stock ``Ex_TLE_Propagation.script`` → 7 cols, 289 data rows, UTC epoch."""

    FIXTURE = FIXTURES / "Ex_TLE_Propagation.txt"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == [
            "ExampleSat.UTCGregorian",
            "ExampleSat.X",
            "ExampleSat.Y",
            "ExampleSat.Z",
            "ExampleSat.VX",
            "ExampleSat.VY",
            "ExampleSat.VZ",
        ]

    def test_row_count(self, df: pd.DataFrame) -> None:
        assert len(df) == 289

    def test_epoch_dtype_and_scale(self, df: pd.DataFrame) -> None:
        assert df["ExampleSat.UTCGregorian"].dtype == np.dtype("datetime64[ns]")
        assert df.attrs["epoch_scales"] == {"ExampleSat.UTCGregorian": "UTC"}

    def test_numeric_columns_are_float64(self, df: pd.DataFrame) -> None:
        for column in df.columns[1:]:
            assert df[column].dtype == np.float64, column

    def test_first_row(self, df: pd.DataFrame) -> None:
        assert df["ExampleSat.UTCGregorian"].iloc[0] == pd.Timestamp("2002-05-04 11:45:15.695")
        assert df["ExampleSat.X"].iloc[0] == pytest.approx(-6775.105759552147)
        assert df["ExampleSat.VZ"].iloc[0] == pytest.approx(7.36232812682508)

    def test_last_row(self, df: pd.DataFrame) -> None:
        assert df["ExampleSat.UTCGregorian"].iloc[-1] == pd.Timestamp("2002-05-05 11:45:15.695")
        assert df["ExampleSat.X"].iloc[-1] == pytest.approx(-244.1779559915077)


# ---------------------------------------------------------------------------
# Ex_FiniteBurnParameters — A1ModJulian numeric epoch + thruster columns
# ---------------------------------------------------------------------------


class TestExFiniteBurnParameters:
    """Stock ``Ex_FiniteBurnParameters.script`` → 11 cols, 121 data rows.

    The only fixture covering the ``A1ModJulian`` epoch-promotion path end to
    end, and the only one with mixed numeric dtypes: ``Isp`` and
    ``ThrustMagnitude`` are all-integer columns and must come out as
    ``int64``, not ``float64``.
    """

    FIXTURE = FIXTURES / "Ex_FiniteBurnParameters.txt"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == [
            "DefaultSC.A1ModJulian",
            "DefaultSC.ChemicalThruster1.MassFlowRate",
            "DefaultSC.ChemicalThruster1.Isp",
            "DefaultSC.ChemicalThruster1.ThrustMagnitude",
            "FiniteBurn1.TotalAcceleration1",
            "FiniteBurn1.TotalAcceleration2",
            "FiniteBurn1.TotalAcceleration3",
            "FiniteBurn1.TotalMassFlowRate",
            "FiniteBurn1.TotalThrust1",
            "FiniteBurn1.TotalThrust2",
            "FiniteBurn1.TotalThrust3",
        ]

    def test_row_count(self, df: pd.DataFrame) -> None:
        assert len(df) == 121

    def test_epoch_dtype_and_scale(self, df: pd.DataFrame) -> None:
        assert df["DefaultSC.A1ModJulian"].dtype == np.dtype("datetime64[ns]")
        assert df.attrs["epoch_scales"] == {"DefaultSC.A1ModJulian": "A1"}

    def test_integer_columns_infer_int64(self, df: pd.DataFrame) -> None:
        """All-integer columns must be ``int64``, not silently promoted to float."""
        assert df["DefaultSC.ChemicalThruster1.Isp"].dtype == np.int64
        assert df["DefaultSC.ChemicalThruster1.ThrustMagnitude"].dtype == np.int64
        assert (df["DefaultSC.ChemicalThruster1.Isp"] == 300).all()
        assert (df["DefaultSC.ChemicalThruster1.ThrustMagnitude"] == 10).all()

    def test_float_columns_are_float64(self, df: pd.DataFrame) -> None:
        for column in [
            "DefaultSC.ChemicalThruster1.MassFlowRate",
            "FiniteBurn1.TotalAcceleration1",
            "FiniteBurn1.TotalAcceleration2",
            "FiniteBurn1.TotalMassFlowRate",
            "FiniteBurn1.TotalThrust2",
        ]:
            assert df[column].dtype == np.float64, column

    def test_first_row(self, df: pd.DataFrame) -> None:
        # A1 MJD 21545.00000039794 → ~0.034 s past J2000 A1 epoch.
        assert df["DefaultSC.A1ModJulian"].iloc[0] == pd.Timestamp("2000-01-01 12:00:00.034382041")
        assert df["FiniteBurn1.TotalAcceleration2"].iloc[0] == pytest.approx(4.195051398634114e-06)
        # Scientific notation must survive float coercion.
        assert df["FiniteBurn1.TotalAcceleration3"].iloc[0] == pytest.approx(5.70755292331172e-07)

    def test_last_row(self, df: pd.DataFrame) -> None:
        assert df["DefaultSC.A1ModJulian"].iloc[-1] == pd.Timestamp("2000-01-01 15:20:00.034381239")
        assert df["FiniteBurn1.TotalThrust1"].iloc[-1] == pytest.approx(2.370270146782424)


# ---------------------------------------------------------------------------
# Ex_AnalyticMassProperties — UTCGregorian + mass-property tensor
# ---------------------------------------------------------------------------


class TestExAnalyticMassProperties:
    """Stock ``Ex_AnalyticMassProperties.script`` → 10 cols, 215 data rows."""

    FIXTURE = FIXTURES / "Ex_AnalyticMassProperties.txt"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == [
            "Sat1.UTCGregorian",
            "Sat1.SystemCenterOfMassX",
            "Sat1.SystemCenterOfMassY",
            "Sat1.SystemCenterOfMassZ",
            "Sat1.SystemMomentOfInertiaXX",
            "Sat1.SystemMomentOfInertiaXY",
            "Sat1.SystemMomentOfInertiaXZ",
            "Sat1.SystemMomentOfInertiaYY",
            "Sat1.SystemMomentOfInertiaYZ",
            "Sat1.SystemMomentOfInertiaZZ",
        ]

    def test_row_count(self, df: pd.DataFrame) -> None:
        assert len(df) == 215

    def test_epoch_dtype_and_scale(self, df: pd.DataFrame) -> None:
        assert df["Sat1.UTCGregorian"].dtype == np.dtype("datetime64[ns]")
        assert df.attrs["epoch_scales"] == {"Sat1.UTCGregorian": "UTC"}

    def test_numeric_columns_are_float64(self, df: pd.DataFrame) -> None:
        for column in df.columns[1:]:
            assert df[column].dtype == np.float64, column

    def test_first_row(self, df: pd.DataFrame) -> None:
        assert df["Sat1.UTCGregorian"].iloc[0] == pd.Timestamp("2023-02-03 12:00:00")
        assert df["Sat1.SystemCenterOfMassX"].iloc[0] == pytest.approx(-0.7837837837837838)
        assert df["Sat1.SystemMomentOfInertiaXX"].iloc[0] == pytest.approx(35797.67567567567)

    def test_last_row(self, df: pd.DataFrame) -> None:
        assert df["Sat1.UTCGregorian"].iloc[-1] == pd.Timestamp("2023-02-03 15:33:20")
        assert df["Sat1.SystemMomentOfInertiaZZ"].iloc[-1] == pytest.approx(20549.61301401532)
