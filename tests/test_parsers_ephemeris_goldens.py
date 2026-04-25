"""Golden-file tests for :func:`gmat_run.parsers.ephemeris.parse`.

Pins the exact structure of the DataFrame that the parser produces from a real
GMAT-emitted CCSDS-OEM file. A format drift in a future GMAT release (header
field added, padding change, alignment change) trips these tests loudly.

The fixture is captured once from a real ``Mission.run`` against the
:file:`tests/integration/fixtures/Ex_LEOEphemeris.script` mission and committed.
Regenerate after a GMAT upgrade by running that script via
``Mission.load(...).run(...)`` and copying the resulting ``.oem`` over the
fixture path. Tests here run without any GMAT install — parsing is pure
pandas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmat_run.parsers.ephemeris import parse

FIXTURES = Path(__file__).parent / "fixtures" / "ephemeris"


class TestExLEOEphemeris:
    """``Ex_LEOEphemeris.script`` -> 6-hour LEO propagation, 60 s steps."""

    FIXTURE = FIXTURES / "Ex_LEOEphemeris.oem"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == ["Epoch", "X", "Y", "Z", "VX", "VY", "VZ"]

    def test_row_count(self, df: pd.DataFrame) -> None:
        # 6 h * 60 steps/h + endpoint = 361 records.
        assert len(df) == 361

    def test_epoch_dtype_and_scale(self, df: pd.DataFrame) -> None:
        assert df["Epoch"].dtype == np.dtype("datetime64[ns]")
        assert df.attrs["epoch_scales"] == {"Epoch": "UTC"}

    def test_state_columns_are_float64(self, df: pd.DataFrame) -> None:
        for column in ["X", "Y", "Z", "VX", "VY", "VZ"]:
            assert df[column].dtype == np.float64, column

    def test_first_record(self, df: pd.DataFrame) -> None:
        assert df["Epoch"].iloc[0] == pd.Timestamp("2026-01-01 12:00:00")
        assert df["X"].iloc[0] == pytest.approx(-5936.162914914940)
        assert df["Y"].iloc[0] == pytest.approx(1590.590059191038)
        assert df["VZ"].iloc[0] == pytest.approx(2.206977572872483e-16)

    def test_last_record(self, df: pd.DataFrame) -> None:
        assert df["Epoch"].iloc[-1] == pd.Timestamp("2026-01-01 18:00:00")
        assert df["X"].iloc[-1] == pytest.approx(3250.046911877658)
        assert df["VX"].iloc[-1] == pytest.approx(-5.726194156065000)

    def test_metadata_attrs(self, df: pd.DataFrame) -> None:
        assert df.attrs["object_name"] == "Sat"
        assert df.attrs["central_body"] == "Earth"
        assert df.attrs["coordinate_system"] == "EME2000"
        assert df.attrs["time_scale"] == "UTC"
        assert df.attrs["interpolation"] == "LAGRANGE"
        # GMAT pads the value with a trailing space (`INTERPOLATION_DEGREE = 7 `);
        # the parser must strip and coerce to int.
        assert df.attrs["interpolation_degree"] == 7

    def test_file_header_attrs(self, df: pd.DataFrame) -> None:
        fh = df.attrs["file_header"]
        assert fh["CCSDS_OEM_VERS"] == "1.0"
        assert fh["ORIGINATOR"] == "GMAT USER"

    def test_no_segments_attr_for_single_segment(self, df: pd.DataFrame) -> None:
        # Single-segment file: per-segment list is omitted to keep the common
        # case clean.
        assert "segments" not in df.attrs
