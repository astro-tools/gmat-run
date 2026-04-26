"""Golden-file tests for :func:`gmat_run.parsers.stk_ephemeris.parse`.

Pins the exact structure of the DataFrame the parser produces from a real
GMAT-emitted STK-TimePosVel file. Format drift in a future GMAT release
(banner version bump, header field added or removed, padding change) trips
these tests loudly.

The fixture is captured once from a real ``Mission.run`` against the
:file:`tests/integration/fixtures/Ex_STKEphemeris.script` mission and
committed. Regenerate after a GMAT upgrade by running that script via
``Mission.load(...).run(...)`` and copying the resulting ``.e`` over the
fixture path. Tests here run without any GMAT install — parsing is pure
pandas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmat_run.parsers.stk_ephemeris import parse

FIXTURES = Path(__file__).parent / "fixtures" / "ephemeris"


class TestExSTKEphemeris:
    """``Ex_STKEphemeris.script`` -> 6-hour LEO propagation, 60 s steps."""

    FIXTURE = FIXTURES / "Ex_STKEphemeris.e"

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
        # First record is at offset 0.0 → Epoch == ScenarioEpoch exactly.
        assert df["Epoch"].iloc[0] == pd.Timestamp("2026-01-01 12:00:00")
        assert df["X"].iloc[0] == pytest.approx(-5936.162914914940)
        assert df["Y"].iloc[0] == pytest.approx(1590.590059191038)
        assert df["VZ"].iloc[0] == pytest.approx(2.206977572872483e-16)

    def test_last_record(self, df: pd.DataFrame) -> None:
        # Last offset is the 6 h mark; floor to ms to absorb sub-microsecond
        # integrator drift in GMAT's offset value.
        assert df["Epoch"].iloc[-1].floor("ms") == pd.Timestamp("2026-01-01 18:00:00")
        # State values are committed to the fixture at the precision GMAT
        # writes; pin to a safe rtol.
        assert df["X"].iloc[-1] == pytest.approx(3250.0469118776, rel=1e-9)
        assert df["VX"].iloc[-1] == pytest.approx(-5.726194156065, rel=1e-9)

    def test_metadata_attrs(self, df: pd.DataFrame) -> None:
        assert df.attrs["central_body"] == "Earth"
        assert df.attrs["coordinate_system"] == "J2000"
        assert df.attrs["interpolation"] == "Lagrange"
        assert df.attrs["distance_unit"] == "Kilometers"
        assert df.attrs["scenario_epoch"] == "01 Jan 2026 12:00:00.000"
        assert df.attrs["time_scale"] == "UTC"

    def test_file_header_attrs(self, df: pd.DataFrame) -> None:
        fh = df.attrs["file_header"]
        assert fh["version"] == "stk.v.10.0"
        assert fh["comments"] == ["# WrittenBy    GMAT R2026a"]
