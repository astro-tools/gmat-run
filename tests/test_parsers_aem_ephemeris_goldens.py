"""Golden-file tests for :func:`gmat_run.parsers.aem_ephemeris.parse`.

Pins the exact structure of the DataFrame the parser produces from the two
NASA/GSFC sample CCSDS-AEM files shipped with GMAT R2026a (under
``data/vehicle/ephem/ccsds/``). A format drift in a future GMAT release or a
parser regression trips these loudly.

Tests run without any GMAT install — parsing is pure pandas; the fixtures are
checked into the repo verbatim.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmat_run.parsers.aem_ephemeris import parse

FIXTURES = Path(__file__).parent / "fixtures" / "aem_ephemeris"


class TestBasicQuatFile:
    """``CCSDS_BasicQuatFile.aem`` — NASA/GSFC quaternion sample."""

    FIXTURE = FIXTURES / "CCSDS_BasicQuatFile.aem"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == ["Epoch", "Q1", "Q2", "Q3", "Q4"]

    def test_row_count(self, df: pd.DataFrame) -> None:
        # Two segments concatenated; the segment boundary at 12:59:27.061
        # is recorded in both, so the total includes that overlap row twice.
        assert len(df) == 124

    def test_epoch_dtype_and_scale(self, df: pd.DataFrame) -> None:
        assert df["Epoch"].dtype == np.dtype("datetime64[ns]")
        assert df.attrs["epoch_scales"] == {"Epoch": "UTC"}

    def test_quaternion_components_are_float64(self, df: pd.DataFrame) -> None:
        for column in ["Q1", "Q2", "Q3", "Q4"]:
            assert df[column].dtype == np.float64, column

    def test_first_record(self, df: pd.DataFrame) -> None:
        assert df["Epoch"].iloc[0] == pd.Timestamp("2000-01-01 11:59:28")
        assert df["Q1"].iloc[0] == pytest.approx(0.1952861484651402)
        assert df["Q2"].iloc[0] == pytest.approx(-0.07946016321668915)
        assert df["Q3"].iloc[0] == pytest.approx(0.3188764252036319)
        assert df["Q4"].iloc[0] == pytest.approx(0.9240493645517975)

    def test_last_record(self, df: pd.DataFrame) -> None:
        assert df["Epoch"].iloc[-1] == pd.Timestamp("2000-01-01 15:19:28")
        assert df["Q1"].iloc[-1] == pytest.approx(0.1939937265864762)
        assert df["Q4"].iloc[-1] == pytest.approx(0.9188383156873665)

    def test_metadata_attrs(self, df: pd.DataFrame) -> None:
        assert df.attrs["attitude_type"] == "QUATERNION"
        assert df.attrs["quaternion_type"] == "LAST"
        assert df.attrs["object_name"] == "DefaultSC"
        assert df.attrs["object_id"] == "NoId"
        assert df.attrs["center_name"] == "Earth"
        assert df.attrs["ref_frame_a"] == "EME2000"
        assert df.attrs["ref_frame_b"] == "SC_BODY_1"
        assert df.attrs["attitude_dir"] == "A2B"
        assert df.attrs["time_scale"] == "UTC"
        assert df.attrs["interpolation"] == "Linear"
        assert df.attrs["interpolation_degree"] == 7
        assert "euler_rot_seq" not in df.attrs

    def test_file_header_attrs(self, df: pd.DataFrame) -> None:
        fh = df.attrs["file_header"]
        assert fh["CCSDS_AEM_VERS"] == "1.0"
        assert fh["ORIGINATOR"] == "NASA/GSFC"
        assert "CREATION_DATE" in fh

    def test_segments_attr_lists_both_meta_blocks(self, df: pd.DataFrame) -> None:
        segments = df.attrs["segments"]
        assert len(segments) == 2
        assert all(seg["ATTITUDE_TYPE"] == "QUATERNION" for seg in segments)
        assert segments[0]["START_TIME"] == "2000-01-01T11:59:28.000"
        assert segments[1]["START_TIME"] == "2000-01-01T12:59:27.061"


class TestBasicEulerFile:
    """``CCSDS_BasicEulerFile.aem`` — NASA/GSFC Euler-angle sample."""

    FIXTURE = FIXTURES / "CCSDS_BasicEulerFile.aem"

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == ["Epoch", "EulerAngle1", "EulerAngle2", "EulerAngle3"]

    def test_row_count(self, df: pd.DataFrame) -> None:
        # Two segments concatenated; segment-boundary row appears twice.
        assert len(df) == 123

    def test_epoch_dtype_and_scale(self, df: pd.DataFrame) -> None:
        assert df["Epoch"].dtype == np.dtype("datetime64[ns]")
        assert df.attrs["epoch_scales"] == {"Epoch": "UTC"}

    def test_components_are_float64(self, df: pd.DataFrame) -> None:
        for column in ["EulerAngle1", "EulerAngle2", "EulerAngle3"]:
            assert df[column].dtype == np.float64, column

    def test_first_record(self, df: pd.DataFrame) -> None:
        assert df["Epoch"].iloc[0] == pd.Timestamp("2000-01-01 11:59:28")
        assert df["EulerAngle1"].iloc[0] == pytest.approx(35.4540963916258)
        assert df["EulerAngle2"].iloc[0] == pytest.approx(-15.74726604135554)
        assert df["EulerAngle3"].iloc[0] == pytest.approx(18.80387738364443)

    def test_last_record(self, df: pd.DataFrame) -> None:
        assert df["Epoch"].iloc[-1] == pd.Timestamp("2000-01-01 15:19:28")
        assert df["EulerAngle1"].iloc[-1] == pytest.approx(37.25929459098576)
        assert df["EulerAngle3"].iloc[-1] == pytest.approx(18.30509314510281)

    def test_metadata_attrs(self, df: pd.DataFrame) -> None:
        assert df.attrs["attitude_type"] == "EULER_ANGLE"
        assert df.attrs["euler_rot_seq"] == "321"
        assert df.attrs["object_name"] == "DefaultSC"
        assert df.attrs["center_name"] == "Earth"
        assert df.attrs["ref_frame_a"] == "EME2000"
        assert df.attrs["ref_frame_b"] == "SC_BODY_1"
        assert df.attrs["attitude_dir"] == "A2B"
        assert df.attrs["time_scale"] == "UTC"
        assert df.attrs["interpolation"] == "LAGRANGE"
        assert df.attrs["interpolation_degree"] == 7
        assert "quaternion_type" not in df.attrs

    def test_file_header_attrs(self, df: pd.DataFrame) -> None:
        fh = df.attrs["file_header"]
        assert fh["CCSDS_AEM_VERS"] == "1.0"
        assert fh["ORIGINATOR"] == "NASA/GSFC"

    def test_segments_attr_lists_both_meta_blocks(self, df: pd.DataFrame) -> None:
        segments = df.attrs["segments"]
        assert len(segments) == 2
        assert all(seg["ATTITUDE_TYPE"] == "EULER_ANGLE" for seg in segments)
