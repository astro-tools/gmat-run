"""Unit tests for :func:`gmat_run.parsers.aem_ephemeris.parse`.

Inline ``tmp_path`` fixtures keep encoding and structure visible at each call
site, mirroring the convention in :mod:`tests.test_parsers_ephemeris`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmat_run.errors import GmatOutputParseError
from gmat_run.parsers.aem_ephemeris import is_aem_ephemeris, parse

# --- helpers -----------------------------------------------------------------


def _write(path: Path, content: str, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Write ``content`` verbatim with the requested encoding and line ending."""
    path.write_bytes(content.replace("\n", newline).encode(encoding))
    return path


_HEADER = """\
CCSDS_AEM_VERS = 1.0
CREATION_DATE  = 2026-04-25T18:54:25
ORIGINATOR     = GMAT USER

"""

_QUAT_META_BASE = """\
META_START
OBJECT_NAME          = Sat
OBJECT_ID            = SatId
CENTER_NAME          = Earth
REF_FRAME_A          = EME2000
REF_FRAME_B          = SC_BODY_1
ATTITUDE_DIR         = A2B
TIME_SYSTEM          = UTC
START_TIME           = 2026-01-01T12:00:00.000
USEABLE_START_TIME   = 2026-01-01T12:00:00.000
USEABLE_STOP_TIME    = 2026-01-01T12:02:00.000
STOP_TIME            = 2026-01-01T12:02:00.000
ATTITUDE_TYPE        = QUATERNION
QUATERNION_TYPE      = LAST
INTERPOLATION_METHOD = Linear
INTERPOLATION_DEGREE = 7
META_STOP

"""

_QUAT_DATA_BASE = """\
DATA_START
2026-01-01T12:00:00.000 0.1  0.2  0.3  0.927361
2026-01-01T12:01:00.000 0.11 0.21 0.31 0.920472
2026-01-01T12:02:00.000 0.12 0.22 0.32 0.913348
DATA_STOP

"""

_QUAT_BASIC = _HEADER + _QUAT_META_BASE + _QUAT_DATA_BASE

_EULER_META_BASE = """\
META_START
OBJECT_NAME          = Sat
CENTER_NAME          = Earth
REF_FRAME_A          = EME2000
REF_FRAME_B          = SC_BODY_1
ATTITUDE_DIR         = A2B
TIME_SYSTEM          = UTC
START_TIME           = 2026-01-01T12:00:00.000
STOP_TIME            = 2026-01-01T12:02:00.000
ATTITUDE_TYPE        = EULER_ANGLE
EULER_ROT_SEQ        = 321
INTERPOLATION_METHOD = LAGRANGE
INTERPOLATION_DEGREE = 7
META_STOP

"""

_EULER_DATA_BASE = """\
DATA_START
2026-01-01T12:00:00.000 35.0  -15.0  18.0
2026-01-01T12:01:00.000 35.1  -15.1  18.1
2026-01-01T12:02:00.000 35.2  -15.2  18.2
DATA_STOP

"""

_EULER_BASIC = _HEADER + _EULER_META_BASE + _EULER_DATA_BASE


# --- happy path: QUATERNION --------------------------------------------------


def test_quaternion_columns(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "q.aem", _QUAT_BASIC))
    assert list(df.columns) == ["Epoch", "Q1", "Q2", "Q3", "Q4"]


def test_quaternion_row_count(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "q.aem", _QUAT_BASIC))
    assert len(df) == 3


def test_quaternion_epoch_dtype_and_scale(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "q.aem", _QUAT_BASIC))
    assert df["Epoch"].dtype == np.dtype("datetime64[ns]")
    assert df["Epoch"].iloc[0] == pd.Timestamp("2026-01-01 12:00:00")
    assert df["Epoch"].iloc[-1] == pd.Timestamp("2026-01-01 12:02:00")
    assert df.attrs["epoch_scales"] == {"Epoch": "UTC"}


def test_quaternion_components_are_float64(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "q.aem", _QUAT_BASIC))
    for column in ["Q1", "Q2", "Q3", "Q4"]:
        assert df[column].dtype == np.float64, column


def test_quaternion_values_round_trip(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "q.aem", _QUAT_BASIC))
    assert df["Q1"].iloc[0] == pytest.approx(0.1)
    assert df["Q4"].iloc[0] == pytest.approx(0.927361)
    assert df["Q4"].iloc[-1] == pytest.approx(0.913348)


def test_quaternion_metadata_attrs(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "q.aem", _QUAT_BASIC))
    assert df.attrs["attitude_type"] == "QUATERNION"
    assert df.attrs["quaternion_type"] == "LAST"
    assert df.attrs["object_name"] == "Sat"
    assert df.attrs["object_id"] == "SatId"
    assert df.attrs["center_name"] == "Earth"
    assert df.attrs["ref_frame_a"] == "EME2000"
    assert df.attrs["ref_frame_b"] == "SC_BODY_1"
    assert df.attrs["attitude_dir"] == "A2B"
    assert df.attrs["time_scale"] == "UTC"
    assert df.attrs["interpolation"] == "Linear"
    assert df.attrs["interpolation_degree"] == 7
    assert isinstance(df.attrs["interpolation_degree"], int)
    # euler_rot_seq is type-specific; absent from a quaternion file.
    assert "euler_rot_seq" not in df.attrs


def test_quaternion_first_variant(tmp_path: Path) -> None:
    """``QUATERNION_TYPE = FIRST`` is preserved verbatim; no column reorder."""
    text = _QUAT_BASIC.replace("QUATERNION_TYPE      = LAST", "QUATERNION_TYPE      = FIRST")
    df = parse(_write(tmp_path / "qfirst.aem", text))
    assert df.attrs["quaternion_type"] == "FIRST"
    # Columns in the same source order — interpretation is on the caller.
    assert list(df.columns) == ["Epoch", "Q1", "Q2", "Q3", "Q4"]


def test_quaternion_type_optional(tmp_path: Path) -> None:
    """``QUATERNION_TYPE`` is informational; absence does not block parsing."""
    text = _QUAT_BASIC.replace("QUATERNION_TYPE      = LAST\n", "")
    df = parse(_write(tmp_path / "qnoqt.aem", text))
    assert "quaternion_type" not in df.attrs
    assert len(df) == 3


# --- happy path: EULER_ANGLE -------------------------------------------------


def test_euler_columns(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "e.aem", _EULER_BASIC))
    assert list(df.columns) == ["Epoch", "EulerAngle1", "EulerAngle2", "EulerAngle3"]


def test_euler_row_count(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "e.aem", _EULER_BASIC))
    assert len(df) == 3


def test_euler_components_are_float64(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "e.aem", _EULER_BASIC))
    for column in ["EulerAngle1", "EulerAngle2", "EulerAngle3"]:
        assert df[column].dtype == np.float64, column


def test_euler_metadata_attrs(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "e.aem", _EULER_BASIC))
    assert df.attrs["attitude_type"] == "EULER_ANGLE"
    assert df.attrs["euler_rot_seq"] == "321"
    assert df.attrs["interpolation"] == "LAGRANGE"
    assert "quaternion_type" not in df.attrs


def test_euler_alternate_rot_seq(tmp_path: Path) -> None:
    text = _EULER_BASIC.replace("EULER_ROT_SEQ        = 321", "EULER_ROT_SEQ        = 312")
    df = parse(_write(tmp_path / "e312.aem", text))
    assert df.attrs["euler_rot_seq"] == "312"


# --- file header -------------------------------------------------------------


def test_file_header_attrs(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "q.aem", _QUAT_BASIC))
    fh = df.attrs["file_header"]
    assert fh["CCSDS_AEM_VERS"] == "1.0"
    assert fh["ORIGINATOR"] == "GMAT USER"  # multi-word value preserved
    assert "CREATION_DATE" in fh


def test_no_segments_attr_for_single_segment_file(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "q.aem", _QUAT_BASIC))
    assert "segments" not in df.attrs


# --- multi-segment files -----------------------------------------------------


_QUAT_SEG_2_META = _QUAT_META_BASE.replace(
    "INTERPOLATION_DEGREE = 7",
    "INTERPOLATION_DEGREE = 4",
).replace(
    "START_TIME           = 2026-01-01T12:00:00.000",
    "START_TIME           = 2026-01-01T12:03:00.000",
)
_QUAT_SEG_2_DATA = """\
DATA_START
2026-01-01T12:03:00.000 0.13 0.23 0.33 0.905990
2026-01-01T12:04:00.000 0.14 0.24 0.34 0.898400
DATA_STOP

"""
_QUAT_MULTI = _HEADER + _QUAT_META_BASE + _QUAT_DATA_BASE + _QUAT_SEG_2_META + _QUAT_SEG_2_DATA


def test_multi_segment_concatenates(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "multi.aem", _QUAT_MULTI))
    assert len(df) == 5
    assert df["Epoch"].iloc[0] == pd.Timestamp("2026-01-01 12:00:00")
    assert df["Epoch"].iloc[-1] == pd.Timestamp("2026-01-01 12:04:00")


def test_multi_segment_records_per_segment_metadata(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "multi.aem", _QUAT_MULTI))
    segments = df.attrs["segments"]
    assert len(segments) == 2
    assert segments[0]["INTERPOLATION_DEGREE"] == "7"
    assert segments[1]["INTERPOLATION_DEGREE"] == "4"


def test_multi_segment_flat_attrs_only_when_consensus(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "multi.aem", _QUAT_MULTI))
    assert df.attrs["attitude_type"] == "QUATERNION"
    assert df.attrs["object_name"] == "Sat"
    assert "interpolation_degree" not in df.attrs


def test_mixed_attitude_type_segments_rejected(tmp_path: Path) -> None:
    """Mixing QUATERNION and EULER_ANGLE in one file is meaningless."""
    text = _HEADER + _QUAT_META_BASE + _QUAT_DATA_BASE + _EULER_META_BASE + _EULER_DATA_BASE
    path = _write(tmp_path / "mixed.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "ATTITUDE_TYPE" in str(excinfo.value)


# --- comments and structural noise -------------------------------------------


def test_comment_lines_skipped(tmp_path: Path) -> None:
    text = (
        _HEADER
        + "COMMENT free-form note above the segment\n"
        + _QUAT_META_BASE.replace(
            "OBJECT_NAME          = Sat\n",
            "OBJECT_NAME          = Sat\nCOMMENT mid-meta\n",
        )
        + _QUAT_DATA_BASE.replace("DATA_START\n", "DATA_START\nCOMMENT mid-data\n")
    )
    df = parse(_write(tmp_path / "commented.aem", text))
    assert len(df) == 3


def test_blank_lines_anywhere_tolerated(tmp_path: Path) -> None:
    text = "\n\n" + _QUAT_BASIC + "\n\n"
    df = parse(_write(tmp_path / "blanks.aem", text))
    assert len(df) == 3


# --- encoding / line endings -------------------------------------------------


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / "bom.aem"
    path.write_bytes(b"\xef\xbb\xbf" + _QUAT_BASIC.encode("utf-8"))
    df = parse(path)
    assert df.attrs["file_header"]["CCSDS_AEM_VERS"] == "1.0"


@pytest.mark.parametrize(("newline", "label"), [("\n", "lf"), ("\r\n", "crlf")], ids=["lf", "crlf"])
def test_line_endings_produce_identical_frame(tmp_path: Path, newline: str, label: str) -> None:
    path = _write(tmp_path / f"{label}.aem", _QUAT_BASIC, newline=newline)
    df = parse(path)
    reference = parse(_write(tmp_path / "reference.aem", _QUAT_BASIC, newline="\n"))
    pd.testing.assert_frame_equal(df, reference)


def test_trailing_whitespace_tolerated(tmp_path: Path) -> None:
    padded = _QUAT_BASIC.replace("INTERPOLATION_DEGREE = 7", "INTERPOLATION_DEGREE = 7   ")
    df = parse(_write(tmp_path / "padded.aem", padded))
    assert df.attrs["interpolation_degree"] == 7


# --- type acceptance --------------------------------------------------------


def test_accepts_str_path(tmp_path: Path) -> None:
    path = _write(tmp_path / "strpath.aem", _QUAT_BASIC)
    df = parse(str(path))
    assert len(df) == 3


# --- malformed input --------------------------------------------------------


def test_empty_file_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.aem", "")
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "empty" in str(excinfo.value).lower()
    assert excinfo.value.path == path


def test_only_blank_lines_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "blanks_only.aem", "\n\n  \n\t\n")
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "empty" in str(excinfo.value).lower()


def test_no_meta_block_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "noseg.aem", _HEADER)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "META_START" in str(excinfo.value)


def test_meta_stop_without_start_raises(tmp_path: Path) -> None:
    text = _HEADER + "META_STOP\n"
    path = _write(tmp_path / "lonely_stop.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "META_STOP" in str(excinfo.value)


def test_unterminated_meta_raises(tmp_path: Path) -> None:
    truncated = _HEADER + _QUAT_META_BASE.replace("META_STOP\n\n", "")
    path = _write(tmp_path / "unterm_meta.aem", truncated)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "META_STOP" in str(excinfo.value)


def test_data_stop_without_start_raises(tmp_path: Path) -> None:
    text = _HEADER + _QUAT_META_BASE + "DATA_STOP\n"
    path = _write(tmp_path / "lonely_data_stop.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "DATA_STOP" in str(excinfo.value)


def test_unterminated_data_raises(tmp_path: Path) -> None:
    truncated = _HEADER + _QUAT_META_BASE + _QUAT_DATA_BASE.replace("DATA_STOP\n\n", "")
    path = _write(tmp_path / "unterm_data.aem", truncated)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "DATA_STOP" in str(excinfo.value)


def test_meta_inside_open_meta_block_raises(tmp_path: Path) -> None:
    """A second META_START before META_STOP is structurally invalid."""
    bad = _QUAT_META_BASE.replace(
        "OBJECT_NAME          = Sat\n",
        "OBJECT_NAME          = Sat\nMETA_START\n",
    )
    text = _HEADER + bad + _QUAT_DATA_BASE
    path = _write(tmp_path / "nested_meta.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "META_START" in str(excinfo.value)


def test_data_start_outside_meta_raises(tmp_path: Path) -> None:
    text = _HEADER + "DATA_START\n2026-01-01T12:00:00.000 0.1 0.2 0.3 0.927\nDATA_STOP\n"
    path = _write(tmp_path / "stray_data.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "DATA_START" in str(excinfo.value)


def test_content_between_segments_raises(tmp_path: Path) -> None:
    """Stray text between META_STOP and DATA_START is rejected."""
    text = _HEADER + _QUAT_META_BASE + "stray text\n" + _QUAT_DATA_BASE
    path = _write(tmp_path / "stray.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "unexpected content" in str(excinfo.value)


def test_malformed_meta_line_raises(tmp_path: Path) -> None:
    text = (
        _HEADER
        + _QUAT_META_BASE.replace(
            "OBJECT_NAME          = Sat\n",
            "OBJECT_NAME          = Sat\nthis line has no equals\n",
        )
        + _QUAT_DATA_BASE
    )
    path = _write(tmp_path / "badmeta.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "KEY = VALUE" in str(excinfo.value) or "expected" in str(excinfo.value).lower()


def test_missing_attitude_type_raises(tmp_path: Path) -> None:
    text = (
        _HEADER
        + _QUAT_META_BASE.replace("ATTITUDE_TYPE        = QUATERNION\n", "")
        + _QUAT_DATA_BASE
    )
    path = _write(tmp_path / "no_attitude_type.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "ATTITUDE_TYPE" in str(excinfo.value)


@pytest.mark.parametrize(
    "rejected",
    ["QUATERNION/DERIVATIVE", "QUATERNION/RATE", "EULER_ANGLE/RATE", "SPIN", "SPIN/NUTATION"],
)
def test_unsupported_attitude_type_raises(tmp_path: Path, rejected: str) -> None:
    text = (
        _HEADER
        + _QUAT_META_BASE.replace(
            "ATTITUDE_TYPE        = QUATERNION", f"ATTITUDE_TYPE        = {rejected}"
        )
        + _QUAT_DATA_BASE
    )
    path = _write(tmp_path / "rejected.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert rejected in str(excinfo.value)
    assert "not supported" in str(excinfo.value)


def test_unrecognised_attitude_type_raises(tmp_path: Path) -> None:
    text = (
        _HEADER
        + _QUAT_META_BASE.replace(
            "ATTITUDE_TYPE        = QUATERNION", "ATTITUDE_TYPE        = NONSENSE"
        )
        + _QUAT_DATA_BASE
    )
    path = _write(tmp_path / "nonsense.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "NONSENSE" in str(excinfo.value)
    assert "unrecognised" in str(excinfo.value)


def test_wrong_record_column_count_raises(tmp_path: Path) -> None:
    bad = "DATA_START\n2026-01-01T12:00:00.000 0.1 0.2\nDATA_STOP\n"
    text = _HEADER + _QUAT_META_BASE + bad
    path = _write(tmp_path / "badcols.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "expected 5" in str(excinfo.value)
    assert excinfo.value.path == path


def test_malformed_epoch_raises(tmp_path: Path) -> None:
    bad = "DATA_START\nnot-an-epoch 0.1 0.2 0.3 0.9\nDATA_STOP\n"
    text = _HEADER + _QUAT_META_BASE + bad
    path = _write(tmp_path / "badepoch.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "epoch" in str(excinfo.value).lower()


def test_non_numeric_value_raises(tmp_path: Path) -> None:
    bad = "DATA_START\n2026-01-01T12:00:00.000 notanumber 0.2 0.3 0.9\nDATA_STOP\n"
    text = _HEADER + _QUAT_META_BASE + bad
    path = _write(tmp_path / "badvalue.aem", text)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "Q1" in str(excinfo.value) or "non-numeric" in str(excinfo.value).lower()


# --- empty data block --------------------------------------------------------


def test_empty_data_block_yields_empty_typed_frame(tmp_path: Path) -> None:
    """A spec-allowed (rare) empty data block: typed frame, no rows."""
    text = _HEADER + _QUAT_META_BASE + "DATA_START\nDATA_STOP\n"
    df = parse(_write(tmp_path / "empty_data.aem", text))
    assert len(df) == 0
    assert list(df.columns) == ["Epoch", "Q1", "Q2", "Q3", "Q4"]
    assert df["Epoch"].dtype == np.dtype("datetime64[ns]")
    for column in ["Q1", "Q2", "Q3", "Q4"]:
        assert df[column].dtype == np.float64


# --- is_aem_ephemeris sniffer ------------------------------------------------


def test_is_aem_ephemeris_recognises_aem(tmp_path: Path) -> None:
    assert is_aem_ephemeris(_write(tmp_path / "q.aem", _QUAT_BASIC)) is True


def test_is_aem_ephemeris_rejects_oem(tmp_path: Path) -> None:
    oem = "CCSDS_OEM_VERS = 1.0\nCREATION_DATE = 2026-04-25\n"
    assert is_aem_ephemeris(_write(tmp_path / "f.oem", oem)) is False


def test_is_aem_ephemeris_skips_blank_and_comment_prefix(tmp_path: Path) -> None:
    text = "\n\nCOMMENT prelude\n" + _QUAT_BASIC
    assert is_aem_ephemeris(_write(tmp_path / "lead.aem", text)) is True


def test_is_aem_ephemeris_handles_missing_file(tmp_path: Path) -> None:
    assert is_aem_ephemeris(tmp_path / "does_not_exist.aem") is False


def test_is_aem_ephemeris_handles_empty_file(tmp_path: Path) -> None:
    assert is_aem_ephemeris(_write(tmp_path / "empty.aem", "")) is False


def test_is_aem_ephemeris_strips_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.aem"
    path.write_bytes(b"\xef\xbb\xbf" + _QUAT_BASIC.encode("utf-8"))
    assert is_aem_ephemeris(path) is True
