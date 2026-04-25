"""Unit tests for :func:`gmat_run.parsers.ephemeris.parse`.

Inline ``tmp_path`` fixtures keep encoding and structure visible at each call
site, mirroring the convention in :mod:`tests.test_parsers_reportfile`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmat_run.errors import GmatOutputParseError
from gmat_run.parsers.ephemeris import parse

# --- helpers -----------------------------------------------------------------


def _write(path: Path, content: str, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Write ``content`` verbatim with the requested encoding and line ending."""
    path.write_bytes(content.replace("\n", newline).encode(encoding))
    return path


# A single-segment OEM file mirroring R2026a's CCSDS-OEM emitter byte-for-byte.
_HEADER = """\
CCSDS_OEM_VERS = 1.0
CREATION_DATE  = 2026-04-25T18:54:25
ORIGINATOR     = GMAT USER

"""

_META_BASE = """\
META_START
OBJECT_NAME          = Sat
OBJECT_ID            = SatId
CENTER_NAME          = Earth
REF_FRAME            = EME2000
TIME_SYSTEM          = UTC
START_TIME           = 2026-01-01T12:00:00.000
USEABLE_START_TIME   = 2026-01-01T12:00:00.000
USEABLE_STOP_TIME    = 2026-01-01T12:02:00.000
STOP_TIME            = 2026-01-01T12:02:00.000
INTERPOLATION        = LAGRANGE
INTERPOLATION_DEGREE = 4
META_STOP

"""

# Each row: epoch + 6 state components in scientific notation. Values rounded
# to keep the literal under the line-length cap; the golden fixture file
# carries the full-precision values for the per-byte regression suite.
_DATA_BASE = (
    "2026-01-01T12:00:00.000  -5.936e+03  1.591e+03  3.337e+03"
    "  -1.955e+00  -7.296e+00  2.207e-16\n"
    "2026-01-01T12:01:00.000  -6.041e+03  1.150e+03  3.330e+03"
    "  -1.536e+00  -7.392e+00  -2.338e-01\n"
    "2026-01-01T12:02:00.000  -6.120e+03  7.041e+02  3.309e+03"
    "  -1.111e+00  -7.457e+00  -4.666e-01\n"
)

_BASIC = _HEADER + _META_BASE + _DATA_BASE


# --- happy path --------------------------------------------------------------


def test_parses_expected_columns(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.oem", _BASIC))
    assert list(df.columns) == ["Epoch", "X", "Y", "Z", "VX", "VY", "VZ"]


def test_row_count_matches_data_records(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.oem", _BASIC))
    assert len(df) == 3


def test_epoch_dtype_and_scale(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.oem", _BASIC))
    assert df["Epoch"].dtype == np.dtype("datetime64[ns]")
    assert df["Epoch"].iloc[0] == pd.Timestamp("2026-01-01 12:00:00")
    assert df["Epoch"].iloc[-1] == pd.Timestamp("2026-01-01 12:02:00")
    assert df.attrs["epoch_scales"] == {"Epoch": "UTC"}


def test_state_columns_are_float64(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.oem", _BASIC))
    for column in ["X", "Y", "Z", "VX", "VY", "VZ"]:
        assert df[column].dtype == np.float64, column


def test_state_values_round_trip(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.oem", _BASIC))
    assert df["X"].iloc[0] == pytest.approx(-5.936e3)
    assert df["VZ"].iloc[0] == pytest.approx(2.207e-16)
    assert df["VZ"].iloc[-1] == pytest.approx(-4.666e-1)


def test_metadata_surfaces_on_attrs(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.oem", _BASIC))
    assert df.attrs["object_name"] == "Sat"
    assert df.attrs["central_body"] == "Earth"
    assert df.attrs["coordinate_system"] == "EME2000"
    assert df.attrs["time_scale"] == "UTC"
    assert df.attrs["interpolation"] == "LAGRANGE"
    # INTERPOLATION_DEGREE is coerced to int for downstream arithmetic.
    assert df.attrs["interpolation_degree"] == 4
    assert isinstance(df.attrs["interpolation_degree"], int)


def test_file_header_surfaces_on_attrs(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.oem", _BASIC))
    fh = df.attrs["file_header"]
    assert fh["CCSDS_OEM_VERS"] == "1.0"
    assert fh["ORIGINATOR"] == "GMAT USER"  # multi-word value preserved
    assert "CREATION_DATE" in fh


def test_no_segments_attr_for_single_segment_file(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.oem", _BASIC))
    assert "segments" not in df.attrs


# --- multi-segment files -----------------------------------------------------


_SEG_2_META = _META_BASE.replace(
    "INTERPOLATION_DEGREE = 4",
    "INTERPOLATION_DEGREE = 7",
).replace(
    "START_TIME           = 2026-01-01T12:00:00.000",
    "START_TIME           = 2026-01-01T12:03:00.000",
)
_SEG_2_DATA = """\
2026-01-01T12:03:00.000  -6.174e+03   2.555e+02   3.273e+03  -6.808e-01  -7.490e+00  -6.974e-01
2026-01-01T12:04:00.000  -6.202e+03  -1.940e+02   3.225e+03  -2.479e-01  -7.492e+00  -9.253e-01
"""
_MULTI_SEGMENT = _HEADER + _META_BASE + _DATA_BASE + _SEG_2_META + _SEG_2_DATA


def test_multi_segment_concatenates(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "multi.oem", _MULTI_SEGMENT))
    assert len(df) == 5  # 3 from segment 1 + 2 from segment 2
    assert df["Epoch"].iloc[0] == pd.Timestamp("2026-01-01 12:00:00")
    assert df["Epoch"].iloc[-1] == pd.Timestamp("2026-01-01 12:04:00")


def test_multi_segment_records_per_segment_metadata(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "multi.oem", _MULTI_SEGMENT))
    segments = df.attrs["segments"]
    assert len(segments) == 2
    assert segments[0]["INTERPOLATION_DEGREE"] == "4"
    assert segments[1]["INTERPOLATION_DEGREE"] == "7"


def test_multi_segment_flat_attrs_only_when_consensus(tmp_path: Path) -> None:
    """Flat ``df.attrs`` keys appear only when every segment agrees on them."""
    df = parse(_write(tmp_path / "multi.oem", _MULTI_SEGMENT))
    # Same across both segments — surface flat.
    assert df.attrs["coordinate_system"] == "EME2000"
    assert df.attrs["central_body"] == "Earth"
    assert df.attrs["time_scale"] == "UTC"
    # Differs between segments — must NOT be surfaced flat.
    assert "interpolation_degree" not in df.attrs


# --- comments and structural noise -------------------------------------------


def test_comment_lines_skipped(tmp_path: Path) -> None:
    """``COMMENT …`` is allowed anywhere per spec — must be tolerated."""
    content = (
        _HEADER
        + "COMMENT this is a free-form note\n"
        + _META_BASE.replace("META_STOP\n\n", "META_STOP\n\nCOMMENT mid-data\n")
        + _DATA_BASE
    )
    df = parse(_write(tmp_path / "commented.oem", content))
    assert len(df) == 3


def test_blank_lines_anywhere_tolerated(tmp_path: Path) -> None:
    content = "\n\n" + _BASIC + "\n\n"
    df = parse(_write(tmp_path / "blanks.oem", content))
    assert len(df) == 3


# --- encoding / line endings -------------------------------------------------


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / "bom.oem"
    path.write_bytes(b"\xef\xbb\xbf" + _BASIC.encode("utf-8"))
    df = parse(path)
    # Header read correctly despite the BOM.
    assert df.attrs["file_header"]["CCSDS_OEM_VERS"] == "1.0"


@pytest.mark.parametrize(("newline", "label"), [("\n", "lf"), ("\r\n", "crlf")], ids=["lf", "crlf"])
def test_line_endings_produce_identical_frame(tmp_path: Path, newline: str, label: str) -> None:
    path = _write(tmp_path / f"{label}.oem", _BASIC, newline=newline)
    df = parse(path)
    reference = parse(_write(tmp_path / "reference.oem", _BASIC, newline="\n"))
    pd.testing.assert_frame_equal(df, reference)


def test_trailing_whitespace_tolerated(tmp_path: Path) -> None:
    """GMAT pads some meta values (`INTERPOLATION_DEGREE = 4 `) and rows."""
    padded = _BASIC.replace("INTERPOLATION_DEGREE = 4", "INTERPOLATION_DEGREE = 4   ")
    df = parse(_write(tmp_path / "padded.oem", padded))
    assert df.attrs["interpolation_degree"] == 4


# --- type acceptance --------------------------------------------------------


def test_accepts_str_path(tmp_path: Path) -> None:
    """``path`` must accept both ``str`` and ``os.PathLike``."""
    path = _write(tmp_path / "strpath.oem", _BASIC)
    df = parse(str(path))
    assert len(df) == 3


# --- malformed input --------------------------------------------------------


def test_empty_file_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.oem", "")
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "empty" in str(excinfo.value).lower()
    assert excinfo.value.path == path


def test_only_blank_lines_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "blanks_only.oem", "\n\n  \n\t\n")
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "empty" in str(excinfo.value).lower()


def test_no_meta_block_raises(tmp_path: Path) -> None:
    """A file without META_START is not a CCSDS-OEM ephemeris."""
    path = _write(tmp_path / "noseg.oem", _HEADER)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "META_START" in str(excinfo.value)


def test_meta_stop_without_start_raises(tmp_path: Path) -> None:
    content = _HEADER + "META_STOP\n"
    path = _write(tmp_path / "lonely_stop.oem", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "META_STOP" in str(excinfo.value)


def test_unterminated_meta_raises(tmp_path: Path) -> None:
    truncated = _HEADER + _META_BASE.replace("META_STOP\n\n", "")
    path = _write(tmp_path / "unterm.oem", truncated)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "META_STOP" in str(excinfo.value)


def test_malformed_meta_line_raises(tmp_path: Path) -> None:
    """A line inside a META block that isn't ``KEY = VALUE`` is rejected."""
    content = (
        _HEADER
        + _META_BASE.replace(
            "OBJECT_NAME          = Sat\n",
            "OBJECT_NAME          = Sat\nthis line has no equals\n",
        )
        + _DATA_BASE
    )
    path = _write(tmp_path / "badmeta.oem", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "KEY = VALUE" in str(excinfo.value) or "expected" in str(excinfo.value).lower()


def test_wrong_record_column_count_raises(tmp_path: Path) -> None:
    bad_row = "2026-01-01T12:03:00.000  1.0  2.0  3.0\n"  # 4 cols, need 7
    content = _HEADER + _META_BASE + _DATA_BASE + bad_row
    path = _write(tmp_path / "badcols.oem", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "expected 7" in str(excinfo.value)
    assert excinfo.value.path == path


def test_malformed_epoch_raises(tmp_path: Path) -> None:
    """An unparseable ISO-8601 epoch surfaces as a typed error."""
    bad_data = "not-an-epoch  1.0  2.0  3.0  4.0  5.0  6.0\n"
    content = _HEADER + _META_BASE + bad_data
    path = _write(tmp_path / "badepoch.oem", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "epoch" in str(excinfo.value).lower()


def test_non_numeric_state_raises(tmp_path: Path) -> None:
    bad = "2026-01-01T12:03:00.000  notanumber  2.0  3.0  4.0  5.0  6.0\n"
    content = _HEADER + _META_BASE + bad
    path = _write(tmp_path / "badstate.oem", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "X" in str(excinfo.value) or "non-numeric" in str(excinfo.value).lower()


# --- covariance blocks ------------------------------------------------------


def test_covariance_blocks_silently_consumed(tmp_path: Path) -> None:
    """v0.2 doesn't expose covariance; the parser must skip the block cleanly."""
    cov = (
        "COVARIANCE_START\n"
        "EPOCH = 2026-01-01T12:00:00.000\n"
        "COV_REF_FRAME = EME2000\n"
        "1.0e-3 2.0e-3 3.0e-3 4.0e-3 5.0e-3 6.0e-3\n"
        "COVARIANCE_STOP\n"
    )
    content = _HEADER + _META_BASE + _DATA_BASE + cov
    df = parse(_write(tmp_path / "withcov.oem", content))
    assert len(df) == 3  # data rows preserved, covariance skipped


def test_unterminated_covariance_raises(tmp_path: Path) -> None:
    cov = "COVARIANCE_START\n1.0 2.0 3.0\n"
    content = _HEADER + _META_BASE + _DATA_BASE + cov
    path = _write(tmp_path / "unterm_cov.oem", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "COVARIANCE" in str(excinfo.value)
