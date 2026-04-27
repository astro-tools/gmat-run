"""Unit tests for :func:`gmat_run.parsers.reportfile.parse`.

Fixture content is written inline via ``tmp_path`` rather than checked-in
files, so encoding and line-ending details stay explicit at each call site.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmat_run.errors import GmatOutputParseError
from gmat_run.parsers.reportfile import parse

# --- helpers -----------------------------------------------------------------


def _write(path: Path, content: str, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Write ``content`` verbatim with the requested encoding and line ending."""
    path.write_bytes(content.replace("\n", newline).encode(encoding))
    return path


# A three-column happy-path sample mirroring GMAT's R2026a layout: Gregorian
# epoch, a float field, a scientific-notation field, and an integer mass.
_HEADER = (
    "Sat.UTCGregorian          Sat.Earth.SMA             Sat.Earth.ECC             Sat.TotalMass"
)
_ROW_1 = "26 Nov 2026 12:00:00.000  6578.136299999994         9.999999999964634e-05     1300000"
_ROW_2 = "26 Nov 2026 12:01:00.000  6578.137326507367         0.0001451202385349637     1300000"
_BASIC = f"{_HEADER}\n{_ROW_1}\n{_ROW_2}\n"


# --- happy path --------------------------------------------------------------


def test_parses_header_columns_verbatim(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.report", _BASIC))
    assert list(df.columns) == [
        "Sat.UTCGregorian",
        "Sat.Earth.SMA",
        "Sat.Earth.ECC",
        "Sat.TotalMass",
    ]


def test_row_count_matches_data_lines(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.report", _BASIC))
    assert len(df) == 2


def test_gregorian_epoch_becomes_datetime64(tmp_path: Path) -> None:
    """UTCGregorian is promoted end-to-end to datetime64[ns] and tagged UTC."""
    df = parse(_write(tmp_path / "basic.report", _BASIC))
    assert df["Sat.UTCGregorian"].dtype == np.dtype("datetime64[ns]")
    assert df["Sat.UTCGregorian"].iloc[0] == pd.Timestamp("2026-11-26 12:00:00")
    assert df["Sat.UTCGregorian"].iloc[1] == pd.Timestamp("2026-11-26 12:01:00")
    assert df.attrs["epoch_scales"]["Sat.UTCGregorian"] == "UTC"


def test_modjulian_column_parses_end_to_end(tmp_path: Path) -> None:
    """TAIModJulian column comes out as datetime64[ns] with the TAI scale tag."""
    content = (
        "Sat.TAIModJulian          Sat.Earth.SMA\n"
        "21545.0                   6578.136\n"
        "21545.5                   6578.137\n"
    )
    df = parse(_write(tmp_path / "mjd.report", content))
    assert df["Sat.TAIModJulian"].dtype == np.dtype("datetime64[ns]")
    assert df["Sat.TAIModJulian"].iloc[0] == pd.Timestamp("2000-01-01 12:00:00")
    assert df.attrs["epoch_scales"]["Sat.TAIModJulian"] == "TAI"


def test_float_column_is_float64(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.report", _BASIC))
    assert df["Sat.Earth.SMA"].dtype == np.float64
    assert df["Sat.Earth.SMA"].iloc[0] == pytest.approx(6578.136299999994)


def test_scientific_notation_parses_as_float(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.report", _BASIC))
    assert df["Sat.Earth.ECC"].dtype == np.float64
    assert df["Sat.Earth.ECC"].iloc[0] == pytest.approx(9.999999999964634e-05)


def test_integer_column_is_int64(tmp_path: Path) -> None:
    """All-integer column should be inferred as int, not float."""
    df = parse(_write(tmp_path / "basic.report", _BASIC))
    assert df["Sat.TotalMass"].dtype == np.int64
    assert df["Sat.TotalMass"].iloc[0] == 1300000


# --- encoding / line endings -------------------------------------------------


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / "bom.report"
    path.write_bytes(b"\xef\xbb\xbf" + _BASIC.encode("utf-8"))
    df = parse(path)
    # First column must not carry a U+FEFF prefix from the BOM.
    assert next(iter(df.columns)) == "Sat.UTCGregorian"


@pytest.mark.parametrize(("newline", "label"), [("\n", "lf"), ("\r\n", "crlf")], ids=["lf", "crlf"])
def test_line_endings_produce_identical_frame(tmp_path: Path, newline: str, label: str) -> None:
    path = _write(tmp_path / f"{label}.report", _BASIC, newline=newline)
    df = parse(path)
    reference = parse(_write(tmp_path / "reference.report", _BASIC, newline="\n"))
    pd.testing.assert_frame_equal(df, reference)


# --- repeated header blocks --------------------------------------------------


def test_repeated_header_is_skipped(tmp_path: Path) -> None:
    """GMAT re-emits the header at mission-sequence segment boundaries."""
    content = f"{_HEADER}\n{_ROW_1}\n{_HEADER}\n{_ROW_2}\n"
    df = parse(_write(tmp_path / "segments.report", content))
    assert len(df) == 2


def test_duplicate_data_row_is_preserved(tmp_path: Path) -> None:
    """Segment-initial state repeats look like duplicated data — keep them."""
    content = f"{_HEADER}\n{_ROW_1}\n{_HEADER}\n{_ROW_1}\n{_ROW_2}\n"
    df = parse(_write(tmp_path / "dupdata.report", content))
    assert len(df) == 3
    assert df["Sat.UTCGregorian"].iloc[0] == df["Sat.UTCGregorian"].iloc[1]


# --- edge cases --------------------------------------------------------------


def test_single_column_report(tmp_path: Path) -> None:
    content = "Sat.TotalMass\n1300000\n1300001\n"
    df = parse(_write(tmp_path / "onecol.report", content))
    assert list(df.columns) == ["Sat.TotalMass"]
    assert df["Sat.TotalMass"].tolist() == [1300000, 1300001]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    content = f"\n{_HEADER}\n\n{_ROW_1}\n\n{_ROW_2}\n\n"
    df = parse(_write(tmp_path / "blanks.report", content))
    assert len(df) == 2


def test_trailing_whitespace_is_tolerated(tmp_path: Path) -> None:
    """GMAT pads data rows with trailing spaces; the parser must not break."""
    padded = f"{_HEADER}   \n{_ROW_1}     \n{_ROW_2}\t\n"
    df = parse(_write(tmp_path / "padded.report", padded))
    assert len(df) == 2
    assert df["Sat.TotalMass"].iloc[0] == 1300000


# --- round-trip --------------------------------------------------------------


def test_round_trip_numeric_values_match_file(tmp_path: Path) -> None:
    """Parsed numeric values must round-trip to the exact source tokens."""
    df = parse(_write(tmp_path / "rt.report", _BASIC))
    assert repr(float(df["Sat.Earth.SMA"].iloc[0])) == "6578.136299999994"
    assert repr(float(df["Sat.Earth.ECC"].iloc[0])) == "9.999999999964634e-05"
    assert int(df["Sat.TotalMass"].iloc[0]) == 1300000


# --- malformed input ---------------------------------------------------------


def test_column_count_mismatch_raises_with_line_number(tmp_path: Path) -> None:
    bad_row = "26 Nov 2026 12:02:00.000  6578.13"  # two tokens instead of four
    content = f"{_HEADER}\n{_ROW_1}\n{bad_row}\n"
    path = _write(tmp_path / "badcols.report", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "line 3" in str(excinfo.value)
    assert "expected 4 columns" in str(excinfo.value)
    assert excinfo.value.path == path


def test_empty_file_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.report", "")
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "empty" in str(excinfo.value).lower()
    assert excinfo.value.path == path


def test_only_blank_lines_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "blanks_only.report", "\n\n  \n\t\n")
    with pytest.raises(GmatOutputParseError):
        parse(path)


def test_accepts_str_path(tmp_path: Path) -> None:
    """``path`` must accept both ``str`` and ``os.PathLike`` — string form."""
    path = _write(tmp_path / "strpath.report", _BASIC)
    df = parse(str(path))
    assert len(df) == 2


# --- convert_to --------------------------------------------------------------


def test_convert_to_unifies_mixed_scale_columns(tmp_path: Path) -> None:
    """Acceptance criterion from issue #66: a mixed-scale ReportFile reads
    out with every epoch column on the same scale and the same Timestamp
    values for a known epoch row.

    The file emits the same physical instant in TAI Gregorian and UTC
    ModJulian. Past the 2017-01-01 leap second, TAI - UTC = 37 s, so:
        TAI Gregorian: 15 Jan 2026 12:00:37.000
        UTC ModJulian: 27770.50000 (2026-01-15 12:00:00 UTC; MJD 0 = 1941-01-05 12:00 UT).
    Note: GMAT's ModJulian epoch is 1941-01-05 12:00:00, so 2026-01-15 12:00
    is exactly (2026-01-15 12:00) - (1941-01-05 12:00) = 31055 days.
    """
    # 2026-01-15 12:00 - 1941-01-05 12:00 = 31055 days (verified below).
    days = (pd.Timestamp("2026-01-15 12:00:00") - pd.Timestamp("1941-01-05 12:00:00")).days
    content = f"Sat.TAIGregorian          Sat.UTCModJulian\n15 Jan 2026 12:00:37.000  {days}.0\n"
    df = parse(_write(tmp_path / "mixed.report", content), convert_to="UTC")
    assert df.attrs["epoch_scales"] == {
        "Sat.TAIGregorian": "UTC",
        "Sat.UTCModJulian": "UTC",
    }
    assert df["Sat.TAIGregorian"].iloc[0] == df["Sat.UTCModJulian"].iloc[0]
    assert df["Sat.TAIGregorian"].iloc[0] == pd.Timestamp("2026-01-15 12:00:00")


def test_convert_to_default_none_is_legacy_behaviour(tmp_path: Path) -> None:
    """Regression: the default keeps each column in its native scale."""
    df = parse(_write(tmp_path / "basic.report", _BASIC))
    assert df.attrs["epoch_scales"] == {"Sat.UTCGregorian": "UTC"}
    assert df["Sat.UTCGregorian"].iloc[0] == pd.Timestamp("2026-11-26 12:00:00")
