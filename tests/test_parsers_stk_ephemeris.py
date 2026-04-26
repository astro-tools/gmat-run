"""Unit tests for :func:`gmat_run.parsers.stk_ephemeris.parse`.

Inline ``tmp_path`` fixtures keep encoding and structure visible at each call
site, mirroring the convention in :mod:`tests.test_parsers_ephemeris`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmat_run.errors import GmatOutputParseError
from gmat_run.parsers.stk_ephemeris import is_stk_ephemeris, parse

# --- helpers -----------------------------------------------------------------


def _write(path: Path, content: str, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Write ``content`` verbatim with the requested encoding and line ending."""
    path.write_bytes(content.replace("\n", newline).encode(encoding))
    return path


# A single-section STK file mirroring R2026a's STK-TimePosVel emitter
# byte-for-byte. Three records at 60 s cadence from a 2026-01-01 12:00:00 UTC
# scenario epoch.
_HEADER = """\
stk.v.11.0
# WrittenBy    GMAT R2026a
"""

_BEGIN = """\
BEGIN Ephemeris
NumberOfEphemerisPoints 3
ScenarioEpoch           01 Jan 2026 12:00:00.000
InterpolationMethod     Lagrange
InterpolationSamplesM1  4
CentralBody             Earth
CoordinateSystem        J2000
DistanceUnit            Kilometers

EphemerisTimePosVel

"""

# Each row: offset_seconds X Y Z VX VY VZ. Values rounded to keep the literal
# under the line-length cap; the golden fixture file carries the
# full-precision values for the per-byte regression suite.
_DATA_BASE = (
    "0.000000000000000e+00  -5.936e+03  1.591e+03  3.337e+03"
    "  -1.955e+00  -7.296e+00  2.207e-16\n"
    "6.000000000000000e+01  -6.041e+03  1.150e+03  3.330e+03"
    "  -1.536e+00  -7.392e+00  -2.338e-01\n"
    "1.200000000000000e+02  -6.120e+03  7.041e+02  3.309e+03"
    "  -1.111e+00  -7.457e+00  -4.666e-01\n"
)

_FOOTER = "\nEND Ephemeris\n"

_BASIC = _HEADER + _BEGIN + _DATA_BASE + _FOOTER


# --- happy path --------------------------------------------------------------


def test_parses_expected_columns(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.e", _BASIC))
    assert list(df.columns) == ["Epoch", "X", "Y", "Z", "VX", "VY", "VZ"]


def test_row_count_matches_data_records(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.e", _BASIC))
    assert len(df) == 3


def test_epoch_is_scenario_epoch_plus_offset(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.e", _BASIC))
    assert df["Epoch"].dtype == np.dtype("datetime64[ns]")
    assert df["Epoch"].iloc[0] == pd.Timestamp("2026-01-01 12:00:00")
    # 60 s / 120 s offsets from ScenarioEpoch.
    assert df["Epoch"].iloc[1] == pd.Timestamp("2026-01-01 12:01:00")
    assert df["Epoch"].iloc[-1] == pd.Timestamp("2026-01-01 12:02:00")


def test_default_time_scale_is_utc(tmp_path: Path) -> None:
    """The .e file does not declare a scale; the parser defaults to UTC."""
    df = parse(_write(tmp_path / "basic.e", _BASIC))
    assert df.attrs["time_scale"] == "UTC"
    assert df.attrs["epoch_scales"] == {"Epoch": "UTC"}


def test_scenario_epoch_text_preserved(tmp_path: Path) -> None:
    """Raw ScenarioEpoch is kept on attrs so callers can reinterpret."""
    df = parse(_write(tmp_path / "basic.e", _BASIC))
    assert df.attrs["scenario_epoch"] == "01 Jan 2026 12:00:00.000"


def test_state_columns_are_float64(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.e", _BASIC))
    for column in ["X", "Y", "Z", "VX", "VY", "VZ"]:
        assert df[column].dtype == np.float64, column


def test_state_values_round_trip(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.e", _BASIC))
    assert df["X"].iloc[0] == pytest.approx(-5.936e3)
    assert df["VZ"].iloc[0] == pytest.approx(2.207e-16)
    assert df["VZ"].iloc[-1] == pytest.approx(-4.666e-1)


def test_metadata_surfaces_on_attrs(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.e", _BASIC))
    assert df.attrs["central_body"] == "Earth"
    assert df.attrs["coordinate_system"] == "J2000"
    assert df.attrs["interpolation"] == "Lagrange"
    # InterpolationSamplesM1 is coerced to int for downstream arithmetic.
    assert df.attrs["interpolation_degree"] == 4
    assert isinstance(df.attrs["interpolation_degree"], int)
    assert df.attrs["distance_unit"] == "Kilometers"


def test_file_header_surfaces_on_attrs(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "basic.e", _BASIC))
    fh = df.attrs["file_header"]
    assert fh["version"] == "stk.v.11.0"
    assert fh["comments"] == ["# WrittenBy    GMAT R2026a"]


# --- header variants ---------------------------------------------------------


def test_optional_meta_keys_omitted_when_absent(tmp_path: Path) -> None:
    """Older GMAT writers omit InterpolationMethod/InterpolationSamplesM1."""
    minimal_meta = """\
stk.v.10.0
BEGIN Ephemeris
ScenarioEpoch           01 Jan 2026 12:00:00.000
CentralBody             Earth
CoordinateSystem        J2000

EphemerisTimePosVel

"""
    content = minimal_meta + _DATA_BASE + _FOOTER
    df = parse(_write(tmp_path / "minimal.e", content))
    assert df.attrs["central_body"] == "Earth"
    assert "interpolation" not in df.attrs
    assert "interpolation_degree" not in df.attrs
    assert "distance_unit" not in df.attrs


def test_no_writtenby_comment_tolerated(tmp_path: Path) -> None:
    """The ``# WrittenBy …`` comment is a soft convention, not required."""
    content = "stk.v.11.0\n" + _BEGIN + _DATA_BASE + _FOOTER
    df = parse(_write(tmp_path / "no_comment.e", content))
    assert df.attrs["file_header"]["comments"] == []


def test_modjulian_scenario_epoch(tmp_path: Path) -> None:
    """A numeric ScenarioEpoch is interpreted as GMAT ModJulian (MJD)."""
    # GMAT MJD 21545.0 = J2000 = 2000-01-01 12:00 UTC (cross-checked in
    # parsers.epoch._GMAT_MJD_EPOCH).
    mod = """\
stk.v.11.0
BEGIN Ephemeris
ScenarioEpoch           21545.0
CentralBody             Earth
CoordinateSystem        J2000

EphemerisTimePosVel

"""
    content = mod + _DATA_BASE + _FOOTER
    df = parse(_write(tmp_path / "mjd.e", content))
    assert df["Epoch"].iloc[0] == pd.Timestamp("2000-01-01 12:00:00")
    # Raw text preserved for downstream re-interpretation.
    assert df.attrs["scenario_epoch"] == "21545.0"


# --- structural noise --------------------------------------------------------


def test_blank_lines_anywhere_tolerated(tmp_path: Path) -> None:
    content = "\n\n" + _BASIC + "\n\n"
    df = parse(_write(tmp_path / "blanks.e", content))
    assert len(df) == 3


def test_end_ephemeris_optional(tmp_path: Path) -> None:
    """Some emitters omit the trailing END Ephemeris; tolerated."""
    content = _HEADER + _BEGIN + _DATA_BASE
    df = parse(_write(tmp_path / "no_end.e", content))
    assert len(df) == 3


# --- encoding / line endings -------------------------------------------------


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / "bom.e"
    path.write_bytes(b"\xef\xbb\xbf" + _BASIC.encode("utf-8"))
    df = parse(path)
    assert df.attrs["file_header"]["version"] == "stk.v.11.0"


@pytest.mark.parametrize(("newline", "label"), [("\n", "lf"), ("\r\n", "crlf")], ids=["lf", "crlf"])
def test_line_endings_produce_identical_frame(tmp_path: Path, newline: str, label: str) -> None:
    path = _write(tmp_path / f"{label}.e", _BASIC, newline=newline)
    df = parse(path)
    reference = parse(_write(tmp_path / "reference.e", _BASIC, newline="\n"))
    pd.testing.assert_frame_equal(df, reference)


def test_trailing_whitespace_tolerated(tmp_path: Path) -> None:
    """GMAT pads some meta values; the parser must strip and coerce."""
    padded = _BASIC.replace("InterpolationSamplesM1  4", "InterpolationSamplesM1  4   ")
    df = parse(_write(tmp_path / "padded.e", padded))
    assert df.attrs["interpolation_degree"] == 4


# --- type acceptance --------------------------------------------------------


def test_accepts_str_path(tmp_path: Path) -> None:
    """``path`` must accept both ``str`` and ``os.PathLike``."""
    path = _write(tmp_path / "strpath.e", _BASIC)
    df = parse(str(path))
    assert len(df) == 3


# --- malformed input --------------------------------------------------------


def test_empty_file_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.e", "")
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "empty" in str(excinfo.value).lower()
    assert excinfo.value.path == path


def test_only_blank_lines_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "blanks_only.e", "\n\n  \n\t\n")
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "empty" in str(excinfo.value).lower()


def test_no_version_banner_raises(tmp_path: Path) -> None:
    """A file without ``stk.v.X.Y`` is not an STK ephemeris."""
    content = _BEGIN + _DATA_BASE + _FOOTER
    path = _write(tmp_path / "noversion.e", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "stk" in str(excinfo.value).lower()


def test_garbage_pre_header_raises(tmp_path: Path) -> None:
    """A non-comment, non-banner line before BEGIN Ephemeris is rejected."""
    content = "junk line\n" + _HEADER + _BEGIN + _DATA_BASE + _FOOTER
    path = _write(tmp_path / "garbage.e", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "stk.v" in str(excinfo.value) or "comment" in str(excinfo.value).lower()


def test_no_begin_ephemeris_raises(tmp_path: Path) -> None:
    content = _HEADER  # banner only, no BEGIN Ephemeris
    path = _write(tmp_path / "no_begin.e", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "BEGIN Ephemeris" in str(excinfo.value)


def test_no_data_section_raises(tmp_path: Path) -> None:
    """BEGIN Ephemeris present but no ``EphemerisTimePosVel`` marker."""
    no_data = """\
stk.v.11.0
BEGIN Ephemeris
ScenarioEpoch           01 Jan 2026 12:00:00.000
CentralBody             Earth
END Ephemeris
"""
    path = _write(tmp_path / "no_data.e", no_data)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "EphemerisTimePosVel" in str(excinfo.value)


def test_unsupported_data_section_raises(tmp_path: Path) -> None:
    """``EphemerisTimePosVelAcc`` and ``EphemerisTimePos`` are explicit fails."""
    for section in ("EphemerisTimePos", "EphemerisTimePosVelAcc"):
        content = _BASIC.replace("EphemerisTimePosVel", section)
        path = _write(tmp_path / f"{section}.e", content)
        with pytest.raises(GmatOutputParseError) as excinfo:
            parse(path)
        assert "unsupported" in str(excinfo.value).lower()


def test_missing_scenario_epoch_raises(tmp_path: Path) -> None:
    no_epoch = """\
stk.v.11.0
BEGIN Ephemeris
CentralBody             Earth
CoordinateSystem        J2000

EphemerisTimePosVel

"""
    content = no_epoch + _DATA_BASE + _FOOTER
    path = _write(tmp_path / "no_epoch.e", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "ScenarioEpoch" in str(excinfo.value)


def test_malformed_scenario_epoch_raises(tmp_path: Path) -> None:
    bad = _BASIC.replace(
        "ScenarioEpoch           01 Jan 2026 12:00:00.000",
        "ScenarioEpoch           not-a-real-epoch",
    )
    path = _write(tmp_path / "badepoch.e", bad)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    # Mixed alpha/numeric → Gregorian path → typed Gregorian error.
    assert "ScenarioEpoch" in str(excinfo.value)


def test_malformed_meta_line_raises(tmp_path: Path) -> None:
    """A meta line that is not whitespace-separated KEY VALUE is rejected."""
    bad = _BASIC.replace(
        "ScenarioEpoch           01 Jan 2026 12:00:00.000",
        "OnlyOneToken\nScenarioEpoch           01 Jan 2026 12:00:00.000",
    )
    path = _write(tmp_path / "badmeta.e", bad)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "KEY VALUE" in str(excinfo.value) or "expected" in str(excinfo.value).lower()


def test_wrong_record_column_count_raises(tmp_path: Path) -> None:
    bad_row = "180.0  1.0  2.0  3.0\n"  # 4 cols, need 7
    content = _HEADER + _BEGIN + _DATA_BASE + bad_row + _FOOTER
    path = _write(tmp_path / "badcols.e", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "expected 7" in str(excinfo.value)
    assert excinfo.value.path == path


def test_non_numeric_offset_raises(tmp_path: Path) -> None:
    bad = "not-a-number  1.0  2.0  3.0  4.0  5.0  6.0\n"
    content = _HEADER + _BEGIN + bad + _FOOTER
    path = _write(tmp_path / "badoffset.e", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "offset" in str(excinfo.value).lower()


def test_non_numeric_state_raises(tmp_path: Path) -> None:
    bad = "0.0  notanumber  2.0  3.0  4.0  5.0  6.0\n"
    content = _HEADER + _BEGIN + bad + _FOOTER
    path = _write(tmp_path / "badstate.e", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "X" in str(excinfo.value) or "non-numeric" in str(excinfo.value).lower()


def test_content_after_end_ephemeris_raises(tmp_path: Path) -> None:
    content = _HEADER + _BEGIN + _DATA_BASE + _FOOTER + "stray content\n"
    path = _write(tmp_path / "trailing.e", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "END Ephemeris" in str(excinfo.value)


def test_duplicate_version_banner_raises(tmp_path: Path) -> None:
    content = "stk.v.11.0\nstk.v.10.0\n" + _BEGIN + _DATA_BASE + _FOOTER
    path = _write(tmp_path / "dupbanner.e", content)
    with pytest.raises(GmatOutputParseError) as excinfo:
        parse(path)
    assert "duplicate" in str(excinfo.value).lower()


# --- format detection -------------------------------------------------------


def test_is_stk_ephemeris_true_for_stk_file(tmp_path: Path) -> None:
    assert is_stk_ephemeris(_write(tmp_path / "basic.e", _BASIC)) is True


def test_is_stk_ephemeris_skips_leading_comments(tmp_path: Path) -> None:
    """Detection looks past ``# …`` comment lines for the banner."""
    content = "# header comment\n# another\n" + _BASIC
    assert is_stk_ephemeris(_write(tmp_path / "comment_first.e", content)) is True


def test_is_stk_ephemeris_false_for_oem_file(tmp_path: Path) -> None:
    oem = """\
CCSDS_OEM_VERS = 1.0
META_START
META_STOP
"""
    assert is_stk_ephemeris(_write(tmp_path / "fake.oem", oem)) is False


def test_is_stk_ephemeris_false_for_missing_file(tmp_path: Path) -> None:
    # Bubbles to ``False`` so dispatch falls through to the OEM parser, which
    # then raises a typed GmatOutputParseError when it tries to open the path.
    assert is_stk_ephemeris(tmp_path / "does-not-exist.e") is False


def test_is_stk_ephemeris_false_for_empty_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.e", "")
    assert is_stk_ephemeris(path) is False
