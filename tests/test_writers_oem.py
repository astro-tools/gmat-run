"""Unit tests for :func:`gmat_run.writers.oem.write_oem`.

The round-trip tests pair the writer with :func:`gmat_run.parsers.ephemeris.parse`
and assert that an ephemeris emitted by the writer is read back with the same
columns, row count, and surfaced attrs. Hand-built DataFrames cover the error
paths (missing required attrs, unknown frame, unsupported time scale) so a
typo in caller code surfaces here rather than as a malformed file downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from gmat_run.parsers.ephemeris import parse as parse_oem
from gmat_run.writers.oem import write_oem

_FIXTURE = Path(__file__).parent / "fixtures" / "ephemeris" / "Ex_LEOEphemeris.oem"


def _hand_built_oem_frame() -> pd.DataFrame:
    """Smallest legal OEM DataFrame, matching the parser's output shape."""
    df = pd.DataFrame(
        {
            "Epoch": pd.to_datetime(["2026-01-01T12:00:00.000", "2026-01-01T12:01:00.000"]),
            "X": [-5936.0, -6040.0],
            "Y": [1590.0, 1149.0],
            "Z": [3336.0, 3329.0],
            "VX": [-1.95, -1.53],
            "VY": [-7.29, -7.39],
            "VZ": [0.0, -0.23],
        }
    )
    df.attrs["coordinate_system"] = "EME2000"
    df.attrs["central_body"] = "Earth"
    df.attrs["epoch_scales"] = {"Epoch": "UTC"}
    df.attrs["time_scale"] = "UTC"
    df.attrs["object_name"] = "Sat"
    return df


# --- happy path -------------------------------------------------------------


def test_round_trip_preserves_state_columns_and_attrs(tmp_path: Path) -> None:
    """Parse a GMAT-emitted OEM, write it, re-parse: state and attrs survive."""
    original = parse_oem(_FIXTURE)
    out = write_oem(original, tmp_path / "round_trip.oem")
    reparsed = parse_oem(out)

    assert list(reparsed.columns) == ["Epoch", "X", "Y", "Z", "VX", "VY", "VZ"]
    assert len(reparsed) == len(original)
    assert reparsed.attrs["coordinate_system"] == original.attrs["coordinate_system"]
    assert reparsed.attrs["central_body"] == original.attrs["central_body"]
    assert reparsed.attrs["object_name"] == original.attrs["object_name"]
    assert reparsed.attrs["epoch_scales"] == original.attrs["epoch_scales"]
    assert reparsed["Epoch"].iloc[0] == original["Epoch"].iloc[0]
    assert reparsed["X"].iloc[0] == pytest.approx(original["X"].iloc[0])


def test_returns_destination_path(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    out = write_oem(df, tmp_path / "out.oem")
    assert out == tmp_path / "out.oem"
    assert out.exists()


def test_creates_missing_parent_directories(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    nested = tmp_path / "nested" / "dirs" / "out.oem"
    write_oem(df, nested)
    assert nested.exists()


def test_originator_kwarg_overrides_default(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    out = write_oem(df, tmp_path / "out.oem", originator="ACME Aerospace")
    text = out.read_text(encoding="utf-8")
    assert "ORIGINATOR" in text
    assert "ACME Aerospace" in text


def test_object_name_kwarg_overrides_attr(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    df.attrs["object_name"] = "Sat"
    out = write_oem(df, tmp_path / "out.oem", object_name="OverrideSat")
    text = out.read_text(encoding="utf-8")
    assert "OverrideSat" in text
    # The attr value must not leak through when the kwarg is set.
    assert "OBJECT_NAME              = Sat\n" not in text


def test_object_name_falls_back_to_unknown(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    df.attrs.pop("object_name")
    out = write_oem(df, tmp_path / "out.oem")
    text = out.read_text(encoding="utf-8")
    assert "UNKNOWN" in text


def test_interpolation_metadata_survives_round_trip(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    df.attrs["interpolation"] = "LAGRANGE"
    df.attrs["interpolation_degree"] = 7
    out = write_oem(df, tmp_path / "out.oem")
    reparsed = parse_oem(out)
    assert reparsed.attrs["interpolation"] == "LAGRANGE"
    assert reparsed.attrs["interpolation_degree"] == 7


# --- frame mapping ----------------------------------------------------------


def test_gmat_frame_alias_maps_to_ccsds(tmp_path: Path) -> None:
    """A GMAT-style coordinate-system name is rewritten to its CCSDS equivalent."""
    df = _hand_built_oem_frame()
    df.attrs["coordinate_system"] = "EarthMJ2000Eq"
    out = write_oem(df, tmp_path / "out.oem")
    text = out.read_text(encoding="utf-8")
    assert "REF_FRAME" in text
    assert "EME2000" in text
    assert "EarthMJ2000Eq" not in text


def test_stk_j2000_maps_to_eme2000(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    df.attrs["coordinate_system"] = "J2000"
    out = write_oem(df, tmp_path / "out.oem")
    assert "EME2000" in out.read_text(encoding="utf-8")


def test_unknown_frame_raises_value_error(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    df.attrs["coordinate_system"] = "MadeUpFrame"
    with pytest.raises(ValueError, match="unknown coordinate system"):
        write_oem(df, tmp_path / "out.oem")


def test_missing_frame_raises_value_error(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    df.attrs.pop("coordinate_system")
    with pytest.raises(ValueError, match="coordinate_system"):
        write_oem(df, tmp_path / "out.oem")


# --- time scale validation --------------------------------------------------


@pytest.mark.parametrize("scale", ["A1", "TAI", "UTC", "TT", "TDB"])
def test_supported_time_scales_round_trip(tmp_path: Path, scale: str) -> None:
    df = _hand_built_oem_frame()
    df.attrs["epoch_scales"] = {"Epoch": scale}
    df.attrs["time_scale"] = scale
    out = write_oem(df, tmp_path / "out.oem")
    text = out.read_text(encoding="utf-8")
    assert "TIME_SYSTEM" in text
    assert scale in text


def test_unsupported_time_scale_raises_value_error(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    df.attrs["epoch_scales"] = {"Epoch": "GPS"}
    df.attrs["time_scale"] = "GPS"
    with pytest.raises(ValueError, match="unsupported TIME_SYSTEM"):
        write_oem(df, tmp_path / "out.oem")


def test_falls_back_to_time_scale_attr_when_epoch_scales_missing(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    df.attrs.pop("epoch_scales")
    df.attrs["time_scale"] = "TAI"
    out = write_oem(df, tmp_path / "out.oem")
    assert "TAI" in out.read_text(encoding="utf-8")


def test_missing_time_scale_raises_value_error(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    df.attrs.pop("epoch_scales")
    df.attrs.pop("time_scale")
    with pytest.raises(ValueError, match="time scale"):
        write_oem(df, tmp_path / "out.oem")


# --- other guards -----------------------------------------------------------


def test_missing_central_body_raises_value_error(tmp_path: Path) -> None:
    df = _hand_built_oem_frame()
    df.attrs.pop("central_body")
    with pytest.raises(ValueError, match="central_body"):
        write_oem(df, tmp_path / "out.oem")


def test_empty_dataframe_raises_value_error(tmp_path: Path) -> None:
    df = _hand_built_oem_frame().iloc[0:0].copy()
    df.attrs["coordinate_system"] = "EME2000"
    df.attrs["central_body"] = "Earth"
    df.attrs["epoch_scales"] = {"Epoch": "UTC"}
    df.attrs["time_scale"] = "UTC"
    with pytest.raises(ValueError, match="empty DataFrame"):
        write_oem(df, tmp_path / "out.oem")


def test_missing_extra_raises_import_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``ccsds_ndm`` is not importable, surface a hint to install the extra."""
    df = _hand_built_oem_frame()
    # Hide every cached ccsds_ndm submodule so the writer's local import path runs
    # cold and trips ImportError. The hint string is the load-bearing assertion.
    for name in [n for n in sys.modules if n == "ccsds_ndm" or n.startswith("ccsds_ndm.")]:
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(ImportError, match=r"\[ccsds-ndm\]"):
        write_oem(df, tmp_path / "out.oem")
