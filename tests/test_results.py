"""Unit tests for :class:`gmat_run.results.Results`.

The lazy-materialisation contract is exercised by pointing the constructor at
``ReportFile`` paths that may or may not exist on disk and observing when (and
how often) the parser actually reads them.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from gmat_run.results import Results

# --- helpers -----------------------------------------------------------------

_HEADER = "Sat.UTCGregorian          Sat.Earth.SMA"
_ROW = "26 Nov 2026 12:00:00.000  6578.136"
_REPORT = f"{_HEADER}\n{_ROW}\n"


def _write_report(path: Path) -> Path:
    path.write_text(_REPORT, encoding="utf-8")
    return path


def _empty(tmp_path: Path) -> Results:
    """A Results with no outputs, just an output_dir and log."""
    return Results(output_dir=tmp_path, log="ok")


# --- constructor / attributes ------------------------------------------------


def test_output_dir_and_log_round_trip(tmp_path: Path) -> None:
    result = Results(output_dir=tmp_path, log="hello\nworld")
    assert result.output_dir == tmp_path
    assert result.log == "hello\nworld"


def test_default_mappings_are_empty(tmp_path: Path) -> None:
    result = _empty(tmp_path)
    assert len(result.reports) == 0
    assert len(result.ephemerides) == 0
    assert len(result.ephemeris_paths) == 0
    assert len(result.contacts) == 0
    assert len(result.contact_paths) == 0


def test_path_mappings_are_read_only(tmp_path: Path) -> None:
    """Path mappings are MappingProxyType views — assignment must fail."""
    result = Results(
        output_dir=tmp_path,
        log="",
        ephemeris_paths={"E1": tmp_path / "E1.eph"},
        contact_paths={"C1": tmp_path / "C1.txt"},
    )
    with pytest.raises(TypeError):
        result.ephemeris_paths["E2"] = tmp_path / "E2.eph"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.contact_paths["C2"] = tmp_path / "C2.txt"  # type: ignore[index]


def test_input_mappings_are_defensively_copied(tmp_path: Path) -> None:
    """Mutating the caller's dict after construction must not affect Results."""
    eph_in: dict[str, Path] = {"E1": tmp_path / "E1.eph"}
    con_in: dict[str, Path] = {"C1": tmp_path / "C1.txt"}
    rep_in: dict[str, Path] = {"R1": _write_report(tmp_path / "R1.txt")}
    result = Results(
        output_dir=tmp_path,
        log="",
        report_paths=rep_in,
        ephemeris_paths=eph_in,
        contact_paths=con_in,
    )

    eph_in["E2"] = tmp_path / "E2.eph"
    con_in["C2"] = tmp_path / "C2.txt"
    rep_in["R2"] = tmp_path / "R2.txt"

    assert list(result.ephemeris_paths) == ["E1"]
    assert list(result.contact_paths) == ["C1"]
    assert list(result.reports) == ["R1"]


# --- reports: happy path -----------------------------------------------------


def test_reports_keyed_by_resource_name(tmp_path: Path) -> None:
    result = Results(
        output_dir=tmp_path,
        log="",
        report_paths={"ReportFile1": _write_report(tmp_path / "ReportFile1.txt")},
    )
    assert list(result.reports) == ["ReportFile1"]
    assert "ReportFile1" in result.reports
    assert len(result.reports) == 1


def test_report_access_returns_dataframe(tmp_path: Path) -> None:
    result = Results(
        output_dir=tmp_path,
        log="",
        report_paths={"ReportFile1": _write_report(tmp_path / "ReportFile1.txt")},
    )
    df = result.reports["ReportFile1"]
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Sat.UTCGregorian", "Sat.Earth.SMA"]
    assert len(df) == 1


# --- reports: lazy materialisation -------------------------------------------


def test_construction_does_not_read_files(tmp_path: Path) -> None:
    """Pointing at a non-existent path must not raise — the parser is lazy."""
    missing = tmp_path / "never_written.txt"
    result = Results(
        output_dir=tmp_path,
        log="",
        report_paths={"R1": missing},
    )
    # All mapping ops that don't touch the value must work fine.
    assert "R1" in result.reports
    assert list(result.reports) == ["R1"]
    assert len(result.reports) == 1


def test_value_access_on_missing_file_raises(tmp_path: Path) -> None:
    """First access is where the parser actually runs — and where I/O fails."""
    result = Results(
        output_dir=tmp_path,
        log="",
        report_paths={"R1": tmp_path / "never_written.txt"},
    )
    with pytest.raises(FileNotFoundError):
        _ = result.reports["R1"]


def test_repeated_access_is_cached(tmp_path: Path) -> None:
    """Same DataFrame object on every access — the parser runs once."""
    result = Results(
        output_dir=tmp_path,
        log="",
        report_paths={"R1": _write_report(tmp_path / "R1.txt")},
    )
    first = result.reports["R1"]
    second = result.reports["R1"]
    assert first is second


def test_cache_survives_underlying_file_deletion(tmp_path: Path) -> None:
    """Once parsed, the DataFrame is independent of the source file."""
    path = _write_report(tmp_path / "R1.txt")
    result = Results(output_dir=tmp_path, log="", report_paths={"R1": path})
    df = result.reports["R1"]
    path.unlink()
    again = result.reports["R1"]
    assert again is df


def test_unknown_report_key_raises(tmp_path: Path) -> None:
    result = _empty(tmp_path)
    with pytest.raises(KeyError) as excinfo:
        _ = result.reports["does_not_exist"]
    assert excinfo.value.args == ("does_not_exist",)


# --- ephemerides / contacts (v0.1 stub) --------------------------------------


def test_ephemeris_paths_round_trip(tmp_path: Path) -> None:
    eph = tmp_path / "E1.eph"
    result = Results(output_dir=tmp_path, log="", ephemeris_paths={"E1": eph})
    assert result.ephemeris_paths["E1"] == eph
    assert list(result.ephemerides) == ["E1"]
    assert "E1" in result.ephemerides
    assert len(result.ephemerides) == 1


def test_ephemeris_value_access_is_unimplemented(tmp_path: Path) -> None:
    result = Results(
        output_dir=tmp_path,
        log="",
        ephemeris_paths={"E1": tmp_path / "E1.eph"},
    )
    with pytest.raises(NotImplementedError) as excinfo:
        _ = result.ephemerides["E1"]
    message = str(excinfo.value)
    assert ".eph" in message
    assert "v0.2" in message
    assert "ephemeris_paths" in message
    assert "'E1'" in message


def test_ephemerides_unknown_key_raises_keyerror_not_notimplemented(tmp_path: Path) -> None:
    """Membership check must distinguish unknown keys from unimplemented values."""
    result = _empty(tmp_path)
    with pytest.raises(KeyError):
        _ = result.ephemerides["nope"]


def test_contact_paths_round_trip(tmp_path: Path) -> None:
    con = tmp_path / "C1.txt"
    result = Results(output_dir=tmp_path, log="", contact_paths={"C1": con})
    assert result.contact_paths["C1"] == con
    assert list(result.contacts) == ["C1"]
    assert "C1" in result.contacts
    assert len(result.contacts) == 1


def test_contact_value_access_is_unimplemented(tmp_path: Path) -> None:
    result = Results(
        output_dir=tmp_path,
        log="",
        contact_paths={"C1": tmp_path / "C1.txt"},
    )
    with pytest.raises(NotImplementedError) as excinfo:
        _ = result.contacts["C1"]
    message = str(excinfo.value)
    assert "ContactLocator" in message
    assert "v0.2" in message
    assert "contact_paths" in message
    assert "'C1'" in message


def test_contacts_unknown_key_raises_keyerror_not_notimplemented(tmp_path: Path) -> None:
    result = _empty(tmp_path)
    with pytest.raises(KeyError):
        _ = result.contacts["nope"]


# --- mapping protocol --------------------------------------------------------


@pytest.mark.parametrize("attr", ["reports", "ephemerides", "contacts"])
def test_mapping_attrs_are_mappings(tmp_path: Path, attr: str) -> None:
    """The three keyed views must satisfy ``Mapping`` at runtime, not just typing."""
    result = _empty(tmp_path)
    assert isinstance(getattr(result, attr), Mapping)


def test_iteration_preserves_insertion_order(tmp_path: Path) -> None:
    """dict ordering is part of the contract — ``Mission.run`` will hand keys
    in declaration order and downstream code shouldn't have to re-sort."""
    paths = {
        "ReportC": _write_report(tmp_path / "C.txt"),
        "ReportA": _write_report(tmp_path / "A.txt"),
        "ReportB": _write_report(tmp_path / "B.txt"),
    }
    result = Results(output_dir=tmp_path, log="", report_paths=paths)
    assert list(result.reports) == ["ReportC", "ReportA", "ReportB"]
