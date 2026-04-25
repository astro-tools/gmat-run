"""Unit tests for :class:`gmat_run.results.Results`.

The lazy-materialisation contract is exercised by pointing the constructor at
``ReportFile`` paths that may or may not exist on disk and observing when (and
how often) the parser actually reads them.
"""

import tempfile
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


# --- ephemerides (lazy, v0.2) ------------------------------------------------


_EPH_FILE = """\
CCSDS_OEM_VERS = 1.0
CREATION_DATE  = 2026-04-25T18:54:25
ORIGINATOR     = GMAT USER

META_START
OBJECT_NAME          = Sat
OBJECT_ID            = SatId
CENTER_NAME          = Earth
REF_FRAME            = EME2000
TIME_SYSTEM          = UTC
START_TIME           = 2026-01-01T12:00:00.000
STOP_TIME            = 2026-01-01T12:00:00.000
INTERPOLATION        = LAGRANGE
INTERPOLATION_DEGREE = 4
META_STOP

2026-01-01T12:00:00.000  -5.936e+03   1.590e+03   3.336e+03  -1.955e+00  -7.296e+00   2.206e-16
"""


def _write_eph(path: Path) -> Path:
    path.write_text(_EPH_FILE, encoding="utf-8")
    return path


def test_ephemeris_paths_round_trip(tmp_path: Path) -> None:
    eph = tmp_path / "E1.eph"
    result = Results(output_dir=tmp_path, log="", ephemeris_paths={"E1": eph})
    assert result.ephemeris_paths["E1"] == eph
    assert list(result.ephemerides) == ["E1"]
    assert "E1" in result.ephemerides
    assert len(result.ephemerides) == 1


def test_ephemeris_value_access_returns_dataframe(tmp_path: Path) -> None:
    """``.ephemerides[k]`` lazily parses the ``.oem`` and returns a typed frame."""
    path = _write_eph(tmp_path / "E1.oem")
    result = Results(output_dir=tmp_path, log="", ephemeris_paths={"E1": path})
    df = result.ephemerides["E1"]
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Epoch", "X", "Y", "Z", "VX", "VY", "VZ"]
    assert df.attrs["coordinate_system"] == "EME2000"
    assert df.attrs["epoch_scales"] == {"Epoch": "UTC"}


def test_ephemeris_lazy_parse_caches(tmp_path: Path) -> None:
    """Once parsed, the DataFrame is independent of the source file."""
    path = _write_eph(tmp_path / "E1.oem")
    result = Results(output_dir=tmp_path, log="", ephemeris_paths={"E1": path})
    df = result.ephemerides["E1"]
    path.unlink()
    again = result.ephemerides["E1"]
    assert again is df


def test_ephemerides_unknown_key_raises_keyerror(tmp_path: Path) -> None:
    """Membership check distinguishes unknown keys from a real parse miss."""
    result = _empty(tmp_path)
    with pytest.raises(KeyError):
        _ = result.ephemerides["nope"]


# --- ephemeris format dispatch (CCSDS-OEM vs STK-TimePosVel) -----------------


_STK_FILE = """\
stk.v.11.0
# WrittenBy    GMAT R2026a
BEGIN Ephemeris
NumberOfEphemerisPoints 1
ScenarioEpoch           01 Jan 2026 12:00:00.000
CentralBody             Earth
CoordinateSystem        J2000

EphemerisTimePosVel

0.0  -5.936e+03  1.591e+03  3.337e+03  -1.955e+00  -7.296e+00  2.207e-16

END Ephemeris
"""


def test_ephemeris_dispatch_routes_oem_and_stk(tmp_path: Path) -> None:
    """One Mission.run can declare both formats; each file routes by content."""
    oem = _write_eph(tmp_path / "OEM.oem")
    stk = tmp_path / "STK.e"
    stk.write_text(_STK_FILE, encoding="utf-8")

    result = Results(
        output_dir=tmp_path,
        log="",
        ephemeris_paths={"OEM": oem, "STK": stk},
    )

    oem_df = result.ephemerides["OEM"]
    stk_df = result.ephemerides["STK"]

    # OEM-specific attr — confirms the OEM parser ran.
    assert oem_df.attrs["coordinate_system"] == "EME2000"
    # STK-specific attr — confirms the STK parser ran.
    assert stk_df.attrs["coordinate_system"] == "J2000"
    assert stk_df.attrs["scenario_epoch"] == "01 Jan 2026 12:00:00.000"


def test_ephemeris_dispatch_ignores_extension(tmp_path: Path) -> None:
    """Format detection is content-based; ``.oem`` extension on STK content is fine."""
    misnamed = tmp_path / "looks_like_oem.oem"
    misnamed.write_text(_STK_FILE, encoding="utf-8")
    result = Results(output_dir=tmp_path, log="", ephemeris_paths={"E": misnamed})
    df = result.ephemerides["E"]
    # An OEM parse on this file would fail immediately; an STK parse succeeds.
    assert df.attrs["scenario_epoch"] == "01 Jan 2026 12:00:00.000"


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


# --- persist ----------------------------------------------------------------


def _result_with_workspace(
    tmp_path: Path,
    *,
    rel_report: str = "r1.txt",
    rel_eph: str = "e1.eph",
    rel_con: str = "c1.txt",
) -> tuple[Results, Path]:
    """A Results pointing at a populated workspace dir, no temp-dir handle.

    Returns ``(result, workspace_dir)``. The workspace contains a parseable
    report, an ephemeris file, a contact file, and a log — same layout
    ``Mission.run`` produces.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_report(workspace / rel_report)
    (workspace / rel_eph).write_text("eph data\n", encoding="utf-8")
    (workspace / rel_con).write_text("contact data\n", encoding="utf-8")
    (workspace / "GmatLog.txt").write_text("log\n", encoding="utf-8")
    result = Results(
        output_dir=workspace,
        log="log\n",
        report_paths={"R1": workspace / rel_report},
        ephemeris_paths={"E1": workspace / rel_eph},
        contact_paths={"C1": workspace / rel_con},
    )
    return result, workspace


class TestPersist:
    def test_copies_artefacts_into_dest(self, tmp_path: Path) -> None:
        result, _ = _result_with_workspace(tmp_path)
        dest = tmp_path / "persisted"

        result.persist(dest)

        assert (dest / "r1.txt").exists()
        assert (dest / "e1.eph").read_text(encoding="utf-8") == "eph data\n"
        assert (dest / "c1.txt").read_text(encoding="utf-8") == "contact data\n"
        assert (dest / "GmatLog.txt").read_text(encoding="utf-8") == "log\n"

    def test_updates_output_dir_and_path_mappings(self, tmp_path: Path) -> None:
        result, _ = _result_with_workspace(tmp_path)
        dest = tmp_path / "persisted"

        result.persist(dest)

        assert result.output_dir == dest
        assert result.ephemeris_paths["E1"] == dest / "e1.eph"
        assert result.contact_paths["C1"] == dest / "c1.txt"
        # Reports' underlying path mapping is rebased too.
        assert result.reports._paths["R1"] == dest / "r1.txt"  # type: ignore[attr-defined]

    def test_returns_self_for_chaining(self, tmp_path: Path) -> None:
        result, _ = _result_with_workspace(tmp_path)
        assert result.persist(tmp_path / "dest") is result

    def test_lazy_report_reads_from_persisted_path(self, tmp_path: Path) -> None:
        result, workspace = _result_with_workspace(tmp_path)
        dest = tmp_path / "persisted"

        result.persist(dest)
        # Wipe the original to confirm the lazy parse hits the new location.
        for f in workspace.iterdir():
            f.unlink()
        df = result.reports["R1"]
        assert list(df.columns) == ["Sat.UTCGregorian", "Sat.Earth.SMA"]

    def test_preserves_already_cached_dataframes(self, tmp_path: Path) -> None:
        result, _ = _result_with_workspace(tmp_path)
        df_before = result.reports["R1"]

        result.persist(tmp_path / "persisted")

        df_after = result.reports["R1"]
        assert df_after is df_before

    def test_releases_temp_workspace(self, tmp_path: Path) -> None:
        # Stand up a real TemporaryDirectory so persist exercises the cleanup
        # path, mirroring what Mission.run() builds for the default case.
        tmpdir = tempfile.TemporaryDirectory(prefix="gmat-run-test-")
        workspace = Path(tmpdir.name)
        _write_report(workspace / "r1.txt")
        result = Results(
            output_dir=workspace,
            log="",
            report_paths={"R1": workspace / "r1.txt"},
        )
        result._workspace = tmpdir

        result.persist(tmp_path / "persisted")

        assert result._workspace is None
        assert not workspace.is_dir()

    def test_explicit_working_dir_is_not_deleted(self, tmp_path: Path) -> None:
        # _workspace stays None when the run had a user-supplied working_dir.
        # persist must not touch that directory.
        result, workspace = _result_with_workspace(tmp_path)
        assert result._workspace is None

        result.persist(tmp_path / "persisted")

        assert workspace.is_dir()
        assert (workspace / "r1.txt").exists()

    def test_absolute_paths_outside_workspace_are_not_migrated(self, tmp_path: Path) -> None:
        # A ReportFile.Filename like "/abs/elsewhere/report.txt" lands outside
        # the workspace and was not rewritten by Mission.run. persist must
        # leave that path intact rather than silently relocating it.
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external = tmp_path / "elsewhere" / "report.txt"
        external.parent.mkdir()
        external.write_text(_REPORT, encoding="utf-8")
        result = Results(
            output_dir=workspace,
            log="",
            report_paths={"Inside": _write_report(workspace / "in.txt")},
            ephemeris_paths={"Outside": external},
        )

        result.persist(tmp_path / "persisted")

        assert result.ephemeris_paths["Outside"] == external
        assert result.reports._paths["Inside"] == tmp_path / "persisted" / "in.txt"  # type: ignore[attr-defined]

    def test_idempotent_when_dest_equals_output_dir(self, tmp_path: Path) -> None:
        result, workspace = _result_with_workspace(tmp_path)
        result.persist(workspace)
        # Path mappings are unchanged.
        assert result.output_dir == workspace
        assert result.reports._paths["R1"] == workspace / "r1.txt"  # type: ignore[attr-defined]

    def test_creates_missing_destination(self, tmp_path: Path) -> None:
        result, _ = _result_with_workspace(tmp_path)
        dest = tmp_path / "nested" / "deep" / "persisted"
        assert not dest.exists()

        result.persist(dest)

        assert dest.is_dir()
        assert (dest / "r1.txt").exists()

    def test_can_persist_twice(self, tmp_path: Path) -> None:
        result, _ = _result_with_workspace(tmp_path)
        first = tmp_path / "first"
        second = tmp_path / "second"

        result.persist(first)
        result.persist(second)

        assert result.output_dir == second
        assert (second / "r1.txt").exists()
        assert result.reports._paths["R1"] == second / "r1.txt"  # type: ignore[attr-defined]
