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
    assert len(result.solver_runs) == 0
    assert len(result.solver_paths) == 0
    assert result.converged == {}


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


# --- ephemerides (lazy) ------------------------------------------------------


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


# --- ephemeris format dispatch (CCSDS-OEM vs STK-TimePosVel vs SPK) ----------


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

# Committed SPK fixture, generated by tests/fixtures/ephemeris/_make_spk.py.
_SPK_FIXTURE = Path(__file__).parent / "fixtures" / "ephemeris" / "Ex_SPK.bsp"


def test_ephemeris_dispatch_routes_oem_stk_and_spk(tmp_path: Path) -> None:
    """One Mission.run can declare three formats; each file routes by content."""
    oem = _write_eph(tmp_path / "OEM.oem")
    stk = tmp_path / "STK.e"
    stk.write_text(_STK_FILE, encoding="utf-8")
    spk = tmp_path / "SPK.bsp"
    spk.write_bytes(_SPK_FIXTURE.read_bytes())

    result = Results(
        output_dir=tmp_path,
        log="",
        ephemeris_paths={"OEM": oem, "STK": stk, "SPK": spk},
    )

    oem_df = result.ephemerides["OEM"]
    stk_df = result.ephemerides["STK"]
    spk_df = result.ephemerides["SPK"]

    # OEM-specific attr — confirms the OEM parser ran.
    assert oem_df.attrs["coordinate_system"] == "EME2000"
    # STK-specific attr — confirms the STK parser ran.
    assert stk_df.attrs["coordinate_system"] == "J2000"
    assert stk_df.attrs["scenario_epoch"] == "01 Jan 2026 12:00:00.000"
    # SPK-specific attr — confirms the SPK parser ran (TDB source scale
    # is unique to the binary format).
    assert spk_df.attrs["time_scale"] == "TDB"
    assert spk_df.attrs["coordinate_system"] == "J2000"


def test_ephemeris_dispatch_ignores_extension(tmp_path: Path) -> None:
    """Format detection is content-based; ``.oem`` extension on STK content is fine."""
    misnamed = tmp_path / "looks_like_oem.oem"
    misnamed.write_text(_STK_FILE, encoding="utf-8")
    result = Results(output_dir=tmp_path, log="", ephemeris_paths={"E": misnamed})
    df = result.ephemerides["E"]
    # An OEM parse on this file would fail immediately; an STK parse succeeds.
    assert df.attrs["scenario_epoch"] == "01 Jan 2026 12:00:00.000"


def test_ephemeris_dispatch_routes_spk_under_misleading_extension(
    tmp_path: Path,
) -> None:
    """A ``.oem`` extension on DAF/SPK bytes still routes to the SPK parser."""
    misnamed = tmp_path / "looks_like_oem.oem"
    misnamed.write_bytes(_SPK_FIXTURE.read_bytes())
    result = Results(output_dir=tmp_path, log="", ephemeris_paths={"E": misnamed})
    df = result.ephemerides["E"]
    assert df.attrs["time_scale"] == "TDB"


def test_contact_paths_round_trip(tmp_path: Path) -> None:
    con = tmp_path / "C1.txt"
    result = Results(output_dir=tmp_path, log="", contact_paths={"C1": con})
    assert result.contact_paths["C1"] == con
    assert list(result.contacts) == ["C1"]
    assert "C1" in result.contacts
    assert len(result.contacts) == 1


_CONTACT_LEGACY_FILE = """\
Target: Sat

Observer: AthGS
Start Time (UTC)            Stop Time (UTC)               Duration (s)
09 Jan 2010 20:36:24.626    09 Jan 2010 23:05:18.684      8934.0587546


Number of events : 1


"""


def _write_contact(path: Path) -> Path:
    path.write_text(_CONTACT_LEGACY_FILE, encoding="utf-8")
    return path


def test_contact_value_access_returns_dataframe(tmp_path: Path) -> None:
    """``.contacts[k]`` lazily parses the report and returns a typed frame."""
    path = _write_contact(tmp_path / "C1.txt")
    result = Results(output_dir=tmp_path, log="", contact_paths={"C1": path})
    df = result.contacts["C1"]
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Observer", "Start", "Stop", "Duration"]
    assert df.attrs["report_format"] == "Legacy"
    assert df.attrs["target"] == "Sat"


def test_contact_lazy_parse_caches(tmp_path: Path) -> None:
    """Once parsed, the DataFrame is independent of the source file."""
    path = _write_contact(tmp_path / "C1.txt")
    result = Results(output_dir=tmp_path, log="", contact_paths={"C1": path})
    df = result.contacts["C1"]
    path.unlink()
    again = result.contacts["C1"]
    assert again is df


def test_contact_construction_does_not_read_files(tmp_path: Path) -> None:
    """Pointing at a non-existent path must not raise — the parser is lazy."""
    result = Results(
        output_dir=tmp_path,
        log="",
        contact_paths={"C1": tmp_path / "never_written.txt"},
    )
    assert "C1" in result.contacts
    assert list(result.contacts) == ["C1"]
    assert len(result.contacts) == 1


def test_contacts_unknown_key_raises_keyerror(tmp_path: Path) -> None:
    """Membership check distinguishes unknown keys from a real parse miss."""
    result = _empty(tmp_path)
    with pytest.raises(KeyError):
        _ = result.contacts["nope"]


# --- solver_runs (lazy) ------------------------------------------------------


_SOLVER_DC_CONVERGED = """\
********************************************************
*** Targeter Text File
*** Using Differential Correction
*** 1 variables
*** 1 goals
*** SolverMode:  Solve
********************************************************

Iteration 1
Running Nominal Pass
Variables:
   Burn.V = 0.5

Goals and achieved values:
   Sat.SMA  Desired: 7000 Achieved: 6999.9995
   Tolerance: 0.001

********************************************************
*** Targeting Completed in 1 iterations
********************************************************
"""

# Same file with the achieved value far from the goal — does not converge.
_SOLVER_DC_DIVERGED = _SOLVER_DC_CONVERGED.replace("6999.9995", "5000.0")


def _write_solver(path: Path, content: str = _SOLVER_DC_CONVERGED) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_solver_paths_round_trip(tmp_path: Path) -> None:
    data = tmp_path / "DC.data"
    result = Results(output_dir=tmp_path, log="", solver_paths={"DC": data})
    assert result.solver_paths["DC"] == data
    assert list(result.solver_runs) == ["DC"]
    assert "DC" in result.solver_runs
    assert len(result.solver_runs) == 1


def test_solver_run_value_access_returns_dataframe(tmp_path: Path) -> None:
    """``.solver_runs[k]`` lazily parses the ``.data`` file into a typed frame."""
    data = _write_solver(tmp_path / "DC.data")
    result = Results(output_dir=tmp_path, log="", solver_paths={"DC": data})
    df = result.solver_runs["DC"]
    assert isinstance(df, pd.DataFrame)
    assert "iteration" in df.columns
    assert "Burn.V" in df.columns
    assert df.attrs["solver_type"] == "DifferentialCorrector"


def test_solver_run_construction_does_not_read_files(tmp_path: Path) -> None:
    """Pointing at a non-existent path must not raise — the parser is lazy."""
    result = Results(output_dir=tmp_path, log="", solver_paths={"DC": tmp_path / "never.data"})
    assert "DC" in result.solver_runs
    assert list(result.solver_runs) == ["DC"]
    assert len(result.solver_runs) == 1


def test_solver_run_lazy_parse_caches(tmp_path: Path) -> None:
    """Once parsed, the DataFrame is independent of the source file."""
    data = _write_solver(tmp_path / "DC.data")
    result = Results(output_dir=tmp_path, log="", solver_paths={"DC": data})
    df = result.solver_runs["DC"]
    data.unlink()
    assert result.solver_runs["DC"] is df


def test_solver_runs_unknown_key_raises_keyerror(tmp_path: Path) -> None:
    result = _empty(tmp_path)
    with pytest.raises(KeyError):
        _ = result.solver_runs["nope"]


def test_solver_paths_are_read_only(tmp_path: Path) -> None:
    result = Results(output_dir=tmp_path, log="", solver_paths={"DC": tmp_path / "DC.data"})
    with pytest.raises(TypeError):
        result.solver_paths["DC2"] = tmp_path / "DC2.data"  # type: ignore[index]


def test_solver_paths_defensively_copied(tmp_path: Path) -> None:
    paths: dict[str, Path] = {"DC": tmp_path / "DC.data"}
    result = Results(output_dir=tmp_path, log="", solver_paths=paths)
    paths["DC2"] = tmp_path / "DC2.data"
    assert list(result.solver_runs) == ["DC"]


def test_converged_reflects_each_solver(tmp_path: Path) -> None:
    ok = _write_solver(tmp_path / "DC.data", _SOLVER_DC_CONVERGED)
    bad = _write_solver(tmp_path / "DC2.data", _SOLVER_DC_DIVERGED)
    result = Results(output_dir=tmp_path, log="", solver_paths={"DC": ok, "DC2": bad})
    assert result.converged == {"DC": True, "DC2": False}


def test_solver_max_iterations_threaded_to_parser(tmp_path: Path) -> None:
    """``solver_max_iterations`` reaches the parser — it distinguishes max_iter."""
    data = _write_solver(tmp_path / "DC.data", _SOLVER_DC_DIVERGED)
    capped = Results(
        output_dir=tmp_path,
        log="",
        solver_paths={"DC": data},
        solver_max_iterations={"DC": 1},
    )
    assert capped.solver_runs["DC"]["status"].iloc[-1] == "max_iter"
    # Without the hint the same non-converged run cannot be told from a failure.
    uncapped = Results(output_dir=tmp_path, log="", solver_paths={"DC": data})
    assert uncapped.solver_runs["DC"]["status"].iloc[-1] == "failed"


def test_solver_run_persist_rebases_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_solver(workspace / "DC.data")
    result = Results(output_dir=workspace, log="", solver_paths={"DC": workspace / "DC.data"})
    dest = tmp_path / "persisted"

    result.persist(dest)

    assert result.solver_paths["DC"] == dest / "DC.data"
    # Lazy parse now resolves against the persisted copy.
    for f in workspace.iterdir():
        f.unlink()
    assert result.solver_runs["DC"].attrs["solver_type"] == "DifferentialCorrector"


# --- mapping protocol --------------------------------------------------------


@pytest.mark.parametrize("attr", ["reports", "ephemerides", "contacts", "solver_runs"])
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

    def test_resolves_relative_dest_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A relative ``dest`` is anchored to the caller's CWD at submit
        # time, not interpreted relative to the workspace or some other
        # implicit base.
        result, _ = _result_with_workspace(tmp_path)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        result.persist("persisted/run_x")

        expected = (cwd / "persisted/run_x").resolve()
        assert result.output_dir.is_absolute()
        assert result.output_dir == expected
        assert (expected / "r1.txt").exists()

    def test_expands_tilde_in_dest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regression guard so the path-helper contract is pinned. Both HOME
        # (POSIX) and USERPROFILE (Windows) are stubbed because Python's
        # expanduser reads USERPROFILE on Windows, not HOME.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        result, _ = _result_with_workspace(tmp_path)

        result.persist("~/persisted_home")

        assert result.output_dir == (tmp_path / "persisted_home").resolve()


# --- write_oem / write_oem_all ----------------------------------------------


def test_write_oem_returns_path_and_writes_file(tmp_path: Path) -> None:
    """``Results.write_oem`` materialises the lazy parse and emits a file."""
    eph = _write_eph(tmp_path / "E1.oem")
    result = Results(output_dir=tmp_path, log="", ephemeris_paths={"E1": eph})

    out = result.write_oem("E1", tmp_path / "out.oem")

    assert out == tmp_path / "out.oem"
    assert out.exists()
    assert "CCSDS_OEM_VERS" in out.read_text(encoding="utf-8")


def test_write_oem_unknown_ephemeris_raises_keyerror(tmp_path: Path) -> None:
    result = Results(output_dir=tmp_path, log="")
    with pytest.raises(KeyError):
        result.write_oem("nope", tmp_path / "out.oem")


def test_write_oem_all_writes_one_file_per_ephemeris(tmp_path: Path) -> None:
    e1 = _write_eph(tmp_path / "E1.oem")
    e2 = _write_eph(tmp_path / "E2.oem")
    result = Results(
        output_dir=tmp_path,
        log="",
        ephemeris_paths={"EphemerisFile1": e1, "EphemerisFile2": e2},
    )

    dest = result.write_oem_all(tmp_path / "out")

    assert dest == tmp_path / "out"
    assert (dest / "EphemerisFile1.oem").is_file()
    assert (dest / "EphemerisFile2.oem").is_file()


def test_write_oem_all_with_no_ephemerides_creates_empty_dir(tmp_path: Path) -> None:
    result = Results(output_dir=tmp_path, log="")
    dest = result.write_oem_all(tmp_path / "empty")
    assert dest.is_dir()
    assert list(dest.iterdir()) == []


# --- __repr__ / _repr_html_ --------------------------------------------------


def test_repr_replaces_default_address_form(tmp_path: Path) -> None:
    result = _empty(tmp_path)
    assert "<gmat_run.results.Results object" not in repr(result)


def test_repr_format_shows_per_mapping_counts(tmp_path: Path) -> None:
    result = Results(
        output_dir=tmp_path,
        log="",
        report_paths={"RF1": tmp_path / "rf1.txt", "RF2": tmp_path / "rf2.txt"},
        ephemeris_paths={"Eph1": tmp_path / "e1.oem"},
    )
    assert repr(result) == "Results(reports=2, ephemerides=1, contacts=0, solver_runs=0)"


def test_repr_html_returns_table_listing_mapping_names(tmp_path: Path) -> None:
    result = Results(
        output_dir=tmp_path,
        log="",
        report_paths={"RF1": tmp_path / "rf1.txt"},
        ephemeris_paths={"Eph1": tmp_path / "e1.oem"},
    )
    html_str = result._repr_html_()
    assert "<table" in html_str
    assert "<code>reports</code>" in html_str
    assert "<code>ephemerides</code>" in html_str
    assert "<code>contacts</code>" in html_str
    assert "<code>solver_runs</code>" in html_str
    assert "RF1" in html_str
    assert "Eph1" in html_str
    assert "<em>none</em>" in html_str  # contacts and solver_runs are empty


def test_repr_html_does_not_materialise_dataframes(tmp_path: Path) -> None:
    # The HTML repr only sees keys; touching reports[name] would parse the
    # underlying file (which doesn't exist here) and raise. Verifying the
    # repr renders without I/O is the regression we care about: notebook
    # users get a useful overview of a Results without paying the parse cost.
    result = Results(
        output_dir=tmp_path,
        log="",
        report_paths={"RF1": tmp_path / "missing.txt"},
    )
    assert "RF1" in result._repr_html_()
