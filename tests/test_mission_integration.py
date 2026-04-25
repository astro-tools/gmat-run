"""Integration tests for :meth:`gmat_run.mission.Mission.run`.

Runs a minimal self-contained ``.script`` end-to-end against a real GMAT
install and asserts the round trip works: ``Mission.load`` → ``run`` →
``Results.reports[name]`` returns a DataFrame.

The fixture script is written into ``tmp_path`` rather than picked from
``samples/`` so the test is independent of which stock samples ship with a
given GMAT release. It exercises the full pipeline (install discovery,
gmatpy bootstrap, script load, output redirection, run, log capture, report
parsing) but does not assume any sample data files exist on disk.

Gated behind the ``integration`` pytest marker; CI runs it on Ubuntu and
Windows against a cached GMAT install.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gmat_run import Mission
from gmat_run.errors import GmatLoadError, GmatNotFoundError
from gmat_run.install import locate_gmat
from gmat_run.runtime import bootstrap

pytestmark = pytest.mark.integration


# A small self-contained mission: a circular LEO propagated for ten minutes
# with one ReportFile that lists the spacecraft state at each step. Avoids
# external data files (TLEs, OEMs, mass-properties tables) so the test runs
# on any GMAT install with the standard EOP/leap-second files in place.
_MINIMAL_SCRIPT = """\
Create Spacecraft Sat
Sat.DateFormat = UTCGregorian
Sat.Epoch = '01 Jan 2026 12:00:00.000'
Sat.CoordinateSystem = EarthMJ2000Eq
Sat.DisplayStateType = Keplerian
Sat.SMA = 7000
Sat.ECC = 0.001
Sat.INC = 28.5
Sat.RAAN = 75
Sat.AOP = 90
Sat.TA = 0

Create ForceModel FM
FM.CentralBody = Earth
FM.PrimaryBodies = {Earth}

Create Propagator Prop
Prop.FM = FM
Prop.Type = PrinceDormand78
Prop.InitialStepSize = 60
Prop.Accuracy = 1e-9

Create ReportFile RF
RF.Filename = 'leo_state.txt'
RF.Add = {Sat.UTCGregorian, Sat.X, Sat.Y, Sat.Z, Sat.SMA}
RF.WriteHeaders = True

BeginMissionSequence
Propagate Prop(Sat) {Sat.ElapsedSecs = 600}
"""


@pytest.fixture(scope="module")
def gmat_available() -> None:
    """Skip module if no usable GMAT install can be discovered and loaded.

    Both halves are checked: discovery (a structurally valid install on disk)
    and bootstrap (gmatpy importable in the current interpreter). The latter
    catches the cross-OS case where a Windows install is mounted into a Linux
    Python — discovery passes, but the .pyd cannot be loaded.

    Module-scoped so the costs aren't paid per test. ``bootstrap`` is
    one-shot-per-process anyway; calling it here primes the cache for the
    subsequent ``Mission.load`` calls.
    """
    try:
        install = locate_gmat()
    except GmatNotFoundError as exc:
        pytest.skip(f"no GMAT install discoverable: {exc}")
    try:
        bootstrap(install)
    except GmatLoadError as exc:
        pytest.skip(f"GMAT install discovered but gmatpy not loadable: {exc}")


@pytest.fixture
def minimal_script(tmp_path: Path) -> Path:
    """Write the minimal mission script into ``tmp_path`` and return its path."""
    script_path = tmp_path / "minimal_leo.script"
    script_path.write_text(_MINIMAL_SCRIPT, encoding="utf-8")
    return script_path


def test_minimal_mission_runs_end_to_end(
    gmat_available: None,
    minimal_script: Path,
) -> None:
    mission = Mission.load(minimal_script)
    result = mission.run()

    # The script declared one ReportFile resource named "RF".
    assert list(result.reports) == ["RF"]

    # Lazy-parse it. The DataFrame should have the expected columns and at
    # least one data row (a 600 s propagation at 60 s steps yields ~11 rows).
    df = result.reports["RF"]
    assert isinstance(df, pd.DataFrame)
    assert "Sat.UTCGregorian" in df.columns
    assert "Sat.SMA" in df.columns
    assert len(df) >= 2

    # Epoch column promoted to datetime64 by the parser.
    assert pd.api.types.is_datetime64_any_dtype(df["Sat.UTCGregorian"])

    # Log capture worked — GMAT writes at least the script-load banner.
    assert isinstance(result.log, str)
    assert len(result.log) > 0

    # The output dir holds the actual report file written by GMAT.
    assert (result.output_dir / "leo_state.txt").is_file()
