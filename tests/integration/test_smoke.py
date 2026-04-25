"""Smoke test: a minimal self-contained mission round-trips end to end.

Runs an inline ``.script`` against a real GMAT install and asserts the full
pipeline (install discovery, gmatpy bootstrap, script load, output
redirection, run, log capture, report parsing) produces a sane DataFrame.
The fixture script is written into ``tmp_path`` rather than picked from
``samples/`` so the test is independent of which stock samples ship with a
given GMAT release. The stock-sample regression coverage lives in
:mod:`tests.integration.test_round_trip`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gmat_run import Mission

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
