"""End-to-end regression for re-running one :class:`Mission` (issue #115).

Loading a ``Mission`` once and calling :meth:`Mission.run` more than once must
keep the runs independent: each run redirects its outputs into that run's own
workspace. The first run rewrites every relative subscriber ``Filename`` to an
absolute path in its workspace; the regression was that this rewrite survived
into the next run, where it was mistaken for a user-pinned destination and the
redirect was skipped — so the second run's report landed in the first run's
directory.

The script is written inline so the test does not depend on which stock
samples ship with a given GMAT release.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gmat_run import Mission

pytestmark = pytest.mark.integration


_SCRIPT = """\
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
def rerun_script(tmp_path: Path) -> Path:
    """Write the inline mission script into ``tmp_path`` and return its path."""
    script_path = tmp_path / "rerun_leo.script"
    script_path.write_text(_SCRIPT, encoding="utf-8")
    return script_path


def test_rerun_redirects_each_run_into_its_own_workspace(
    gmat_available: None,
    rerun_script: Path,
) -> None:
    mission = Mission.load(rerun_script)

    first = mission.run()
    second = mission.run()

    # Distinct temp workspaces — both stay alive while their Results are held.
    assert first.output_dir != second.output_dir

    first_report = first.reports._paths["RF"]  # type: ignore[attr-defined]
    second_report = second.reports._paths["RF"]  # type: ignore[attr-defined]

    # Each run's report landed under that run's own workspace, on disk —
    # before the fix the second run's report lived under first.output_dir.
    assert first_report.parent == first.output_dir
    assert first_report.is_file()
    assert second_report.parent == second.output_dir
    assert second_report.is_file()

    # Between runs the engine field reflects the script's declared value, not
    # a workspace path — that is what keeps the redirect working a second time.
    assert mission["RF.Filename"] == "leo_state.txt"
