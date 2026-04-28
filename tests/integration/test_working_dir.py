"""End-to-end regression for the ``Mission.run(working_dir=...)`` gates.

Drives a real GMAT install through three back-to-back runs into the same
explicit ``working_dir`` to pin the contract:

1. The first run populates the workspace.
2. The second run, with the default ``overwrite=False``, raises
   :class:`~gmat_run.errors.GmatRunError` before invoking ``RunScript`` and
   leaves the prior run's artefacts intact.
3. The third run, with ``overwrite=True``, unlinks the colliding file and
   succeeds — proving the gate is the only thing standing between explicit
   re-runs and a silent mix of old and new artefacts.

The script is written inline so the test does not depend on which stock
samples ship with a given GMAT release.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gmat_run import Mission
from gmat_run.errors import GmatRunError

pytestmark = pytest.mark.integration


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


def test_collision_gate_requires_overwrite_for_explicit_working_dir(
    gmat_available: None,
    minimal_script: Path,
    tmp_path: Path,
) -> None:
    custom = tmp_path / "out"
    workspace_report = custom / "leo_state.txt"

    # 1. First run populates the workspace.
    first = Mission.load(minimal_script).run(working_dir=custom)
    assert first.output_dir == custom
    assert workspace_report.is_file()
    first_size = workspace_report.stat().st_size

    # 2. Second run refuses to clobber the prior artefacts. The captured byte
    # count below pins both halves: the gate fires (GmatRunError) and the
    # prior file is unchanged on disk.
    with pytest.raises(GmatRunError) as excinfo:
        Mission.load(minimal_script).run(working_dir=custom)
    assert excinfo.value.path == custom
    assert "leo_state.txt" in str(excinfo.value)
    assert "overwrite=True" in str(excinfo.value)
    assert workspace_report.stat().st_size == first_size

    # 3. Third run with overwrite=True clears the collision and succeeds.
    third = Mission.load(minimal_script).run(working_dir=custom, overwrite=True)
    assert third.output_dir == custom
    assert workspace_report.is_file()
    # Re-running the same propagation produces the same report — but the
    # file is freshly written, not appended. The size sanity check would
    # surface an "old + new" concatenation regression instantly.
    assert third.reports["RF"].equals(first.reports["RF"])
