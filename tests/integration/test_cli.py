"""End-to-end smoke for the ``gmat-run`` CLI against a real GMAT install.

Mirrors :mod:`tests.integration.test_smoke` but drives the run via
:func:`gmat_run.cli.main` so the entry-point wiring is exercised. The
fixture script is identical in spirit (circular LEO, 10-minute propagation,
one ``ReportFile``) and is written into ``tmp_path`` so the test does not
depend on which stock samples ship with a given GMAT release.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gmat_run import cli

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
    script_path = tmp_path / "minimal_leo.script"
    script_path.write_text(_MINIMAL_SCRIPT, encoding="utf-8")
    return script_path


def test_cli_run_persists_outputs_end_to_end(
    gmat_available: None,
    minimal_script: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "out"

    code = cli.main(["run", str(minimal_script), "--out", str(out_dir)])

    assert code == cli.EXIT_OK

    # --out copied artefacts to the requested directory and updated the
    # summary's output-dir line accordingly.
    assert (out_dir / "leo_state.txt").is_file()
    out = capsys.readouterr().out
    assert f"Output directory: {out_dir}" in out
    assert "RF:" in out
    assert "rows" in out
