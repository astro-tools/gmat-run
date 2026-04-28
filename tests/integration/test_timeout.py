"""End-to-end tests for ``Mission.run(timeout=...)`` against a real GMAT install.

These exercise the real subprocess path: parent spawns a child via
``-m gmat_run.cli _internal-run``, which boots gmatpy, loads the script,
applies overrides, and runs. The driver and the child handler are
unit-tested in :mod:`tests.test_subprocess`; this suite covers the path
nothing else can — the full Popen → child → gmatpy → kill round-trip.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gmat_run import Mission
from gmat_run.errors import GmatTimeoutError

pytestmark = pytest.mark.integration


# A minimal mission that propagates for 1e9 seconds (~31 years). Even the
# fastest integrator step rate cannot finish this in any wall-clock window
# the test cares about, so a parent-side timeout will always fire first.
# Same shape as ``tests.integration.test_smoke._MINIMAL_SCRIPT`` so any
# discovery / bootstrap wiring shared between the two stays exercised.
_HANGING_SCRIPT = """\
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
Prop.InitialStepSize = 1
Prop.Accuracy = 1e-12

Create ReportFile RF
RF.Filename = 'hanging.txt'
RF.Add = {Sat.UTCGregorian, Sat.X}

BeginMissionSequence
Propagate Prop(Sat) {Sat.ElapsedSecs = 1e9}
"""


@pytest.fixture
def hanging_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "hanging.script"
    script_path.write_text(_HANGING_SCRIPT, encoding="utf-8")
    return script_path


def test_timeout_raises_gmat_timeout_error_within_wall_clock_window(
    gmat_available: None,
    hanging_script: Path,
) -> None:
    """A 2 s timeout on a 1e9 s propagation raises within ~5 s wall-clock.

    Generous upper bound because Windows / macOS CI runners cold-start the
    gmatpy bootstrap inside the child and that adds ~1-2 s on top of the
    timeout itself. The 5 s ceiling still catches a runaway parent that
    failed to kill the child.
    """
    mission = Mission.load(hanging_script)
    started = time.monotonic()
    with pytest.raises(GmatTimeoutError) as excinfo:
        mission.run(timeout=2.0)
    elapsed = time.monotonic() - started

    assert excinfo.value.requested_timeout == 2.0
    # The exception's reported elapsed should match the parent's measurement
    # to within a small tolerance — both are wall-clock from the same start.
    assert excinfo.value.elapsed == pytest.approx(elapsed, abs=1.0)
    # Hard ceiling — if this trips, the kill ladder isn't reaping the child.
    assert elapsed < 5.0


def test_timeout_subprocess_path_runs_finite_mission_to_completion(
    gmat_available: None,
    tmp_path: Path,
) -> None:
    """A finite mission with ``timeout=`` returns a populated :class:`Results`.

    Cross-check that the subprocess path is not just for hanging scripts —
    a normal mission completes inside its cap and yields a Results that
    looks like the in-process run's. The DataFrames-byte-equal comparison
    against the in-process path is covered by the broader regression suite;
    here we just confirm round-trip.
    """
    import pandas as pd

    script_path = tmp_path / "finite.script"
    script_path.write_text(
        # 600 s propagation — same as test_smoke's _MINIMAL_SCRIPT.
        _HANGING_SCRIPT.replace("Sat.ElapsedSecs = 1e9", "Sat.ElapsedSecs = 600")
        .replace("Prop.InitialStepSize = 1", "Prop.InitialStepSize = 60")
        .replace("hanging.txt", "finite.txt"),
        encoding="utf-8",
    )

    mission = Mission.load(script_path)
    result = mission.run(timeout=60.0)

    assert list(result.reports) == ["RF"]
    df = result.reports["RF"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 2
    assert "Sat.UTCGregorian" in df.columns
    assert isinstance(result.log, str)
    assert len(result.log) > 0
