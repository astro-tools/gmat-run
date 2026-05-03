"""Smoke test: a minimal self-contained mission round-trips end to end.

Runs ``tests/integration/fixtures/Ex_MinimalLEO.script`` (a circular LEO
propagated for ten minutes with one ReportFile) against a real GMAT install
and asserts the full pipeline (install discovery, gmatpy bootstrap, script
load, output redirection, run, log capture, report parsing) produces a sane
DataFrame. The fixture is hand-written rather than picked from ``samples/``
so the test is independent of which stock samples ship with a given GMAT
release; the stock-sample regression coverage lives in
:mod:`tests.integration.test_round_trip`.

The same fixture is reused by ``tests/canonical_image_smoke.py``, the
standalone runner driven by the ``canonical-image-smoke`` CI cell.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from gmat_run import Mission

pytestmark = pytest.mark.integration


_FIXTURE_SCRIPT = Path(__file__).parent / "fixtures" / "Ex_MinimalLEO.script"


@pytest.fixture
def minimal_script(tmp_path: Path) -> Path:
    """Copy the minimal mission fixture into ``tmp_path`` and return its path."""
    script_path = tmp_path / "minimal_leo.script"
    shutil.copyfile(_FIXTURE_SCRIPT, script_path)
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


def test_keplerian_field_override_after_load_runs_clean(
    gmat_available: None,
    minimal_script: Path,
) -> None:
    """Pre-run subscript writes against Keplerian fields actually take effect.

    Regression guard: prior to the ``gmat.Initialize()`` call inside
    ``Mission.load``, writing ``mission["Sat.SMA"]`` after load raised a
    spurious "ECC > 1" validation error from GMAT, even though the loaded
    spacecraft was clearly elliptical. The pattern is documented in
    ``docs/getting-started.md`` and demonstrated in
    ``docs/examples/02_parameter_sweep.ipynb``, so it has to survive engine
    upgrades.
    """
    new_sma = 7500.0

    mission = Mission.load(minimal_script)
    mission["Sat.SMA"] = new_sma
    result = mission.run()

    # Override survives the run — the report's first SMA row reflects the
    # written value, modulo numerical round-trip through Cartesian.
    df = result.reports["RF"]
    assert df["Sat.SMA"].iloc[0] == pytest.approx(new_sma, rel=1e-6)
