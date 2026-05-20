"""Integration tests for :attr:`Results.solver_runs` against real GMAT runs.

Drives the trimmed stock samples under ``tests/integration/fixtures/`` through
a full ``Mission.load`` -> ``run`` -> ``solver_runs`` pipeline:

* ``Ex_GEOTransfer.script`` — three ``Target`` blocks sharing one
  ``DifferentialCorrector``. The shared solver overwrites its ``.data`` file
  per block, so the surfaced DataFrame covers the last block only.
* the same script with ``DC.MaximumIterations`` forced to 1 — a deliberately
  non-converged run that must end ``"max_iter"`` rather than raise.
* ``Ex_Yukon_AlgebraicOptimization.script`` — a ``Yukon`` optimizer run.
* ``Ex_MinimalLEO.script`` — no solver at all, so the mapping stays empty.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from gmat_run import Mission

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).parent / "fixtures"


def _staged(fixture_name: str, tmp_path: Path) -> Path:
    """Copy a fixture script into ``tmp_path`` and return the copy's path."""
    dest = tmp_path / fixture_name
    shutil.copyfile(_FIXTURES / fixture_name, dest)
    return dest


def test_geotransfer_differential_corrector_converges(gmat_available: None, tmp_path: Path) -> None:
    mission = Mission.load(_staged("Ex_GEOTransfer.script", tmp_path))
    result = mission.run()

    assert list(result.solver_runs) == ["DC"]
    df = result.solver_runs["DC"]
    assert isinstance(df, pd.DataFrame)

    # Three Target blocks share DC and overwrite the same .data file; the
    # surfaced run is the last block ("Lower Apogee": vary MOI.Element1 to
    # achieve geoSat.Earth.SMA).
    assert df.attrs["solver_type"] == "DifferentialCorrector"
    assert "MOI.Element1" in df.columns
    for column in (
        "geoSat.Earth.SMA",
        "geoSat.Earth.SMA_desired",
        "geoSat.Earth.SMA_residual",
        "geoSat.Earth.SMA_tolerance",
    ):
        assert column in df.columns

    # It converged: the final row is stamped, the attr agrees, and the
    # convenience map agrees.
    assert df["status"].iloc[-1] == "converged"
    assert df.attrs["converged"] is True
    assert result.converged == {"DC": True}

    # The .data log was redirected into the run workspace, like every other
    # output — Results.solver_paths carries the resolved location.
    assert result.solver_paths["DC"].parent == result.output_dir
    # The rewrite is reverted once the run is over, so the engine field
    # reflects the loaded script again and a second run() redirects afresh
    # (issue #115).
    assert mission["DC.ReportFile"] == "DifferentialCorrectorDC1.data"


def test_geotransfer_max_iterations_one_ends_max_iter(gmat_available: None, tmp_path: Path) -> None:
    """A deliberately non-converged run ends ``"max_iter"`` and does not raise."""
    mission = Mission.load(_staged("Ex_GEOTransfer.script", tmp_path))
    mission["DC.MaximumIterations"] = 1
    result = mission.run()

    df = result.solver_runs["DC"]
    assert df["status"].iloc[-1] == "max_iter"
    assert df.attrs["converged"] is False
    assert result.converged["DC"] is False


def test_yukon_optimization_converges(gmat_available: None, tmp_path: Path) -> None:
    mission = Mission.load(_staged("Ex_Yukon_AlgebraicOptimization.script", tmp_path))
    result = mission.run()

    assert "Yukon1" in result.solver_runs
    df = result.solver_runs["Yukon1"]
    assert df.attrs["solver_type"] == "Yukon"
    # Yukon's normalised frame: variables, the cost function, per-constraint
    # residual — no goal quartet (the file carries none).
    assert "cost" in df.columns
    assert {"X1", "X2"}.issubset(df.columns)

    assert df["status"].iloc[-1] == "converged"
    assert result.converged["Yukon1"] is True


def test_mission_without_solver_yields_empty_mapping(gmat_available: None, tmp_path: Path) -> None:
    mission = Mission.load(_staged("Ex_MinimalLEO.script", tmp_path))
    result = mission.run()

    assert dict(result.solver_runs) == {}
    assert result.converged == {}
