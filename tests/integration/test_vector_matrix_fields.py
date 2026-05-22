"""Integration coverage for RVECTOR / RMATRIX dotted-path field access.

Issue #141: gmatpy's ``SetField`` has no matrix overload (only ``RealArray`` /
``StringArray`` / scalars), so an RMATRIX write of a nested list is rejected.
``Mission.__setitem__`` therefore routes RMATRIX writes through ``SetMatrix``
with a gmat ``Rmatrix`` (see ``Mission._build_rmatrix``). These tests pin that
round-trip — and the RVECTOR write path — against real gmatpy, which the
fake-backed unit tests in ``test_mission.py`` cannot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gmat_run import Mission

pytestmark = pytest.mark.integration

_SCRIPT = Path(__file__).parent / "fixtures" / "Ex_MinimalLEO.script"


def test_rmatrix_field_reads_as_nested_list(gmat_available: None) -> None:
    """``Sat.Covariance`` (RMATRIX_TYPE) reads back as a nested list of floats."""
    mission = Mission.load(_SCRIPT)
    cov = mission["Sat.Covariance"]
    assert isinstance(cov, list)
    assert len(cov) == 6
    assert all(isinstance(row, list) and len(row) == 6 for row in cov)
    assert all(isinstance(value, float) for row in cov for value in row)


def test_rmatrix_field_write_round_trips(gmat_available: None) -> None:
    """An RMATRIX field written through the dotted-path setter round-trips.

    Regression for issue #141: gmatpy ``SetField`` rejects a nested list, so
    ``Mission.__setitem__`` writes RMATRIX fields via ``SetMatrix``.
    """
    mission = Mission.load(_SCRIPT)
    target = [[float(i * 6 + j) for j in range(6)] for i in range(6)]
    mission["Sat.Covariance"] = target
    assert mission["Sat.Covariance"] == target


def test_rvector_field_write_is_accepted(gmat_available: None) -> None:
    """An RVECTOR field accepts a list write through the dotted-path setter.

    ``ReportFile.UpperLeft`` is RVECTOR_TYPE; a flat list maps to ``SetField``'s
    ``RealArray`` overload. Read-back is not asserted: the only RVECTOR fields
    in a headless mission are Subscriber GUI fields, which are unsized headless
    and raise on ``GetVector`` — so this confirms the write path only.
    """
    mission = Mission.load(_SCRIPT)
    mission["RF.UpperLeft"] = [0.1, 0.2]  # must not raise
