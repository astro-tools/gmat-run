"""End-to-end test for :attr:`gmat_run.Mission.attitude_inputs`.

Loads ``Ex_AEMAttitude.script`` through a real GMAT install via
:meth:`gmat_run.Mission.load` and asserts that the lazy attitude-inputs
mapping discovers the Spacecraft's CCSDS-AEM file, resolves the script-
relative path, and parses it into the expected DataFrame shape.

No round-trip golden CSV here — GMAT does not *write* AEM (its
``EphemerisFile`` writer supports CCSDS-OEM, SPK, Code-500, and
STK-TimePosVel only), so there's no GMAT-emitted artefact to compare
against. The fixture file is the in-repo NASA/GSFC sample at
``tests/fixtures/aem_ephemeris/CCSDS_BasicQuatFile.aem``; correctness of
its contents is pinned by the parser-level golden tests in
``tests/test_parsers_aem_ephemeris_goldens.py``. This test exercises the
end-to-end *discovery* path: GMAT actually parsed the script, the
Spacecraft's Attitude/AttitudeFileName fields surfaced through gmatpy, and
the relative path resolved against the script directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gmat_run import Mission

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def aem_fixture() -> Path:
    return Path(__file__).parents[1] / "fixtures" / "aem_ephemeris" / "CCSDS_BasicQuatFile.aem"


def test_attitude_inputs_discovers_aem_file(
    gmat_available: None, fixtures_dir: Path, aem_fixture: Path
) -> None:
    script = fixtures_dir / "Ex_AEMAttitude.script"
    mission = Mission.load(script)

    assert list(mission.attitude_input_paths) == ["Sat"]
    resolved = mission.attitude_input_paths["Sat"].resolve()
    assert resolved == aem_fixture.resolve()


def test_attitude_inputs_parses_aem_file(gmat_available: None, fixtures_dir: Path) -> None:
    script = fixtures_dir / "Ex_AEMAttitude.script"
    mission = Mission.load(script)

    df = mission.attitude_inputs["Sat"]
    # Match the golden assertions on CCSDS_BasicQuatFile.aem (two-segment
    # NASA/GSFC quaternion sample). Loose checks here — the per-record
    # comparison lives in test_parsers_aem_ephemeris_goldens.py.
    assert list(df.columns) == ["Epoch", "Q1", "Q2", "Q3", "Q4"]
    assert df.attrs["attitude_type"] == "QUATERNION"
    assert df.attrs["quaternion_type"] == "LAST"
    assert df.attrs["object_name"] == "DefaultSC"
    assert len(df) == 124
