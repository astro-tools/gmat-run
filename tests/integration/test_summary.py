"""Integration test for :meth:`Mission.summary` against a real GMAT install.

Loads the ``Ex_LEOEphemeris`` fixture (already used by the OEM round-trip
tests) and asserts the snapshot covers the script's resources, output
resources, and command sequence as gmatpy reports them. Also locks in the
contract that both :class:`Mission` and :class:`Results` have non-default
``__repr__`` and ``_repr_html_`` methods — the issue's acceptance criterion
that the address-style repr is gone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gmat_run import Mission

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "fixtures" / "Ex_LEOEphemeris.script"


@pytest.fixture(scope="module")
def loaded_mission(gmat_available: None) -> Mission:
    return Mission.load(_FIXTURE)


def test_summary_lists_named_resources(loaded_mission: Mission) -> None:
    summary = loaded_mission.summary()
    assert summary.script_name == "Ex_LEOEphemeris.script"
    categories = {g.category: g.names for g in summary.resource_groups}
    # Sat is the only Spacecraft; FM and Prop are the only ForceModel and
    # Propagator. EF is the only EphemerisFile (and the only output).
    assert categories.get("Spacecraft") == ("Sat",)
    assert categories.get("ForceModel") == ("FM",)
    assert categories.get("Propagator") == ("Prop",)
    assert categories.get("EphemerisFile") == ("EF",)
    assert summary.spacecraft_count == 1


def test_output_resources_match_declared_files(loaded_mission: Mission) -> None:
    summary = loaded_mission.summary()
    output_categories = {g.category: g.names for g in summary.output_resources}
    assert output_categories == {"EphemerisFile": ("EF",)}


def test_command_sequence_contains_propagate(loaded_mission: Mission) -> None:
    summary = loaded_mission.summary()
    types = [c.type_name for c in summary.commands]
    # The script's mission sequence is a single ``Propagate``.
    assert "Propagate" in types
    assert summary.command_count >= 1


def test_mission_repr_is_not_default_address_form(loaded_mission: Mission) -> None:
    text = repr(loaded_mission)
    assert "<gmat_run.mission.Mission object" not in text
    assert text.startswith("Mission(")
    assert "spacecraft=1" in text


def test_mission_repr_html_renders_html_table(loaded_mission: Mission) -> None:
    html_str = loaded_mission._repr_html_()
    assert "<table" in html_str
    assert "Sat" in html_str  # the Spacecraft name shows up in the resource table


def test_results_repr_and_repr_html_are_not_default(loaded_mission: Mission) -> None:
    result = loaded_mission.run()
    try:
        text = repr(result)
        assert "<gmat_run.results.Results object" not in text
        assert text.startswith("Results(")
        # The fixture declares one EphemerisFile and no ReportFiles or
        # ContactLocators, so the per-mapping counts pin to a known shape.
        assert "ephemerides=1" in text
        assert "reports=0" in text
        assert "contacts=0" in text

        html_str = result._repr_html_()
        assert "<table" in html_str
        assert "<code>ephemerides</code>" in html_str
        assert "EF" in html_str  # resource name from the script
    finally:
        # Drop the Results so its TemporaryDirectory is cleaned up before the
        # module-scoped Mission fixture goes out of scope (Windows file
        # handles get cranky otherwise).
        del result
