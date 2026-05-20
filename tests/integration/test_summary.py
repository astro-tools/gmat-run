"""Integration test for :meth:`Mission.summary` against a real GMAT install.

Loads the ``Ex_LEOEphemeris`` fixture (already used by the OEM round-trip
tests) and asserts the snapshot covers the script's resources, output
resources, and command sequence as gmatpy reports them. Also locks in the
contract that both :class:`Mission` and :class:`Results` have non-default
``__repr__`` and ``_repr_html_`` methods — the issue's acceptance criterion
that the address-style repr is gone.

The ``Ex_GEOTransfer`` fixture additionally exercises a mission built from
``Target`` blocks: GMAT loops each ``EndTarget`` back to its owning ``Target``,
which used to send ``summary()`` into unbounded recursion and a SIGSEGV
(issue #114). These tests are the regression guard for that crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gmat_run import Mission

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "fixtures" / "Ex_LEOEphemeris.script"
_GEO_FIXTURE = Path(__file__).parent / "fixtures" / "Ex_GEOTransfer.script"


@pytest.fixture(scope="module")
def loaded_mission(gmat_available: None) -> Mission:
    return Mission.load(_FIXTURE)


@pytest.fixture(scope="module")
def geo_mission(gmat_available: None) -> Mission:
    """A mission whose sequence contains three ``Target`` branch blocks."""
    return Mission.load(_GEO_FIXTURE)


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


# --- branch-bearing mission (issue #114 regression) --------------------------


def test_summary_returns_on_branch_mission(geo_mission: Mission) -> None:
    # The core regression: summary() used to recurse through each Target's
    # EndTarget loopback and crash the process with SIGSEGV. Reaching this
    # assertion at all means the walk terminated.
    summary = geo_mission.summary()
    assert summary.script_name == "Ex_GEOTransfer.script"
    assert summary.command_count >= 1


def test_summary_outlines_top_level_branch_sequence(geo_mission: Mission) -> None:
    summary = geo_mission.summary()
    types = [c.type_name for c in summary.commands]
    # NoOp / BeginMissionSequence sentinels are dropped; three Target blocks
    # sit between the Propagate commands of the transfer.
    assert types == [
        "Propagate",
        "Target",
        "Propagate",
        "Propagate",
        "Target",
        "Target",
        "Propagate",
    ]


def test_target_block_children_are_summarised_one_level_deep(
    geo_mission: Mission,
) -> None:
    summary = geo_mission.summary()
    targets = [c for c in summary.commands if c.type_name == "Target"]
    assert len(targets) == 3
    # The 'Raise Apogee' Target body, with the EndTarget marker excluded.
    assert [c.type_name for c in targets[0].children] == [
        "Vary",
        "Maneuver",
        "Propagate",
        "Achieve",
    ]
    # The targeter bodies are flat — no commands nested below depth one.
    assert all(t.nested_count == 0 for t in targets)


def test_branch_mission_repr_and_repr_html_render(geo_mission: Mission) -> None:
    text = repr(geo_mission)
    assert text.startswith("Mission(")
    assert "<gmat_run.mission.Mission object" not in text
    html_str = geo_mission._repr_html_()
    assert "<table" in html_str
    assert "Target" in html_str
