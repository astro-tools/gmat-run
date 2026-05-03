"""Unit tests for :mod:`gmat_run.summary`.

The summary walker only touches a slice of the gmatpy surface —
``Moderator.GetListOfObjects``, ``GetObject``, the ``GmatBase`` contract
(``GetTypeName`` / ``IsOfType``), and the command-graph linked list
(``GetFirstCommand`` / ``GetNext`` / ``GetChildCommand`` / ``IsOfType``). The
minimal fakes below mirror exactly that surface, so they exercise the same
code paths as real gmatpy without dragging in the larger fake from
``test_mission.py``.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import Any

from gmat_run.summary import (
    CommandOutline,
    MissionSummary,
    ResourceGroup,
    build_mission_summary,
    format_results_html,
)

# --- minimal fakes ------------------------------------------------------------


class _FakeBase:
    """Stand-in for a configured GMAT object.

    The ``IsOfType`` set always includes the object's own ``type_name`` so a
    test that classifies via class name (``Spacecraft``, ``ReportFile``, ...)
    doesn't need to spell out the inheritance chain.
    """

    def __init__(
        self,
        type_name: str,
        name: str,
        *,
        is_of_type: tuple[str, ...] = (),
    ) -> None:
        self._type = type_name
        self._name = name
        self._is_of_type = set(is_of_type) | {type_name}

    def GetTypeName(self) -> str:
        return self._type

    def GetName(self) -> str:
        return self._name

    def IsOfType(self, type_name: str) -> bool:
        return type_name in self._is_of_type


class _FakeCommand:
    """Stand-in for a GmatCommand node in the mission sequence."""

    def __init__(
        self,
        type_name: str,
        *,
        generating: str = "",
        children: list[_FakeCommand] | None = None,
        is_branch: bool = False,
    ) -> None:
        self._type = type_name
        self._generating = generating
        self._children = children or []
        self._is_branch = is_branch
        self._next: _FakeCommand | None = None

    def GetTypeName(self) -> str:
        return self._type

    def GetGeneratingString(self) -> str:
        return self._generating

    def IsOfType(self, type_name: str) -> bool:
        if type_name == "BranchCommand":
            return self._is_branch
        return type_name == self._type

    def GetNext(self) -> _FakeCommand | None:
        return self._next

    def GetChildCommand(self, *_args: Any) -> _FakeCommand | None:
        return self._children[0] if self._children else None


def _link(*commands: _FakeCommand) -> _FakeCommand:
    """Wire siblings via ``GetNext`` and return the head."""
    for prev, nxt in pairwise(commands):
        prev._next = nxt
    return commands[0]


def _make_gmat(
    *,
    objects: dict[str, _FakeBase] | None = None,
    first_command: _FakeCommand | None = None,
) -> ModuleType:
    """Build a tiny fake gmat module with just the surface summary touches."""
    module = ModuleType("fake_gmat_summary")
    # Single enum is enough — the walker dedupes by name across enums, and
    # the fake's GetListOfObjects returns the same registry for any id.
    module.UNKNOWN_OBJECT = 1  # type: ignore[attr-defined]
    registry: dict[str, _FakeBase] = dict(objects or {})

    class _FakeModerator:
        def GetListOfObjects(self, _type_id: int) -> list[str]:
            return list(registry.keys())

        def GetFirstCommand(self) -> _FakeCommand | None:
            return first_command

    fake_moderator = _FakeModerator()

    class _ModeratorProxy:
        @staticmethod
        def Instance() -> _FakeModerator:
            return fake_moderator

    module.Moderator = _ModeratorProxy  # type: ignore[attr-defined]
    module.GetObject = lambda name: registry.get(name)  # type: ignore[attr-defined]
    return module


# --- empty mission ------------------------------------------------------------


def test_empty_mission_summary_has_no_resources_or_commands() -> None:
    summary = build_mission_summary(_make_gmat(), Path("empty.script"))
    assert summary.script_name == "empty.script"
    assert summary.resource_groups == ()
    assert summary.output_resources == ()
    assert summary.commands == ()
    assert summary.spacecraft_count == 0
    assert summary.command_count == 0


def test_summary_with_no_moderator_returns_empty() -> None:
    # A gmat module without a Moderator at all (extreme defensive case)
    # should still yield a usable MissionSummary.
    module = ModuleType("no_moderator_gmat")
    summary = build_mission_summary(module, Path("orphan.script"))
    assert summary.resource_groups == ()
    assert summary.commands == ()


# --- categorisation -----------------------------------------------------------


def test_named_resource_types_land_in_their_buckets() -> None:
    summary = build_mission_summary(
        _make_gmat(
            objects={
                "Sat": _FakeBase("Spacecraft", "Sat"),
                "FM": _FakeBase("ODEModel", "FM"),
                "Prop": _FakeBase("PropSetup", "Prop"),
                "EME": _FakeBase("CoordinateSystem", "EME"),
                "TOI": _FakeBase("ImpulsiveBurn", "TOI"),
                "Continuous": _FakeBase("FiniteBurn", "Continuous"),
                "RF": _FakeBase("ReportFile", "RF"),
                "Eph": _FakeBase("EphemerisFile", "Eph"),
                "Contact": _FakeBase("ContactLocator", "Contact"),
            }
        ),
        Path("everything.script"),
    )
    categories = {g.category: g.names for g in summary.resource_groups}
    assert categories == {
        "Spacecraft": ("Sat",),
        "ForceModel": ("FM",),
        "Propagator": ("Prop",),
        "CoordinateSystem": ("EME",),
        "ImpulsiveBurn": ("TOI",),
        "FiniteBurn": ("Continuous",),
        "ReportFile": ("RF",),
        "EphemerisFile": ("Eph",),
        "ContactLocator": ("Contact",),
    }
    assert summary.spacecraft_count == 1


def test_solver_subclass_classified_via_is_of_type() -> None:
    summary = build_mission_summary(
        _make_gmat(
            objects={
                "DC": _FakeBase("DifferentialCorrector", "DC", is_of_type=("Solver",)),
            }
        ),
        Path("dc.script"),
    )
    assert summary.resource_groups == (ResourceGroup(category="Solver", names=("DC",)),)


def test_subscriber_subclass_classified_via_is_of_type() -> None:
    # OrbitView / GroundTrackPlot / XYPlot are GUI-only Subscriber subclasses
    # not in the named bucket list. They should land under "Subscriber".
    summary = build_mission_summary(
        _make_gmat(
            objects={
                "OV": _FakeBase("OrbitView", "OV", is_of_type=("Subscriber",)),
                "GT": _FakeBase("GroundTrackPlot", "GT", is_of_type=("Subscriber",)),
            }
        ),
        Path("plots.script"),
    )
    assert summary.resource_groups == (ResourceGroup(category="Subscriber", names=("OV", "GT")),)


def test_unknown_type_lands_in_other_bucket() -> None:
    summary = build_mission_summary(
        _make_gmat(
            objects={"Tank": _FakeBase("FuelTank", "Tank")},
        ),
        Path("hardware.script"),
    )
    assert summary.resource_groups == (ResourceGroup(category="Other", names=("Tank",)),)


def test_resource_groups_follow_declared_order() -> None:
    # Display order is fixed by the summary module, not by registry order.
    summary = build_mission_summary(
        _make_gmat(
            objects={
                "RF": _FakeBase("ReportFile", "RF"),
                "Sat": _FakeBase("Spacecraft", "Sat"),
                "FM": _FakeBase("ODEModel", "FM"),
            }
        ),
        Path("order.script"),
    )
    categories = [g.category for g in summary.resource_groups]
    assert categories == ["Spacecraft", "ForceModel", "ReportFile"]


def test_categories_with_no_members_are_omitted() -> None:
    # Only Spacecraft is configured — no FiniteBurn group should appear.
    summary = build_mission_summary(
        _make_gmat(objects={"Sat": _FakeBase("Spacecraft", "Sat")}),
        Path("only-sat.script"),
    )
    assert [g.category for g in summary.resource_groups] == ["Spacecraft"]


# --- output resources ---------------------------------------------------------


def test_output_resources_only_lists_file_producing_categories() -> None:
    # Spacecraft / ForceModel are present but never appear under outputs.
    summary = build_mission_summary(
        _make_gmat(
            objects={
                "Sat": _FakeBase("Spacecraft", "Sat"),
                "FM": _FakeBase("ODEModel", "FM"),
                "RF": _FakeBase("ReportFile", "RF"),
                "Eph": _FakeBase("EphemerisFile", "Eph"),
            }
        ),
        Path("outputs.script"),
    )
    output_categories = [g.category for g in summary.output_resources]
    assert output_categories == ["ReportFile", "EphemerisFile"]


def test_output_resources_empty_when_no_file_producers_declared() -> None:
    summary = build_mission_summary(
        _make_gmat(objects={"Sat": _FakeBase("Spacecraft", "Sat")}),
        Path("no-output.script"),
    )
    assert summary.output_resources == ()


# --- command walk -------------------------------------------------------------


def test_command_walk_returns_top_level_commands_in_order() -> None:
    head = _link(
        _FakeCommand("BeginMissionSequence"),
        _FakeCommand("Propagate", generating="Propagate Prop(Sat) {Sat.ElapsedDays = 1};"),
        _FakeCommand("Maneuver", generating="Maneuver TOI(Sat);"),
    )
    summary = build_mission_summary(_make_gmat(first_command=head), Path("seq.script"))
    assert [c.type_name for c in summary.commands] == ["Propagate", "Maneuver"]
    assert summary.command_count == 2
    assert summary.commands[0].summary == "Propagate Prop(Sat) {Sat.ElapsedDays = 1};"


def test_command_walk_accepts_first_command_being_user_command() -> None:
    # Some fakes (and conceivably future GMAT changes) return the first user
    # command directly without a BeginMissionSequence sentinel head. The
    # walker should not skip it just because the head isn't NoOp.
    head = _link(
        _FakeCommand("Propagate", generating="Propagate Prop(Sat);"),
        _FakeCommand("Maneuver", generating="Maneuver TOI(Sat);"),
    )
    summary = build_mission_summary(_make_gmat(first_command=head), Path("seq.script"))
    assert [c.type_name for c in summary.commands] == ["Propagate", "Maneuver"]


def test_command_walk_skips_no_op_head() -> None:
    head = _link(
        _FakeCommand("NoOp"),
        _FakeCommand("Propagate", generating="Propagate Prop(Sat);"),
    )
    summary = build_mission_summary(_make_gmat(first_command=head), Path("seq.script"))
    assert [c.type_name for c in summary.commands] == ["Propagate"]


def test_branch_command_renders_children_one_level_deep() -> None:
    vary1 = _FakeCommand("Vary", generating="Vary DC(TOI.Element1 = 0.5);")
    vary2 = _FakeCommand("Vary", generating="Vary DC(TOI.Element2 = 0.5);")
    achieve = _FakeCommand("Achieve", generating="Achieve DC(Sat.SMA = 7100);")
    target = _FakeCommand(
        "Target",
        generating="Target DC;",
        children=[_link(vary1, vary2, achieve)],
        is_branch=True,
    )
    summary = build_mission_summary(
        _make_gmat(first_command=_link(_FakeCommand("BeginMissionSequence"), target)),
        Path("target.script"),
    )
    (cmd,) = summary.commands
    assert cmd.type_name == "Target"
    assert [c.type_name for c in cmd.children] == ["Vary", "Vary", "Achieve"]
    assert cmd.nested_count == 0


def test_nested_count_captures_descendants_below_depth_one() -> None:
    # Build: Target -> [Vary, Inner-Branch -> [Propagate, Maneuver]]
    # The Inner-Branch is at depth 1 (rendered as a child); the Propagate
    # and Maneuver inside it live at depth 2 and contribute to nested_count.
    inner = _FakeCommand(
        "If",
        generating="If Sat.SMA > 7000;",
        children=[
            _link(
                _FakeCommand("Propagate", generating="Propagate Prop(Sat);"),
                _FakeCommand("Maneuver", generating="Maneuver TOI(Sat);"),
            )
        ],
        is_branch=True,
    )
    vary = _FakeCommand("Vary", generating="Vary DC(TOI.Element1 = 0.5);")
    target = _FakeCommand(
        "Target",
        generating="Target DC;",
        children=[_link(vary, inner)],
        is_branch=True,
    )
    summary = build_mission_summary(
        _make_gmat(first_command=_link(_FakeCommand("BeginMissionSequence"), target)),
        Path("nested.script"),
    )
    (cmd,) = summary.commands
    assert [c.type_name for c in cmd.children] == ["Vary", "If"]
    # 2 commands inside If — both descendants of the depth-1 Inner-Branch,
    # so they are summarised by the count rather than the children tuple.
    assert cmd.nested_count == 2


def test_branch_with_no_children_renders_empty() -> None:
    # If GMAT exposes a branch command but it has no body, the walker should
    # still produce a CommandOutline with empty children rather than skip.
    branch = _FakeCommand("BeginScript", is_branch=True, children=[])
    summary = build_mission_summary(
        _make_gmat(first_command=_link(_FakeCommand("BeginMissionSequence"), branch)),
        Path("empty-branch.script"),
    )
    (cmd,) = summary.commands
    assert cmd.children == ()
    assert cmd.nested_count == 0


def test_command_summary_truncated_to_width() -> None:
    long_line = "Propagate Prop(Sat) {Sat.ElapsedDays = 1, " + ("X" * 200) + "};"
    head = _link(
        _FakeCommand("BeginMissionSequence"),
        _FakeCommand("Propagate", generating=long_line),
    )
    summary = build_mission_summary(_make_gmat(first_command=head), Path("long.script"))
    cmd = summary.commands[0]
    assert len(cmd.summary) <= 100
    assert cmd.summary.endswith("…")


def test_command_summary_keeps_only_first_non_blank_line() -> None:
    multi = "\n\nPropagate Prop(Sat);\nGMAT details on the next line\n"
    head = _link(
        _FakeCommand("BeginMissionSequence"),
        _FakeCommand("Propagate", generating=multi),
    )
    summary = build_mission_summary(_make_gmat(first_command=head), Path("multi.script"))
    assert summary.commands[0].summary == "Propagate Prop(Sat);"


def test_command_summary_blank_when_engine_returns_empty() -> None:
    head = _link(
        _FakeCommand("BeginMissionSequence"),
        _FakeCommand("Maneuver", generating=""),
    )
    summary = build_mission_summary(_make_gmat(first_command=head), Path("blank.script"))
    cmd = summary.commands[0]
    assert cmd.type_name == "Maneuver"
    assert cmd.summary == ""


# --- text repr ---------------------------------------------------------------


def test_text_repr_renders_resources_outputs_and_sequence() -> None:
    head = _link(
        _FakeCommand("BeginMissionSequence"),
        _FakeCommand("Propagate", generating="Propagate Prop(Sat) {Sat.ElapsedDays = 1};"),
    )
    summary = build_mission_summary(
        _make_gmat(
            objects={
                "Sat": _FakeBase("Spacecraft", "Sat"),
                "RF": _FakeBase("ReportFile", "RF"),
            },
            first_command=head,
        ),
        Path("flyby.script"),
    )
    text = repr(summary)
    assert "MissionSummary('flyby.script')" in text
    assert "Spacecraft (1): Sat" in text
    assert "ReportFile (1): RF" in text
    assert "Outputs" in text
    assert "Mission sequence (1 commands)" in text
    assert "1. Propagate — Propagate Prop(Sat) {Sat.ElapsedDays = 1};" in text


def test_text_repr_handles_empty_mission() -> None:
    summary = build_mission_summary(_make_gmat(), Path("empty.script"))
    text = repr(summary)
    assert "MissionSummary('empty.script')" in text
    assert "Mission sequence (0 commands)" in text
    assert "(empty)" in text


def test_text_repr_marks_nested_count_in_branch() -> None:
    inner = _FakeCommand(
        "If",
        generating="If Sat.SMA > 7000;",
        children=[
            _link(
                _FakeCommand("Propagate", generating="Propagate Prop(Sat);"),
            )
        ],
        is_branch=True,
    )
    target = _FakeCommand(
        "Target",
        generating="Target DC;",
        children=[_link(inner)],
        is_branch=True,
    )
    summary = build_mission_summary(
        _make_gmat(first_command=_link(_FakeCommand("BeginMissionSequence"), target)),
        Path("nested.script"),
    )
    text = repr(summary)
    assert "(1 nested commands)" in text


# --- HTML repr ---------------------------------------------------------------


def test_html_repr_includes_resource_table_and_sequence() -> None:
    head = _link(
        _FakeCommand("BeginMissionSequence"),
        _FakeCommand("Propagate", generating="Propagate Prop(Sat);"),
    )
    summary = build_mission_summary(
        _make_gmat(
            objects={"Sat": _FakeBase("Spacecraft", "Sat")},
            first_command=head,
        ),
        Path("flyby.script"),
    )
    html_str = summary._repr_html_()
    assert "<table" in html_str
    assert "<code>flyby.script</code>" in html_str
    assert "Spacecraft" in html_str
    assert "<ol>" in html_str
    assert "Propagate" in html_str


def test_html_repr_escapes_resource_names() -> None:
    summary = build_mission_summary(
        _make_gmat(objects={"<bad>": _FakeBase("Spacecraft", "<bad>")}),
        Path("safe.script"),
    )
    html_str = summary._repr_html_()
    assert "<bad>" not in html_str.replace("&lt;bad&gt;", "")
    assert "&lt;bad&gt;" in html_str


def test_html_repr_for_empty_mission_marks_sequence_empty() -> None:
    summary = build_mission_summary(_make_gmat(), Path("empty.script"))
    html_str = summary._repr_html_()
    assert "(empty)" in html_str
    assert "0 commands" in html_str


def test_html_repr_renders_branch_children_and_nested_count() -> None:
    inner = _FakeCommand(
        "If",
        generating="If Sat.SMA > 7000;",
        children=[_link(_FakeCommand("Propagate", generating="Propagate Prop(Sat);"))],
        is_branch=True,
    )
    target = _FakeCommand(
        "Target",
        generating="Target DC;",
        children=[_link(_FakeCommand("Vary", generating="Vary DC;"), inner)],
        is_branch=True,
    )
    summary = build_mission_summary(
        _make_gmat(first_command=_link(_FakeCommand("BeginMissionSequence"), target)),
        Path("nested.script"),
    )
    html_str = summary._repr_html_()
    # Inner <ul> for the depth-1 children, plus the nested-count notice for
    # the descendants past depth 1.
    assert "<ul>" in html_str
    assert "Vary" in html_str
    assert "(1 nested commands)" in html_str


def test_html_repr_falls_back_to_type_only_when_summary_blank() -> None:
    # GMAT command without a generating string renders as just the type name —
    # the HTML/text formatters drop the dash separator entirely.
    head = _link(
        _FakeCommand("BeginMissionSequence"),
        _FakeCommand("Maneuver", generating=""),
    )
    summary = build_mission_summary(_make_gmat(first_command=head), Path("blank.script"))
    html_str = summary._repr_html_()
    text = repr(summary)
    assert "Maneuver" in html_str
    assert " — " not in html_str.split("<ol>", 1)[1].split("</ol>", 1)[0]
    assert "1. Maneuver" in text


# --- format_results_html -----------------------------------------------------


def test_format_results_html_lists_each_mapping() -> None:
    out = format_results_html(
        report_names=("RF1", "RF2"),
        ephemeris_names=("Eph1",),
        contact_names=(),
    )
    assert "<code>reports</code>" in out
    assert "<code>ephemerides</code>" in out
    assert "<code>contacts</code>" in out
    assert "RF1, RF2" in out
    assert "Eph1" in out
    assert "<em>none</em>" in out


def test_format_results_html_escapes_names() -> None:
    out = format_results_html(
        report_names=("<RF>",),
        ephemeris_names=(),
        contact_names=(),
    )
    assert "<RF>" not in out.replace("&lt;RF&gt;", "")
    assert "&lt;RF&gt;" in out


def test_format_results_html_when_all_mappings_empty() -> None:
    out = format_results_html(
        report_names=(),
        ephemeris_names=(),
        contact_names=(),
    )
    assert out.count("<em>none</em>") == 3


# --- dataclass shape sanity --------------------------------------------------


def test_resource_group_is_frozen() -> None:
    group = ResourceGroup(category="Spacecraft", names=("Sat",))
    try:
        group.category = "ForceModel"  # type: ignore[misc]
    except Exception:  # frozen dataclass raises FrozenInstanceError
        pass
    else:
        raise AssertionError("expected ResourceGroup to reject mutation")


def test_command_outline_defaults() -> None:
    outline = CommandOutline(type_name="Propagate", summary="Propagate Prop(Sat);")
    assert outline.children == ()
    assert outline.nested_count == 0


def test_mission_summary_is_a_dataclass_record() -> None:
    summary = MissionSummary(
        script_name="x.script",
        resource_groups=(),
        output_resources=(),
        commands=(),
    )
    assert summary.script_name == "x.script"
    assert summary.spacecraft_count == 0
    assert summary.command_count == 0
