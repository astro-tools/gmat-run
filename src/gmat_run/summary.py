"""Read-only snapshot of a loaded :class:`~gmat_run.mission.Mission` graph.

Backs :meth:`gmat_run.mission.Mission.summary` and the notebook-friendly
``__repr__`` / ``_repr_html_`` on both :class:`~gmat_run.mission.Mission` and
:class:`~gmat_run.results.Results`.

The walkers here only call gmatpy's public interface — ``Moderator`` and the
``GmatBase`` / ``GmatCommand`` contracts — so a fake gmat module that mirrors
those (like the one in ``tests/test_mission.py``) drives the same code path
as real gmatpy.

The snapshot is name-only: resource names, command type names, and a
truncated first-line summary per command. It does not materialise field
values — ``mission["Sat.SMA"]`` already serves that need.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

__all__ = [
    "CommandOutline",
    "MissionSummary",
    "ResourceGroup",
    "build_mission_summary",
    "format_results_html",
]


# --- categorisation -----------------------------------------------------------

# Display order for resource categories in :class:`MissionSummary`. The named
# buckets come from the issue; "Other" catches anything reachable from the
# configured-object enumeration that doesn't match a named type.
_RESOURCE_CATEGORY_ORDER: Final = (
    "Spacecraft",
    "ForceModel",
    "Propagator",
    "CoordinateSystem",
    "ImpulsiveBurn",
    "FiniteBurn",
    "ReportFile",
    "EphemerisFile",
    "ContactLocator",
    "Solver",
    "Subscriber",
    "Other",
)

# Subset that produces files at run time. Ordered to match the keys on
# :class:`gmat_run.results.Results` (``reports`` / ``ephemerides`` /
# ``contacts``).
_OUTPUT_CATEGORY_ORDER: Final = (
    "ReportFile",
    "EphemerisFile",
    "ContactLocator",
)

# GMAT ``GetTypeName()`` -> display category. ``PropSetup`` is what scripts
# call ``Propagator``, and ``ODEModel`` is what scripts call ``ForceModel``
# (see the class hierarchy in ``.claude/skills/gmat-python/references/objects.md``
# under ``src/.repo-template`` for the cross-reference). Both legacy spellings
# are accepted in case a future GMAT release exposes the script-level name
# directly.
_TYPE_NAME_TO_CATEGORY: Final = {
    "Spacecraft": "Spacecraft",
    "ODEModel": "ForceModel",
    "ForceModel": "ForceModel",
    "PropSetup": "Propagator",
    "Propagator": "Propagator",
    "CoordinateSystem": "CoordinateSystem",
    "ImpulsiveBurn": "ImpulsiveBurn",
    "FiniteBurn": "FiniteBurn",
    "ReportFile": "ReportFile",
    "EphemerisFile": "EphemerisFile",
    "ContactLocator": "ContactLocator",
}

# Enum attribute names probed on the gmat module to enumerate configured
# objects. ``UNKNOWN_OBJECT`` is the union over every configured type when
# present; the named buckets are walked in addition so a fake gmat module
# that only exposes a subset of enums still finds its resources. Each name is
# resolved with ``getattr(..., None)`` and tolerated when missing.
_RESOURCE_ENUM_ATTRS: Final = (
    "UNKNOWN_OBJECT",
    "SPACECRAFT",
    "FORCE_MODEL",
    "PROP_SETUP",
    "COORDINATE_SYSTEM",
    "BURN",
    "SUBSCRIBER",
    "EVENT_LOCATOR",
    "SOLVER",
)

# Maximum width for a single command's first-line summary. GMAT's
# ``GetGeneratingString`` for a Propagate can run hundreds of characters;
# truncating keeps notebook reprs readable without dropping the leading text.
_COMMAND_SUMMARY_WIDTH: Final = 100

# Backstops for the command-tree walk. GMAT wires every ``BranchEnd``'s
# ``GetNext()`` back to the owning branch command, so a naive walk cycles
# forever. The walkers below break that cycle at the ``BranchEnd``; these caps
# only matter if a branch command ever fails to report its ``BranchEnd`` (an
# exotic plugin), in which case the summary degrades to a truncated outline
# instead of a native stack overflow. Real missions nest a handful of levels
# deep and no branch command exposes more than a couple of branches.
_MAX_BRANCH_DEPTH: Final = 64
_MAX_BRANCH_COUNT: Final = 64


# --- dataclasses --------------------------------------------------------------


@dataclass(frozen=True)
class ResourceGroup:
    """One resource category and the resource names that fall under it.

    Names appear in the order GMAT returns them from
    ``Moderator.GetListOfObjects`` for the matching enum, which corresponds to
    declaration order in the loaded ``.script``.
    """

    category: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class CommandOutline:
    """One node in the mission-sequence outline.

    ``children`` is the depth-1 expansion of a branch command (``If``,
    ``For``, ``While``, ``Target``, ``Optimize``, ``BeginScript``, ...).
    Deeper nesting is not materialised; ``nested_count`` is the number of
    descendants past the first nested level so the caller can tell that, e.g.,
    a ``Target`` with three ``Vary`` children each containing a ``Propagate``
    has 3 nested commands beyond what ``children`` shows.

    ``summary`` is the first non-blank line of the command's
    ``GetGeneratingString`` output, truncated. It is ``""`` when the engine
    returns nothing useful — some plugin commands simply don't implement that
    method.
    """

    type_name: str
    summary: str
    children: tuple[CommandOutline, ...] = ()
    nested_count: int = 0


@dataclass(frozen=True)
class MissionSummary:
    """Read-only snapshot of a loaded mission's structure.

    Returned by :meth:`gmat_run.mission.Mission.summary`. Backs both
    :meth:`gmat_run.mission.Mission.__repr__` (one-line text) and
    :meth:`gmat_run.mission.Mission._repr_html_` (notebook table).

    Attributes:
        script_name: ``Path.name`` of the loaded script.
        resource_groups: One :class:`ResourceGroup` per non-empty category,
            in :data:`_RESOURCE_CATEGORY_ORDER`.
        output_resources: Subset of ``resource_groups`` covering only the
            categories that produce files at run time
            (``ReportFile`` / ``EphemerisFile`` / ``ContactLocator``).
        commands: Top-level mission-sequence commands in declaration order.
    """

    script_name: str
    resource_groups: tuple[ResourceGroup, ...]
    output_resources: tuple[ResourceGroup, ...]
    commands: tuple[CommandOutline, ...]

    @property
    def spacecraft_count(self) -> int:
        """Number of Spacecraft resources in the loaded script."""
        for group in self.resource_groups:
            if group.category == "Spacecraft":
                return len(group.names)
        return 0

    @property
    def command_count(self) -> int:
        """Number of top-level commands in the mission sequence."""
        return len(self.commands)

    def __repr__(self) -> str:
        lines: list[str] = [f"MissionSummary({self.script_name!r})"]
        if self.resource_groups:
            lines.append("")
            lines.append("Resources")
            for group in self.resource_groups:
                names = ", ".join(group.names)
                lines.append(f"  {group.category} ({len(group.names)}): {names}")
        if self.output_resources:
            lines.append("")
            lines.append("Outputs")
            for group in self.output_resources:
                names = ", ".join(group.names)
                lines.append(f"  {group.category}: {names}")
        lines.append("")
        lines.append(f"Mission sequence ({_count(self.command_count, 'command')})")
        if not self.commands:
            lines.append("  (empty)")
        else:
            for idx, cmd in enumerate(self.commands, start=1):
                lines.append(f"  {idx}. {_format_command_line(cmd)}")
                for sub in cmd.children:
                    lines.append(f"      - {_format_command_line(sub)}")
                if cmd.nested_count:
                    lines.append(f"      ({_count(cmd.nested_count, 'nested command')})")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return _format_mission_html(self)


# --- public walkers -----------------------------------------------------------


def build_mission_summary(gmat: ModuleType, script_path: Path) -> MissionSummary:
    """Build a :class:`MissionSummary` snapshot from a loaded gmatpy graph.

    Args:
        gmat: The bootstrapped gmatpy module (or a fake mirroring its
            ``Moderator`` / ``GmatBase`` / ``GmatCommand`` surface).
        script_path: Source script path; only ``script_path.name`` is used in
            the snapshot.
    """
    by_category = _walk_resources(gmat)
    resource_groups = tuple(
        ResourceGroup(category=cat, names=tuple(by_category[cat]))
        for cat in _RESOURCE_CATEGORY_ORDER
        if by_category.get(cat)
    )
    output_resources = tuple(
        ResourceGroup(category=cat, names=tuple(by_category[cat]))
        for cat in _OUTPUT_CATEGORY_ORDER
        if by_category.get(cat)
    )
    commands = tuple(_walk_commands(gmat))
    return MissionSummary(
        script_name=script_path.name,
        resource_groups=resource_groups,
        output_resources=output_resources,
        commands=commands,
    )


def format_results_html(
    *,
    report_names: Iterable[str],
    ephemeris_names: Iterable[str],
    contact_names: Iterable[str],
    solver_run_names: Iterable[str],
) -> str:
    """Render a small HTML table for :meth:`gmat_run.results.Results._repr_html_`.

    Lists names per output mapping. No DataFrames are materialised; this only
    sees the keys.
    """
    rows: list[tuple[str, tuple[str, ...]]] = [
        ("reports", tuple(report_names)),
        ("ephemerides", tuple(ephemeris_names)),
        ("contacts", tuple(contact_names)),
        ("solver_runs", tuple(solver_run_names)),
    ]
    parts: list[str] = ['<div class="gmat-run-results">']
    parts.append("<strong>Results</strong>")
    parts.append("<table>")
    parts.append(
        "<thead><tr>"
        '<th style="text-align:left">Output</th>'
        '<th style="text-align:right">Count</th>'
        '<th style="text-align:left">Names</th>'
        "</tr></thead>"
    )
    parts.append("<tbody>")
    for label, names in rows:
        cell = ", ".join(html.escape(n) for n in names) if names else "<em>none</em>"
        parts.append(
            f"<tr>"
            f'<td style="text-align:left"><code>{label}</code></td>'
            f'<td style="text-align:right">{len(names)}</td>'
            f'<td style="text-align:left">{cell}</td>'
            f"</tr>"
        )
    parts.append("</tbody></table>")
    parts.append("</div>")
    return "".join(parts)


# --- resource walk ------------------------------------------------------------


def _walk_resources(gmat: ModuleType) -> dict[str, list[str]]:
    """Enumerate configured objects and bucket them by display category.

    Walks every enum in :data:`_RESOURCE_ENUM_ATTRS` the gmat module exposes,
    dedupes names across enums (since ``UNKNOWN_OBJECT`` overlaps the named
    buckets), classifies each via :func:`_categorize`, and returns a dict
    keyed by category. Categories with zero entries are absent from the
    result.
    """
    moderator_proxy = getattr(gmat, "Moderator", None)
    if moderator_proxy is None:
        return {}
    moderator = moderator_proxy.Instance()
    seen: set[str] = set()
    by_category: dict[str, list[str]] = {}
    for attr in _RESOURCE_ENUM_ATTRS:
        type_id = getattr(gmat, attr, None)
        if type_id is None:
            continue
        try:
            names = list(moderator.GetListOfObjects(type_id))
        except Exception:
            continue
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            obj = _safe_get_object(gmat, name)
            if obj is None:
                continue
            category = _categorize(obj)
            by_category.setdefault(category, []).append(name)
    return by_category


def _categorize(obj: Any) -> str:
    """Return the display category for one configured object."""
    type_name = _safe_type_name(obj)
    if type_name in _TYPE_NAME_TO_CATEGORY:
        return _TYPE_NAME_TO_CATEGORY[type_name]
    if _is_of_type(obj, "Solver"):
        return "Solver"
    if _is_of_type(obj, "Subscriber"):
        return "Subscriber"
    return "Other"


# --- command walk -------------------------------------------------------------


def _walk_commands(gmat: ModuleType) -> list[CommandOutline]:
    """Walk the loaded mission command sequence one level deep.

    Returns an empty list when the gmat module exposes no ``Moderator`` or
    when ``GetFirstCommand`` raises (fakes that don't model the command
    graph). Real GMAT heads every sequence with a ``NoOp -> BeginMissionSequence``
    sentinel prefix — every leading ``NoOp`` / ``BeginMissionSequence`` node is
    skipped, since neither is a user command.
    """
    moderator_proxy = getattr(gmat, "Moderator", None)
    if moderator_proxy is None:
        return []
    try:
        first = moderator_proxy.Instance().GetFirstCommand()
    except Exception:
        return []
    if first is None:
        return []
    node: Any = first
    while node is not None and _safe_type_name(node) in {"NoOp", "BeginMissionSequence"}:
        node = _safe_next(node)
    outlines: list[CommandOutline] = []
    while node is not None:
        outlines.append(_outline_command(node, depth=0))
        node = _safe_next(node)
    return outlines


def _outline_command(node: Any, *, depth: int) -> CommandOutline:
    """Build a :class:`CommandOutline` for a single command node.

    ``depth=0`` materialises the children of a branch command. ``depth>=1``
    short-circuits child materialisation so deeper nesting is summarised by
    :attr:`CommandOutline.nested_count` on the depth-0 ancestor.
    """
    type_name = _safe_type_name(node) or "Command"
    summary = _command_summary(node)
    children: tuple[CommandOutline, ...] = ()
    nested_count = 0
    if depth == 0 and _is_branch(node):
        child_outlines: list[CommandOutline] = []
        nested = 0
        for child in _iter_branch_children(node):
            child_outlines.append(_outline_command(child, depth=1))
            nested += _count_descendants(child, depth=1)
        children = tuple(child_outlines)
        nested_count = nested
    return CommandOutline(
        type_name=type_name,
        summary=summary,
        children=children,
        nested_count=nested_count,
    )


def _count_descendants(node: Any, depth: int) -> int:
    """Total commands nested under ``node`` (depth >= 2 from the top level).

    Recurses through branch bodies via :func:`_iter_branch_children`, which
    stops at each ``BranchEnd``. ``depth`` is bounded by
    :data:`_MAX_BRANCH_DEPTH` purely as a backstop: if a branch command ever
    fails to expose its ``BranchEnd`` the count is silently truncated rather
    than overflowing the native stack (issue #114).
    """
    if depth >= _MAX_BRANCH_DEPTH or not _is_branch(node):
        return 0
    count = 0
    for child in _iter_branch_children(node):
        count += 1
        count += _count_descendants(child, depth + 1)
    return count


def _iter_branch_children(node: Any) -> Iterator[Any]:
    """Yield the body commands of every branch of a branch command.

    GMAT terminates each branch with a ``BranchEnd`` marker (``EndTarget`` /
    ``EndIf`` / ...) whose ``GetNext()`` points *back* at the owning branch
    command. The walk therefore stops at the first ``BranchEnd`` — that node is
    neither yielded nor followed — which keeps it inside the branch instead of
    looping back and running off into the rest of the mission sequence
    (issue #114).

    Every branch is enumerated: ``GetChildCommand(0)``, ``GetChildCommand(1)``,
    ... until the engine returns ``None``, so an ``If``/``Else`` contributes
    both arms.

    There is deliberately no ``id()``-based cycle guard. gmatpy hands out a
    fresh SWIG proxy per call, so proxy identity tracks neither the underlying
    C++ command (false negatives) nor — once a proxy is freed and CPython
    recycles its address — distinct commands (false positives that silently
    truncate a valid walk). The ``BranchEnd`` stop here and the depth cap in
    :func:`_count_descendants` bound the walk without relying on identity;
    following ``GetNext()`` alone always terminates at the end of the sequence.
    """
    for index in range(_MAX_BRANCH_COUNT):
        child = _safe_child_command(node, index)
        if child is None:
            break
        while child is not None and not _is_branch_end(child):
            yield child
            child = _safe_next(child)


def _command_summary(node: Any) -> str:
    """First non-blank line of ``GetGeneratingString``, truncated.

    Returns ``""`` when the engine returns nothing or raises — the type name
    in :attr:`CommandOutline.type_name` carries the essential information.
    """
    try:
        text = str(node.GetGeneratingString())
    except Exception:
        return ""
    if not text:
        return ""
    line = next((s.strip() for s in text.splitlines() if s.strip()), "")
    if len(line) > _COMMAND_SUMMARY_WIDTH:
        line = line[: _COMMAND_SUMMARY_WIDTH - 1].rstrip() + "…"
    return line


def _is_branch(node: Any) -> bool:
    """True if the command node has nested children (``If`` / ``For`` / …)."""
    try:
        return bool(node.IsOfType("BranchCommand"))
    except Exception:
        return False


def _is_branch_end(node: Any) -> bool:
    """True if ``node`` is a branch terminator (``EndTarget`` / ``EndIf`` / …).

    GMAT wires every ``BranchEnd``'s ``GetNext()`` back to the owning branch
    command, so a child walk must stop at one rather than follow it — see
    :func:`_iter_branch_children`.
    """
    try:
        return bool(node.IsOfType("BranchEnd"))
    except Exception:
        return False


def _safe_child_command(node: Any, index: int) -> Any:
    """``GetChildCommand(index)`` tolerant of fakes and plugin commands.

    Real gmatpy exposes ``GetChildCommand(Integer whichOne=0)`` and returns
    ``None`` for an out-of-range index — that is how
    :func:`_iter_branch_children` discovers a branch command's branch count.
    The nullary fallback covers a hypothetical binding that omits the
    parameter entirely.
    """
    try:
        return node.GetChildCommand(index)
    except TypeError:
        if index != 0:
            return None
        try:
            return node.GetChildCommand()
        except Exception:
            return None
    except Exception:
        return None


def _safe_next(node: Any) -> Any:
    """``GetNext()`` tolerant of fakes and plugin commands that raise."""
    try:
        return node.GetNext()
    except Exception:
        return None


# --- shared object helpers ----------------------------------------------------


def _safe_get_object(gmat: ModuleType, name: str) -> Any:
    try:
        return gmat.GetObject(name)
    except Exception:
        return None


def _safe_type_name(obj: Any) -> str:
    try:
        return str(obj.GetTypeName())
    except Exception:
        return ""


def _is_of_type(obj: Any, type_name: str) -> bool:
    try:
        return bool(obj.IsOfType(type_name))
    except Exception:
        return False


# --- HTML rendering -----------------------------------------------------------


def _format_mission_html(summary: MissionSummary) -> str:
    """Render the notebook HTML for a :class:`MissionSummary`."""
    parts: list[str] = ['<div class="gmat-run-mission-summary">']
    parts.append(f"<strong>Mission:</strong> <code>{html.escape(summary.script_name)}</code>")

    if summary.resource_groups:
        parts.append('<table class="gmat-run-resources">')
        parts.append(
            "<thead><tr>"
            '<th style="text-align:left">Resource type</th>'
            '<th style="text-align:right">Count</th>'
            '<th style="text-align:left">Names</th>'
            "</tr></thead>"
        )
        parts.append("<tbody>")
        for group in summary.resource_groups:
            names_html = ", ".join(html.escape(n) for n in group.names)
            parts.append(
                f"<tr>"
                f'<td style="text-align:left">{html.escape(group.category)}</td>'
                f'<td style="text-align:right">{len(group.names)}</td>'
                f'<td style="text-align:left">{names_html}</td>'
                f"</tr>"
            )
        parts.append("</tbody></table>")

    if summary.output_resources:
        parts.append("<strong>Outputs</strong>")
        parts.append('<table class="gmat-run-outputs">')
        parts.append(
            "<thead><tr>"
            '<th style="text-align:left">Resource type</th>'
            '<th style="text-align:left">Names</th>'
            "</tr></thead>"
        )
        parts.append("<tbody>")
        for group in summary.output_resources:
            names_html = ", ".join(html.escape(n) for n in group.names)
            parts.append(
                f"<tr>"
                f'<td style="text-align:left">{html.escape(group.category)}</td>'
                f'<td style="text-align:left">{names_html}</td>'
                f"</tr>"
            )
        parts.append("</tbody></table>")

    parts.append(f"<strong>Mission sequence</strong> ({_count(summary.command_count, 'command')})")
    if summary.commands:
        parts.append("<ol>")
        for cmd in summary.commands:
            parts.append("<li>" + _format_command_html(cmd))
            if cmd.children:
                parts.append("<ul>")
                for sub in cmd.children:
                    parts.append("<li>" + _format_command_html(sub) + "</li>")
                parts.append("</ul>")
            if cmd.nested_count:
                parts.append(f"<div><em>({_count(cmd.nested_count, 'nested command')})</em></div>")
            parts.append("</li>")
        parts.append("</ol>")
    else:
        parts.append("<div><em>(empty)</em></div>")

    parts.append("</div>")
    return "".join(parts)


def _format_command_html(cmd: CommandOutline) -> str:
    type_html = f"<code>{html.escape(cmd.type_name)}</code>"
    if cmd.summary:
        return f"{type_html} — {html.escape(cmd.summary)}"
    return type_html


def _format_command_line(cmd: CommandOutline) -> str:
    if cmd.summary:
        return f"{cmd.type_name} — {cmd.summary}"
    return cmd.type_name


def _count(n: int, noun: str) -> str:
    """Render a count with a naively pluralised noun — ``1 command`` / ``2 commands``."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"
