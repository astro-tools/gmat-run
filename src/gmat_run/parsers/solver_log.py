"""Parse a GMAT solver ``.data`` file into a :class:`pandas.DataFrame`.

Every ``Target`` / ``Optimize`` run writes a per-``Solver`` text file recording
its iteration history. GMAT names it ``<TypeName><SolverName>.data`` under
``OUTPUT_PATH`` unless the script overrides ``<Solver>.ReportFile``; the file is
produced whether or not that field is set.

The format is solver-specific. :func:`parse` sniffs the header and dispatches:

* ``*** Targeter Text File`` -> :class:`DifferentialCorrector` format. One
  ``Iteration N`` block per pass, each carrying the ``Vary`` variables and a
  ``Goals and achieved values:`` table (desired / achieved / tolerance per
  ``Achieve``). ``DifferentialCorrector`` does not stamp an explicit
  converged/failed terminator, so convergence is inferred from the last
  iteration's residual against its tolerance.
* ``*** Performing Yukon Optimization`` -> :class:`Yukon` format. One
  ``Iteration N; Function Evaluation M; Nominal Pass`` record per cost-function
  evaluation, each carrying the variables, the ``Cost Function Value``, and a
  ``Delta`` per nonlinear constraint. Yukon stamps ``*** The Optimizer
  Converged!`` explicitly.

Both formats normalise to a DataFrame with one row per nominal pass: an
``iteration`` column, one ``float64`` column per variable (script names
verbatim, dots and all), a ``status`` column, and solver-specific value
columns. ``DifferentialCorrector`` adds the goal quartet ``<goal>`` /
``<goal>_desired`` / ``<goal>_residual`` / ``<goal>_tolerance``; ``Yukon`` adds
``cost`` and ``<constraint>_residual`` (the file carries no achieved value,
desired bound, or tolerance for a Yukon constraint). ``df.attrs`` carries
``solver_type``, ``solver_mode``, ``n_iterations``, ``n_variables``,
``n_goals``, ``converged``, and ``source_path``.

The parser is pure file-format work — no ``gmatpy`` import — so it is
unit-testable against committed fixtures alone, mirroring the other parsers in
this subpackage. The one runtime input it cannot read from the file is the
solver's ``MaximumIterations``; :func:`parse` takes it as the optional
``max_iterations`` keyword so a non-converged run can be told apart as
``"max_iter"`` rather than the catch-all ``"failed"``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

import pandas as pd

from gmat_run.errors import GmatOutputParseError

__all__ = ["parse"]


# Solver types this module understands, surfaced in the error raised on an
# unrecognised header so the caller knows what is and isn't supported.
_SUPPORTED_SOLVERS: Final = ("DifferentialCorrector", "Yukon")

# Header substrings that select a format-specific parser. Matched against the
# first few lines so a stray leading blank line does not defeat dispatch.
_DC_HEADER_MARKER: Final = "Targeter Text File"
_YUKON_HEADER_MARKER: Final = "Performing Yukon Optimization"

# --- DifferentialCorrector patterns ------------------------------------------
_DC_ITERATION_RE: Final = re.compile(r"^Iteration\s+(\d+)$")
_DC_NVARS_RE: Final = re.compile(r"\*\*\*\s+(\d+)\s+variables")
_DC_NGOALS_RE: Final = re.compile(r"\*\*\*\s+(\d+)\s+goals")
_DC_MODE_RE: Final = re.compile(r"\*\*\*\s+SolverMode:\s+(\S+)")
# Goal line: "<name>  Desired: <d> Achieved: <a>". The goal name is a
# resource-qualified path (dots, no spaces); the two-space gap before
# "Desired:" is GMAT's column padding. ``.+?`` is non-greedy so the name stops
# at the first "Desired:" anchor.
_DC_GOAL_RE: Final = re.compile(
    r"^(?P<name>.+?)\s+Desired:\s+(?P<desired>\S+)\s+Achieved:\s+(?P<achieved>\S+)$"
)
_DC_TOLERANCE_RE: Final = re.compile(r"^Tolerance:\s+(?P<tolerance>\S+)$")

# --- Yukon patterns ----------------------------------------------------------
_YUKON_NVARS_RE: Final = re.compile(
    r"\*\*\*\s+(?P<vars>\d+)\s+variables;\s+"
    r"(?P<eq>\d+)\s+equality constraints;\s+"
    r"(?P<ineq>\d+)\s+inequality constraints"
)
# Record header: "<Solver> Iteration N; Function Evaluation M; <kind>". Only
# "Nominal Pass" records become rows — perturbation passes are folded away.
_YUKON_RECORD_RE: Final = re.compile(
    r"^(?P<solver>\S+)\s+Iteration\s+(?P<iteration>\d+);\s+"
    r"Function Evaluation\s+(?P<feval>\d+);\s+(?P<kind>.+)$"
)
_YUKON_DELTA_RE: Final = re.compile(r"^Delta\s+(?P<name>\S+)\s*=\s*(?P<value>\S+)$")
_YUKON_TERMINATOR_RE: Final = re.compile(r"Optimization Completed in\s+(\d+)\s+iterations")
_YUKON_CONVERGED_MARKER: Final = "The Optimizer Converged"
_YUKON_NOMINAL_KIND: Final = "Nominal Pass"

# "<name> = <value>" — shared by both formats for variable assignments.
_ASSIGNMENT_RE: Final = re.compile(r"^(?P<name>\S+)\s*=\s*(?P<value>\S+)$")

# Per-row status strings.
_RUNNING: Final = "running"
_CONVERGED: Final = "converged"
_MAX_ITER: Final = "max_iter"
_FAILED: Final = "failed"


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def parse(
    path: str | os.PathLike[str],
    *,
    max_iterations: int | None = None,
) -> pd.DataFrame:
    """Parse a GMAT solver ``.data`` file into a :class:`pandas.DataFrame`.

    The returned schema is one row per nominal pass, with columns:

    * ``iteration`` (``int64``) — the iteration number GMAT reports. For Yukon
      this repeats when an iteration spans several function evaluations.
    * one ``float64`` column per ``Vary`` variable, named verbatim as the
      script declares it (``MOI.Element1``, ``X1``, …).
    * ``status`` (``string``) — ``"running"`` on every row except the last,
      which carries ``"converged"``, ``"max_iter"``, or ``"failed"``.
    * **DifferentialCorrector** adds, per ``Achieve`` goal, the quartet
      ``<goal>`` (achieved), ``<goal>_desired``, ``<goal>_residual``
      (``achieved - desired``), and ``<goal>_tolerance``.
    * **Yukon** adds ``cost`` (the ``Cost Function Value``) and, per nonlinear
      constraint, ``<constraint>_residual`` (GMAT's ``Delta``). Yukon's file
      carries no achieved value, desired bound, or tolerance for a constraint.

    ``df.attrs``:

    * ``solver_type`` — ``"DifferentialCorrector"`` or ``"Yukon"``.
    * ``solver_mode`` — from the ``*** SolverMode:`` line; ``"unknown"`` when
      the format (Yukon) does not write one.
    * ``n_iterations`` — iterations GMAT reported completing.
    * ``n_variables`` / ``n_goals`` — variable and goal/constraint counts.
    * ``converged`` (``bool``) — explicit for Yukon (the converged stamp),
      inferred for DifferentialCorrector (last-iteration residual vs.
      tolerance).
    * ``source_path`` — ``str`` of the file the frame was parsed from.

    Args:
        path: Path to the solver ``.data`` file on disk.
        max_iterations: The solver's ``MaximumIterations`` setting. The file
            does not record it, so without it a non-converged run cannot be
            told apart as ``"max_iter"`` and falls back to ``"failed"``.

    Returns:
        A DataFrame with one row per nominal pass, in file order.

    Raises:
        GmatOutputParseError: The file is empty, its header matches no
            supported solver type, it contains no iterations, or an iteration
            block is malformed (missing variables/goals, an unpaired goal
            line, or a non-numeric value).
    """
    path = Path(path)

    # ``utf-8-sig`` strips an optional BOM; ``newline=None`` activates
    # universal-newline translation so CRLF and LF files parse identically.
    with path.open(encoding="utf-8-sig", newline=None) as fh:
        lines = fh.read().splitlines()

    if not any(line.strip() for line in lines):
        raise GmatOutputParseError("file is empty", path)

    header = "\n".join(lines[:12])
    if _DC_HEADER_MARKER in header:
        df = _parse_differential_corrector(lines, path, max_iterations)
    elif _YUKON_HEADER_MARKER in header:
        df = _parse_yukon(lines, path, max_iterations)
    else:
        raise GmatOutputParseError(
            "unrecognised solver-log header; supported solver types: "
            f"{', '.join(_SUPPORTED_SOLVERS)}",
            path,
        )

    df.attrs["source_path"] = str(path)
    return df


# ----------------------------------------------------------------------------
# DifferentialCorrector
# ----------------------------------------------------------------------------


def _parse_differential_corrector(
    lines: list[str], path: Path, max_iterations: int | None
) -> pd.DataFrame:
    """Parse a ``*** Targeter Text File`` into the normalised DataFrame."""
    solver_mode = _scan_capture(lines, _DC_MODE_RE, default="unknown")
    declared_vars = _scan_int(lines, _DC_NVARS_RE)
    declared_goals = _scan_int(lines, _DC_NGOALS_RE)

    starts = [i for i, line in enumerate(lines) if _DC_ITERATION_RE.match(line.strip())]
    if not starts:
        raise GmatOutputParseError("targeter file contains no iterations", path)

    # Each block runs to the next ``Iteration`` header; the last to end of
    # file. The Jacobian / Inverse Jacobian / scaled-estimate blocks that
    # trail each iteration are simply not consumed by the block parser.
    bounds = [*starts, len(lines)]
    records = [_parse_dc_block(lines[bounds[k] : bounds[k + 1]], path) for k in range(len(starts))]

    var_names = list(records[0].variables)
    goal_names = [goal.name for goal in records[0].goals]
    _check_dc_consistency(records, var_names, goal_names, path)

    if declared_vars is not None and declared_vars != len(var_names):
        raise GmatOutputParseError(
            f"header declares {declared_vars} variables but iterations carry {len(var_names)}",
            path,
        )
    if declared_goals is not None and declared_goals != len(goal_names):
        raise GmatOutputParseError(
            f"header declares {declared_goals} goals but iterations carry {len(goal_names)}",
            path,
        )

    columns: dict[str, pd.Series] = {
        "iteration": pd.Series([r.iteration for r in records], dtype="int64"),
    }
    for name in var_names:
        columns[name] = pd.Series([r.variables[name] for r in records], dtype="float64", name=name)
    for index, goal_name in enumerate(goal_names):
        achieved = [r.goals[index].achieved for r in records]
        desired = [r.goals[index].desired for r in records]
        tolerance = [r.goals[index].tolerance for r in records]
        columns[goal_name] = pd.Series(achieved, dtype="float64", name=goal_name)
        columns[f"{goal_name}_desired"] = pd.Series(desired, dtype="float64")
        columns[f"{goal_name}_residual"] = pd.Series(
            [a - d for a, d in zip(achieved, desired, strict=True)], dtype="float64"
        )
        columns[f"{goal_name}_tolerance"] = pd.Series(tolerance, dtype="float64")

    converged = all(
        abs(goal.achieved - goal.desired) <= goal.tolerance for goal in records[-1].goals
    )
    n_iterations = len(records)
    columns["status"] = _status_column(len(records), converged, n_iterations, max_iterations)

    df = pd.DataFrame(columns)
    df.attrs.update(
        solver_type="DifferentialCorrector",
        solver_mode=solver_mode,
        n_iterations=n_iterations,
        n_variables=len(var_names),
        n_goals=len(goal_names),
        converged=converged,
    )
    return df


class _DcGoal:
    """One ``Achieve`` goal's values within a single iteration block."""

    __slots__ = ("achieved", "desired", "name", "tolerance")

    def __init__(self, name: str, desired: float, achieved: float, tolerance: float) -> None:
        self.name = name
        self.desired = desired
        self.achieved = achieved
        self.tolerance = tolerance


class _DcBlock:
    """The parsed contents of one ``Iteration N`` block."""

    __slots__ = ("goals", "iteration", "variables")

    def __init__(self, iteration: int, variables: dict[str, float], goals: list[_DcGoal]) -> None:
        self.iteration = iteration
        self.variables = variables
        self.goals = goals


def _parse_dc_block(block: list[str], path: Path) -> _DcBlock:
    """Parse one ``Iteration N`` block into its variables and goals."""
    match = _DC_ITERATION_RE.match(block[0].strip())
    if match is None:  # pragma: no cover — block[0] is an Iteration line by construction
        raise GmatOutputParseError(f"expected an 'Iteration' header, got {block[0]!r}", path)
    iteration = int(match.group(1))

    variables = _parse_dc_variables(block, iteration, path)
    goals = _parse_dc_goals(block, iteration, path)
    return _DcBlock(iteration, variables, goals)


def _parse_dc_variables(block: list[str], iteration: int, path: Path) -> dict[str, float]:
    """Read the ``Variables:`` section of a DifferentialCorrector block."""
    index = _section_start(block, "Variables:")
    if index is None:
        raise GmatOutputParseError(f"iteration {iteration}: missing 'Variables:' section", path)
    variables: dict[str, float] = {}
    for line in block[index:]:
        stripped = line.strip()
        if not stripped:
            break
        match = _ASSIGNMENT_RE.match(stripped)
        if match is None:
            break
        name = match.group("name")
        variables[name] = _to_float(
            match.group("value"), f"iteration {iteration}: variable {name!r}", path
        )
    if not variables:
        raise GmatOutputParseError(f"iteration {iteration}: no variables listed", path)
    return variables


def _parse_dc_goals(block: list[str], iteration: int, path: Path) -> list[_DcGoal]:
    """Read the ``Goals and achieved values:`` section of a block."""
    index = _section_start(block, "Goals and achieved values:")
    if index is None:
        raise GmatOutputParseError(
            f"iteration {iteration}: missing 'Goals and achieved values:' section", path
        )
    goals: list[_DcGoal] = []
    cursor = index
    while cursor < len(block):
        stripped = block[cursor].strip()
        if not stripped:
            break
        goal_match = _DC_GOAL_RE.match(stripped)
        if goal_match is None:
            break
        tolerance_line = block[cursor + 1].strip() if cursor + 1 < len(block) else ""
        tolerance_match = _DC_TOLERANCE_RE.match(tolerance_line)
        if tolerance_match is None:
            raise GmatOutputParseError(
                f"iteration {iteration}: goal {goal_match.group('name')!r} is not "
                "followed by a 'Tolerance:' line",
                path,
            )
        where = f"iteration {iteration}: goal {goal_match.group('name')!r}"
        goals.append(
            _DcGoal(
                name=goal_match.group("name"),
                desired=_to_float(goal_match.group("desired"), f"{where} desired", path),
                achieved=_to_float(goal_match.group("achieved"), f"{where} achieved", path),
                tolerance=_to_float(tolerance_match.group("tolerance"), f"{where} tolerance", path),
            )
        )
        cursor += 2
    if not goals:
        raise GmatOutputParseError(f"iteration {iteration}: no goals listed", path)
    return goals


def _check_dc_consistency(
    records: list[_DcBlock], var_names: list[str], goal_names: list[str], path: Path
) -> None:
    """Confirm every iteration block declares the same variables and goals."""
    for record in records:
        if list(record.variables) != var_names:
            raise GmatOutputParseError(
                f"iteration {record.iteration}: variable set "
                f"{sorted(record.variables)} differs from {sorted(var_names)}",
                path,
            )
        if [goal.name for goal in record.goals] != goal_names:
            raise GmatOutputParseError(
                f"iteration {record.iteration}: goal set "
                f"{sorted(g.name for g in record.goals)} differs from {sorted(goal_names)}",
                path,
            )


# ----------------------------------------------------------------------------
# Yukon
# ----------------------------------------------------------------------------


def _parse_yukon(lines: list[str], path: Path, max_iterations: int | None) -> pd.DataFrame:
    """Parse a ``*** Performing Yukon Optimization`` file into the DataFrame."""
    declared = _scan_match(lines, _YUKON_NVARS_RE)
    declared_vars = int(declared.group("vars")) if declared else None
    declared_goals = int(declared.group("eq")) + int(declared.group("ineq")) if declared else None

    starts = [i for i, line in enumerate(lines) if _YUKON_RECORD_RE.match(line.strip()) is not None]
    if not starts:
        raise GmatOutputParseError("Yukon file contains no iteration records", path)

    bounds = [*starts, len(lines)]
    records: list[_YukonRecord] = []
    for k in range(len(starts)):
        record = _parse_yukon_record(lines[bounds[k] : bounds[k + 1]], path)
        # Only nominal passes become rows; perturbation passes are folded away.
        if record is not None:
            records.append(record)
    if not records:
        raise GmatOutputParseError("Yukon file has no nominal-pass records", path)

    var_names = list(records[0].variables)
    constraint_names = list(records[0].constraints)
    _check_yukon_consistency(records, var_names, constraint_names, path)

    if declared_vars is not None and declared_vars != len(var_names):
        raise GmatOutputParseError(
            f"header declares {declared_vars} variables but records carry {len(var_names)}",
            path,
        )

    columns: dict[str, pd.Series] = {
        "iteration": pd.Series([r.iteration for r in records], dtype="int64"),
    }
    for name in var_names:
        columns[name] = pd.Series([r.variables[name] for r in records], dtype="float64", name=name)
    columns["cost"] = pd.Series([r.cost for r in records], dtype="float64", name="cost")
    for name in constraint_names:
        columns[f"{name}_residual"] = pd.Series(
            [r.constraints[name] for r in records], dtype="float64"
        )

    converged = any(_YUKON_CONVERGED_MARKER in line for line in lines)
    n_iterations = _yukon_iteration_count(lines, records)
    n_goals = declared_goals if declared_goals is not None else len(constraint_names)
    columns["status"] = _status_column(len(records), converged, n_iterations, max_iterations)

    df = pd.DataFrame(columns)
    df.attrs.update(
        solver_type="Yukon",
        solver_mode="unknown",
        n_iterations=n_iterations,
        n_variables=len(var_names),
        n_goals=n_goals,
        converged=converged,
    )
    return df


class _YukonRecord:
    """One ``Nominal Pass`` record from a Yukon file."""

    __slots__ = ("constraints", "cost", "iteration", "variables")

    def __init__(
        self,
        iteration: int,
        variables: dict[str, float],
        cost: float,
        constraints: dict[str, float],
    ) -> None:
        self.iteration = iteration
        self.variables = variables
        self.cost = cost
        self.constraints = constraints


def _parse_yukon_record(block: list[str], path: Path) -> _YukonRecord | None:
    """Parse one Yukon record; return ``None`` for non-nominal passes."""
    match = _YUKON_RECORD_RE.match(block[0].strip())
    if match is None:  # pragma: no cover — block[0] is a record header by construction
        raise GmatOutputParseError(f"expected a Yukon record header, got {block[0]!r}", path)
    if match.group("kind").strip() != _YUKON_NOMINAL_KIND:
        return None

    iteration = int(match.group("iteration"))
    where = f"Yukon iteration {iteration}"
    variables: dict[str, float] = {}
    constraints: dict[str, float] = {}
    cost: float | None = None

    for line in block[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Variables:"):
            variables = _parse_yukon_assignments(stripped[len("Variables:") :], where, path)
        elif stripped.startswith("Cost Function Value:"):
            cost = _to_float(
                stripped[len("Cost Function Value:") :].strip(), f"{where}: cost", path
            )
        elif stripped.startswith("Delta "):
            delta_match = _YUKON_DELTA_RE.match(stripped)
            if delta_match is not None:
                name = delta_match.group("name")
                constraints[name] = _to_float(
                    delta_match.group("value"), f"{where}: constraint {name!r}", path
                )

    if not variables:
        raise GmatOutputParseError(f"{where}: no variables listed", path)
    if cost is None:
        raise GmatOutputParseError(f"{where}: missing 'Cost Function Value:'", path)
    return _YukonRecord(iteration, variables, cost, constraints)


def _parse_yukon_assignments(text: str, where: str, path: Path) -> dict[str, float]:
    """Parse ``"  X1 = 0, X2 = 0"`` into ``{"X1": 0.0, "X2": 0.0}``."""
    assignments: dict[str, float] = {}
    for chunk in text.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        match = _ASSIGNMENT_RE.match(stripped)
        if match is None:
            raise GmatOutputParseError(f"{where}: malformed assignment {stripped!r}", path)
        name = match.group("name")
        assignments[name] = _to_float(match.group("value"), f"{where}: variable {name!r}", path)
    return assignments


def _check_yukon_consistency(
    records: list[_YukonRecord],
    var_names: list[str],
    constraint_names: list[str],
    path: Path,
) -> None:
    """Confirm every record carries the same variables and constraints."""
    for record in records:
        if list(record.variables) != var_names:
            raise GmatOutputParseError(
                f"Yukon iteration {record.iteration}: variable set "
                f"{sorted(record.variables)} differs from {sorted(var_names)}",
                path,
            )
        if list(record.constraints) != constraint_names:
            raise GmatOutputParseError(
                f"Yukon iteration {record.iteration}: constraint set "
                f"{sorted(record.constraints)} differs from {sorted(constraint_names)}",
                path,
            )


def _yukon_iteration_count(lines: list[str], records: list[_YukonRecord]) -> int:
    """Iterations Yukon reported, from the terminator or the records seen."""
    for line in lines:
        match = _YUKON_TERMINATOR_RE.search(line)
        if match is not None:
            return int(match.group(1))
    return max(r.iteration for r in records)


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------


def _status_column(
    n_rows: int, converged: bool, n_iterations: int, max_iterations: int | None
) -> pd.Series:
    """Build the ``status`` column: ``running`` until the terminal last row."""
    status = [_RUNNING] * n_rows
    if converged:
        terminal = _CONVERGED
    elif max_iterations is not None and n_iterations >= max_iterations:
        terminal = _MAX_ITER
    else:
        terminal = _FAILED
    status[-1] = terminal
    return pd.Series(status, dtype="string", name="status")


def _section_start(block: list[str], label: str) -> int | None:
    """Index of the line *after* the line whose stripped text equals ``label``."""
    for index, line in enumerate(block):
        if line.strip() == label:
            return index + 1
    return None


def _scan_match(lines: list[str], pattern: re.Pattern[str]) -> re.Match[str] | None:
    """First match of ``pattern`` anywhere in ``lines``, or ``None``."""
    for line in lines:
        match = pattern.search(line)
        if match is not None:
            return match
    return None


def _scan_capture(lines: list[str], pattern: re.Pattern[str], *, default: str) -> str:
    """First capture group of ``pattern`` across ``lines``, or ``default``."""
    match = _scan_match(lines, pattern)
    return match.group(1) if match is not None else default


def _scan_int(lines: list[str], pattern: re.Pattern[str]) -> int | None:
    """First capture group of ``pattern`` as an ``int``, or ``None``."""
    match = _scan_match(lines, pattern)
    return int(match.group(1)) if match is not None else None


def _to_float(token: str, where: str, path: Path) -> float:
    """Convert ``token`` to ``float``; raise a located parse error on failure."""
    try:
        return float(token)
    except ValueError as exc:
        raise GmatOutputParseError(f"{where}: non-numeric value {token!r}", path) from exc
