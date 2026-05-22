"""Unit tests for :func:`gmat_run.parsers.solver_log.parse`.

Fixture-backed tests pin the parser against real GMAT R2026a ``.data`` captures
committed under ``tests/fixtures/solver_log/``; inline ``tmp_path`` fixtures
cover the malformed-input and edge cases, mirroring
:mod:`tests.test_parsers_contact`. No GMAT install is required.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gmat_run.errors import GmatOutputParseError
from gmat_run.parsers.solver_log import parse

_FIXTURES = Path(__file__).parent / "fixtures" / "solver_log"


# --- helpers -----------------------------------------------------------------


def _write(path: Path, content: str, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Write ``content`` verbatim with the requested encoding and line ending."""
    path.write_bytes(content.replace("\n", newline).encode(encoding))
    return path


# A minimal but well-formed DifferentialCorrector file: one converged
# iteration, one variable, one goal. Edge-case tests mutate copies of this.
_DC_MINIMAL = """\
********************************************************
*** Targeter Text File
*** Using Differential Correction
*** 1 variables
*** 1 goals
*** SolverMode:  Solve
********************************************************

Iteration 1
Running Nominal Pass
Variables:
   Burn.V = 0.5

Goals and achieved values:
   Sat.SMA  Desired: 7000 Achieved: 6999.9995
   Tolerance: 0.001

********************************************************
*** Targeting Completed in 1 iterations
********************************************************
"""

# A minimal Yukon file: one nominal pass, one variable, one constraint.
_YUKON_MINIMAL = """\
********************************************************
*** Performing Yukon Optimization (using "Opt")
*** 1 variables; 0 equality constraints; 1 inequality constraints
   Variables:  X
   Inequality Constraints:  C
********************************************************
Opt Iteration 0; Function Evaluation 1; Nominal Pass
   Variables:  X = 1
   Cost Function Value: 5
   Inequality Constraint Variances:
      Delta C = -2

*** Optimization Completed in 0 iterations and 1 function evaluations
*** The Optimizer Converged!
"""


# --- DifferentialCorrector: converged fixture --------------------------------


def test_dc_converged_fixture_shape() -> None:
    df = parse(_FIXTURES / "dc_converged_R2026a.data", max_iterations=25)
    assert list(df.columns) == [
        "iteration",
        "MOI.Element1",
        "geoSat.Earth.SMA",
        "geoSat.Earth.SMA_desired",
        "geoSat.Earth.SMA_residual",
        "geoSat.Earth.SMA_tolerance",
        "status",
    ]
    assert len(df) == 7
    assert df["iteration"].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_dc_converged_fixture_dtypes() -> None:
    df = parse(_FIXTURES / "dc_converged_R2026a.data", max_iterations=25)
    assert df["iteration"].dtype == "int64"
    assert df["MOI.Element1"].dtype == "float64"
    assert df["geoSat.Earth.SMA_residual"].dtype == "float64"
    assert df["status"].dtype == "string"


def test_dc_converged_fixture_status_progression() -> None:
    df = parse(_FIXTURES / "dc_converged_R2026a.data", max_iterations=25)
    assert df["status"].tolist() == ["running"] * 6 + ["converged"]


def test_dc_converged_fixture_residual_is_achieved_minus_desired() -> None:
    df = parse(_FIXTURES / "dc_converged_R2026a.data", max_iterations=25)
    row = df.iloc[-1]
    assert row["geoSat.Earth.SMA_desired"] == 42166.9
    assert row["geoSat.Earth.SMA_residual"] == pytest.approx(
        row["geoSat.Earth.SMA"] - row["geoSat.Earth.SMA_desired"]
    )
    # Final residual is inside tolerance — that is why the run converged.
    assert abs(row["geoSat.Earth.SMA_residual"]) <= row["geoSat.Earth.SMA_tolerance"]


def test_dc_converged_fixture_attrs() -> None:
    df = parse(_FIXTURES / "dc_converged_R2026a.data", max_iterations=25)
    assert df.attrs["solver_type"] == "DifferentialCorrector"
    assert df.attrs["solver_mode"] == "Solve"
    assert df.attrs["n_iterations"] == 7
    assert df.attrs["n_variables"] == 1
    assert df.attrs["n_goals"] == 1
    assert df.attrs["converged"] is True
    assert df.attrs["source_path"].endswith("dc_converged_R2026a.data")


# --- DifferentialCorrector: max-iter fixture ---------------------------------


def test_dc_maxiter_fixture_ends_max_iter() -> None:
    df = parse(_FIXTURES / "dc_maxiter_R2026a.data", max_iterations=1)
    assert len(df) == 1
    assert df["status"].iloc[-1] == "max_iter"
    assert df.attrs["converged"] is False
    assert df.attrs["n_iterations"] == 1


def test_dc_maxiter_without_max_iterations_falls_back_to_failed() -> None:
    # Without the MaximumIterations hint a non-converged run cannot be told
    # apart from a generic failure.
    df = parse(_FIXTURES / "dc_maxiter_R2026a.data")
    assert df["status"].iloc[-1] == "failed"


def test_dc_non_converged_below_max_iterations_is_failed() -> None:
    df = parse(_FIXTURES / "dc_maxiter_R2026a.data", max_iterations=25)
    # One iteration, cap of 25 — it stopped early without converging.
    assert df["status"].iloc[-1] == "failed"


# --- Yukon: converged fixture ------------------------------------------------


def test_yukon_converged_fixture_shape() -> None:
    df = parse(_FIXTURES / "yukon_converged_R2026a.data", max_iterations=500)
    assert list(df.columns) == ["iteration", "X1", "X2", "cost", "G_residual", "status"]
    assert len(df) == 3
    # The iteration column repeats — one row per nominal pass, not per iteration.
    assert df["iteration"].tolist() == [0, 1, 1]


def test_yukon_converged_fixture_values() -> None:
    df = parse(_FIXTURES / "yukon_converged_R2026a.data", max_iterations=500)
    last = df.iloc[-1]
    assert last["X1"] == pytest.approx(2.0)
    assert last["X2"] == pytest.approx(2.0)
    assert last["cost"] == pytest.approx(0.0, abs=1e-20)
    assert df["status"].tolist() == ["running", "running", "converged"]


def test_yukon_converged_fixture_attrs() -> None:
    df = parse(_FIXTURES / "yukon_converged_R2026a.data", max_iterations=500)
    assert df.attrs["solver_type"] == "Yukon"
    assert df.attrs["solver_mode"] == "unknown"
    assert df.attrs["n_iterations"] == 1
    assert df.attrs["n_variables"] == 2
    assert df.attrs["n_goals"] == 1
    assert df.attrs["converged"] is True
    assert df.attrs["source_path"].endswith("yukon_converged_R2026a.data")


# --- Yukon: max-iter fixture -------------------------------------------------


def test_yukon_maxiter_fixture_ends_max_iter() -> None:
    df = parse(_FIXTURES / "yukon_maxiter_R2026a.data", max_iterations=3)
    assert df["status"].iloc[-1] == "max_iter"
    assert df.attrs["solver_type"] == "Yukon"
    assert df.attrs["converged"] is False
    assert df.attrs["n_iterations"] == 3


def test_yukon_maxiter_without_max_iterations_falls_back_to_failed() -> None:
    # Without the MaximumIterations hint a non-converged run cannot be told
    # apart from a generic failure.
    df = parse(_FIXTURES / "yukon_maxiter_R2026a.data")
    assert df["status"].iloc[-1] == "failed"


# --- minimal-template happy paths --------------------------------------------


def test_dc_minimal_template_parses(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "dc.data", _DC_MINIMAL), max_iterations=25)
    assert df["status"].iloc[-1] == "converged"
    assert df["Burn.V"].iloc[0] == pytest.approx(0.5)


def test_yukon_minimal_template_parses(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "y.data", _YUKON_MINIMAL), max_iterations=500)
    assert df["status"].iloc[-1] == "converged"
    assert df["C_residual"].iloc[0] == pytest.approx(-2.0)


def test_crlf_line_endings_parse_identically(tmp_path: Path) -> None:
    lf = parse(_write(tmp_path / "lf.data", _DC_MINIMAL, newline="\n"), max_iterations=25)
    crlf = parse(_write(tmp_path / "crlf.data", _DC_MINIMAL, newline="\r\n"), max_iterations=25)
    pd.testing.assert_frame_equal(lf, crlf)


def test_utf8_bom_is_tolerated(tmp_path: Path) -> None:
    df = parse(_write(tmp_path / "bom.data", _DC_MINIMAL, encoding="utf-8-sig"))
    assert df.attrs["solver_type"] == "DifferentialCorrector"


# --- dispatch / file-level errors --------------------------------------------


def test_empty_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GmatOutputParseError, match="empty"):
        parse(_write(tmp_path / "empty.data", "\n  \n\n"))


def test_unrecognised_header_lists_supported_types(tmp_path: Path) -> None:
    content = "*** Performing SNOPT Optimization\n*** 1 variables\n"
    with pytest.raises(GmatOutputParseError, match="DifferentialCorrector, Yukon"):
        parse(_write(tmp_path / "snopt.data", content))


# --- DifferentialCorrector: malformed input ----------------------------------


def test_dc_no_iterations_raises(tmp_path: Path) -> None:
    header = "\n".join(_DC_MINIMAL.splitlines()[:7]) + "\n"
    with pytest.raises(GmatOutputParseError, match="no iterations"):
        parse(_write(tmp_path / "dc.data", header))


def test_dc_missing_variables_section_raises(tmp_path: Path) -> None:
    broken = _DC_MINIMAL.replace("Variables:\n   Burn.V = 0.5\n\n", "")
    with pytest.raises(GmatOutputParseError, match="missing 'Variables:'"):
        parse(_write(tmp_path / "dc.data", broken))


def test_dc_missing_goals_section_raises(tmp_path: Path) -> None:
    broken = _DC_MINIMAL.replace(
        "Goals and achieved values:\n"
        "   Sat.SMA  Desired: 7000 Achieved: 6999.9995\n"
        "   Tolerance: 0.001\n\n",
        "",
    )
    with pytest.raises(GmatOutputParseError, match="missing 'Goals"):
        parse(_write(tmp_path / "dc.data", broken))


def test_dc_goal_without_tolerance_raises(tmp_path: Path) -> None:
    broken = _DC_MINIMAL.replace("   Tolerance: 0.001\n", "")
    with pytest.raises(GmatOutputParseError, match="Tolerance"):
        parse(_write(tmp_path / "dc.data", broken))


def test_dc_non_numeric_variable_raises(tmp_path: Path) -> None:
    broken = _DC_MINIMAL.replace("Burn.V = 0.5", "Burn.V = lots")
    with pytest.raises(GmatOutputParseError, match="non-numeric"):
        parse(_write(tmp_path / "dc.data", broken))


def test_dc_header_variable_count_mismatch_raises(tmp_path: Path) -> None:
    broken = _DC_MINIMAL.replace("*** 1 variables", "*** 4 variables")
    with pytest.raises(GmatOutputParseError, match="declares 4 variables"):
        parse(_write(tmp_path / "dc.data", broken))


def test_dc_inconsistent_variable_set_across_iterations_raises(tmp_path: Path) -> None:
    second = """
Iteration 2
Running Nominal Pass
Variables:
   Other.V = 0.6

Goals and achieved values:
   Sat.SMA  Desired: 7000 Achieved: 7000.0
   Tolerance: 0.001
"""
    broken = _DC_MINIMAL.replace(
        "********************************************************\n"
        "*** Targeting Completed in 1 iterations",
        second + "********************************************************\n"
        "*** Targeting Completed in 2 iterations",
    )
    with pytest.raises(GmatOutputParseError, match="variable set"):
        parse(_write(tmp_path / "dc.data", broken))


# --- Yukon: malformed input --------------------------------------------------


def test_yukon_no_records_raises(tmp_path: Path) -> None:
    header = "\n".join(_YUKON_MINIMAL.splitlines()[:6]) + "\n"
    with pytest.raises(GmatOutputParseError, match="no iteration records"):
        parse(_write(tmp_path / "y.data", header))


def test_yukon_missing_cost_raises(tmp_path: Path) -> None:
    broken = _YUKON_MINIMAL.replace("   Cost Function Value: 5\n", "")
    with pytest.raises(GmatOutputParseError, match="Cost Function Value"):
        parse(_write(tmp_path / "y.data", broken))


def test_yukon_malformed_assignment_raises(tmp_path: Path) -> None:
    broken = _YUKON_MINIMAL.replace("   Variables:  X = 1", "   Variables:  X 1")
    with pytest.raises(GmatOutputParseError, match="malformed assignment"):
        parse(_write(tmp_path / "y.data", broken))


def test_yukon_perturbation_passes_are_skipped(tmp_path: Path) -> None:
    # A perturbation pass between two nominal passes must not become a row.
    extra = """Opt Iteration 0; Function Evaluation 2; Perturbation Pass 1
   Variables:  X = 1.0001
   Cost Function Value: 5.1
   Inequality Constraint Variances:
      Delta C = -1.9

"""
    augmented = _YUKON_MINIMAL.replace(
        "*** Optimization Completed", extra + "*** Optimization Completed"
    )
    df = parse(_write(tmp_path / "y.data", augmented), max_iterations=500)
    assert len(df) == 1  # only the nominal pass survives


def test_yukon_non_converged_without_max_iterations_is_failed(tmp_path: Path) -> None:
    broken = _YUKON_MINIMAL.replace("*** The Optimizer Converged!\n", "")
    df = parse(_write(tmp_path / "y.data", broken))
    assert df["status"].iloc[-1] == "failed"
    assert df.attrs["converged"] is False


def test_yukon_iteration_count_falls_back_without_terminator(tmp_path: Path) -> None:
    # Strip the "Optimization Completed" line — n_iterations falls back to the
    # highest iteration number actually seen in the records.
    broken = _YUKON_MINIMAL.replace(
        "*** Optimization Completed in 0 iterations and 1 function evaluations\n", ""
    )
    df = parse(_write(tmp_path / "y.data", broken))
    assert df.attrs["n_iterations"] == 0


def test_dc_header_goal_count_mismatch_raises(tmp_path: Path) -> None:
    broken = _DC_MINIMAL.replace("*** 1 goals", "*** 3 goals")
    with pytest.raises(GmatOutputParseError, match="declares 3 goals"):
        parse(_write(tmp_path / "dc.data", broken))


def test_dc_variables_section_may_end_without_blank_line(tmp_path: Path) -> None:
    # The variable list runs straight into the goals header with no blank
    # separator — the parser ends the section on the first non-assignment line.
    tight = _DC_MINIMAL.replace("   Burn.V = 0.5\n\nGoals", "   Burn.V = 0.5\nGoals")
    df = parse(_write(tmp_path / "dc.data", tight), max_iterations=25)
    assert df["Burn.V"].iloc[0] == pytest.approx(0.5)


def test_dc_empty_variables_section_raises(tmp_path: Path) -> None:
    broken = _DC_MINIMAL.replace("Variables:\n   Burn.V = 0.5\n", "Variables:\n")
    with pytest.raises(GmatOutputParseError, match="no variables listed"):
        parse(_write(tmp_path / "dc.data", broken))


def test_dc_empty_goals_section_raises(tmp_path: Path) -> None:
    broken = _DC_MINIMAL.replace(
        "Goals and achieved values:\n"
        "   Sat.SMA  Desired: 7000 Achieved: 6999.9995\n"
        "   Tolerance: 0.001\n",
        "Goals and achieved values:\n",
    )
    with pytest.raises(GmatOutputParseError, match="no goals listed"):
        parse(_write(tmp_path / "dc.data", broken))


def test_dc_without_solver_mode_line_is_unknown(tmp_path: Path) -> None:
    broken = _DC_MINIMAL.replace("*** SolverMode:  Solve\n", "")
    df = parse(_write(tmp_path / "dc.data", broken), max_iterations=25)
    assert df.attrs["solver_mode"] == "unknown"


def test_dc_inconsistent_goal_set_across_iterations_raises(tmp_path: Path) -> None:
    second = """
Iteration 2
Running Nominal Pass
Variables:
   Burn.V = 0.6

Goals and achieved values:
   Sat.ECC  Desired: 0 Achieved: 0.0
   Tolerance: 0.001
"""
    broken = _DC_MINIMAL.replace(
        "********************************************************\n"
        "*** Targeting Completed in 1 iterations",
        second + "********************************************************\n"
        "*** Targeting Completed in 2 iterations",
    )
    with pytest.raises(GmatOutputParseError, match="goal set"):
        parse(_write(tmp_path / "dc.data", broken))


def test_yukon_all_perturbation_passes_raises(tmp_path: Path) -> None:
    broken = _YUKON_MINIMAL.replace("Nominal Pass", "Perturbation Pass 1")
    with pytest.raises(GmatOutputParseError, match="no nominal-pass records"):
        parse(_write(tmp_path / "y.data", broken))


def test_yukon_header_variable_count_mismatch_raises(tmp_path: Path) -> None:
    broken = _YUKON_MINIMAL.replace("*** 1 variables", "*** 5 variables")
    with pytest.raises(GmatOutputParseError, match="declares 5 variables"):
        parse(_write(tmp_path / "y.data", broken))


def test_yukon_empty_variables_raises(tmp_path: Path) -> None:
    broken = _YUKON_MINIMAL.replace("   Variables:  X = 1\n", "   Variables:  \n")
    with pytest.raises(GmatOutputParseError, match="no variables listed"):
        parse(_write(tmp_path / "y.data", broken))


def test_yukon_inconsistent_variable_set_raises(tmp_path: Path) -> None:
    second = """Opt Iteration 1; Function Evaluation 2; Nominal Pass
   Variables:  Y = 2
   Cost Function Value: 3
   Inequality Constraint Variances:
      Delta C = -1

"""
    broken = _YUKON_MINIMAL.replace(
        "*** Optimization Completed", second + "*** Optimization Completed"
    )
    with pytest.raises(GmatOutputParseError, match="variable set"):
        parse(_write(tmp_path / "y.data", broken))


def test_yukon_inconsistent_constraint_set_raises(tmp_path: Path) -> None:
    second = """Opt Iteration 1; Function Evaluation 2; Nominal Pass
   Variables:  X = 2
   Cost Function Value: 3
   Inequality Constraint Variances:
      Delta D = -1

"""
    broken = _YUKON_MINIMAL.replace(
        "*** Optimization Completed", second + "*** Optimization Completed"
    )
    with pytest.raises(GmatOutputParseError, match="constraint set"):
        parse(_write(tmp_path / "y.data", broken))
