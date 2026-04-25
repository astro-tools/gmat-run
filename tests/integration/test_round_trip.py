"""Round-trip regression: stock GMAT samples vs committed CSV goldens.

Each parametrised case runs a real ``Ex_*.script`` from
``<install.root>/samples/`` end-to-end via :class:`gmat_run.Mission`, then
diffs every ReportFile output against ``tests/integration/golden/<sample>__<report>.csv``
with :func:`pandas.testing.assert_frame_equal`. Any change to the parser, the
field-override path, the working-directory handling, or upstream GMAT itself
that perturbs report output trips a CI failure.

The samples and the resources they declare are intentionally narrow:

* ``Ex_TLE_Propagation`` — SGP4-initialised propagation through an EGM-96
  point-mass force model. Exercises the propagation + ``ReportFile`` path.
* ``Ex_AnalyticMassProperties`` — finite-burn maneuver with mass depletion
  driven by analytic mass-properties parameters. Exercises burn-driven
  parameter recalculation alongside the report writer.

We deliberately picked samples without an ``OpenFramesView``/
``OpenFramesInterface`` resource: the API runtime does not load that GUI
plugin, and a script that declares one fails to parse headlessly.

Goldens are CSV (text-diffable in PRs, no parquet engine dependency) at
``%.15g`` precision (15 significant digits — sufficient for double-precision
GMAT output). Epoch columns are written as their printed Gregorian /
ModJulian strings; on read the production
:func:`gmat_run.parsers.epoch.promote_epochs` re-promotes them to
``datetime64[ns]`` so the comparison runs against typed values, the same as
what :meth:`Mission.run` returns.

Regenerate goldens with ``pytest --regenerate-golden tests/integration/`` on
a machine with the supported GMAT version installed; the run rewrites the
files and skips each case so the result is loud about *not* having compared.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gmat_run import Mission
from gmat_run.parsers.epoch import promote_epochs

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class Sample:
    """One stock-sample test case.

    Attributes:
        script_name: Filename under ``<install.root>/samples/``.
        reports: ``{report-resource-name: golden-file-stem}``. Multiple
            reports per script are supported but uncommon — each maps to its
            own golden file under ``tests/integration/golden/``.
        rtol: Relative tolerance handed to ``assert_frame_equal``.
        atol: Absolute tolerance handed to ``assert_frame_equal``.
    """

    script_name: str
    reports: Mapping[str, str]
    rtol: float = 1e-9
    atol: float = 1e-9


SAMPLES = [
    Sample(
        script_name="Ex_TLE_Propagation.script",
        reports={"RF": "Ex_TLE_Propagation__RF"},
    ),
    Sample(
        # Burn integration accumulates more numerical state than pure
        # propagation; a slightly looser tolerance accommodates platform-
        # libm differences without weakening the regression signal — a
        # real parser/coercion regression still moves values by orders
        # of magnitude more than 1e-6.
        script_name="Ex_AnalyticMassProperties.script",
        reports={"RF": "Ex_AnalyticMassProperties__RF"},
        rtol=1e-6,
        atol=1e-6,
    ),
]


def _ids(sample: Sample) -> str:
    return sample.script_name.removesuffix(".script")


@pytest.fixture(params=SAMPLES, ids=_ids)
def sample(request: pytest.FixtureRequest) -> Sample:
    param: Sample = request.param
    return param


def _run_sample(sample: Sample, samples_dir: Path, tmp_path: Path) -> dict[str, pd.DataFrame]:
    """Load and run a stock sample under an isolated workspace.

    Returns ``{report-name: DataFrame}`` for every ReportFile the script
    declared. Surfaces parse errors and run failures as the underlying
    exception so pytest's traceback names the actual culprit.

    The script is loaded *in place* from ``samples/`` rather than copied to
    ``tmp_path``. Stock samples often carry relative-path references into the
    install's ``data/`` tree (e.g. ``Spacecraft.ModelFile = '../data/...'``)
    that resolve only when GMAT parses the script from its original location.
    Output redirection is unaffected: :meth:`Mission.run` rewrites every
    relative ``ReportFile.Filename`` to land under ``working_dir``, so we
    still get the per-test isolation the suite needs.
    """
    src = samples_dir / sample.script_name
    if not src.is_file():
        pytest.skip(f"sample not present in this GMAT install: {src}")

    mission = Mission.load(src)
    result = mission.run(working_dir=tmp_path / "run")
    return {name: result.reports[name] for name in sample.reports}


def _read_golden(path: Path) -> pd.DataFrame:
    """Load a golden CSV and re-promote its epoch columns to datetime64.

    Mirrors what the production :class:`Results` exposes — the report parser
    runs ``promote_epochs`` as its final step, so the golden is round-tripped
    through the same code path before comparison.
    """
    df = pd.read_csv(path)
    return promote_epochs(df)


def _write_golden(df: pd.DataFrame, path: Path) -> None:
    # Convert datetime columns to ISO-8601-like strings for round-trip
    # through ``promote_epochs`` on read. Using the parser's recognised
    # Gregorian format keeps a single canonical text form across CI hosts.
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%d %b %Y %H:%M:%S.%f").str.slice(stop=-3)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.15g")


def test_sample_round_trip(
    sample: Sample,
    gmat_available: None,
    samples_dir: Path,
    golden_dir: Path,
    tmp_path: Path,
    regenerate_golden: bool,
) -> None:
    actual = _run_sample(sample, samples_dir, tmp_path)
    assert set(actual) == set(sample.reports), (
        f"GMAT wrote {sorted(actual)} but the test expected {sorted(sample.reports)}"
    )

    regenerated: list[Path] = []
    for report_name, golden_stem in sample.reports.items():
        golden_path = golden_dir / f"{golden_stem}.csv"
        df = actual[report_name]

        if regenerate_golden:
            _write_golden(df, golden_path)
            regenerated.append(golden_path)
            continue

        if not golden_path.is_file():
            pytest.fail(
                f"missing golden {golden_path} — run "
                f"`pytest --regenerate-golden tests/integration/` to create it"
            )
        expected = _read_golden(golden_path)
        assert_frame_equal(
            df,
            expected,
            rtol=sample.rtol,
            atol=sample.atol,
            check_dtype=True,
        )

    if regenerated:
        names = ", ".join(str(p.relative_to(golden_dir.parent.parent)) for p in regenerated)
        pytest.skip(f"regenerated goldens: {names}")
