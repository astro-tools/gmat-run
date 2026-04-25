"""Round-trip regression: stock GMAT samples vs committed CSV goldens.

Each parametrised case runs a real ``Ex_*.script`` end-to-end via
:class:`gmat_run.Mission`, then diffs every ReportFile and EphemerisFile
output against ``tests/integration/golden/<sample>__<resource>.csv`` with
:func:`pandas.testing.assert_frame_equal`. Any change to a parser, the
field-override path, the working-directory handling, or upstream GMAT itself
that perturbs the output trips a CI failure.

Scripts come from two locations:

* ``<install.root>/samples/`` for stock GMAT samples (the default).
* ``tests/integration/fixtures/`` for in-repo scripts authored when no stock
  sample fits — currently the EphemerisFile case, since every stock sample
  emitting one declares ``OpenFramesInterface``/``OpenFramesView`` and the
  API runtime does not load that GUI plugin (the script then fails to parse
  headlessly).

The resources are intentionally narrow:

* ``Ex_TLE_Propagation`` (stock) — SGP4-initialised propagation through an
  EGM-96 point-mass force model. Exercises the propagation + ``ReportFile``
  path.
* ``Ex_AnalyticMassProperties`` (stock) — finite-burn maneuver with mass
  depletion driven by analytic mass-properties parameters. Exercises
  burn-driven parameter recalculation alongside the report writer.
* ``Ex_LEOEphemeris`` (in-repo fixture) — 6 h LEO propagation that emits both
  a ``ReportFile`` and a CCSDS-OEM ``EphemerisFile``. Exercises the
  ephemeris parser + lazy ``Results.ephemerides`` dispatch end-to-end.

Goldens are CSV (text-diffable in PRs, no parquet engine dependency) at
``%.15g`` precision (15 significant digits — sufficient for double-precision
GMAT output). Epoch columns are written as their printed Gregorian /
ModJulian / ISO-8601 strings; on read the production parser is re-applied so
the comparison runs against typed values, the same as what
:meth:`Mission.run` returns.

Regenerate goldens with ``pytest --regenerate-golden tests/integration/`` on
a machine with the supported GMAT version installed; the run rewrites the
files and skips each case so the result is loud about *not* having compared.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gmat_run import Mission
from gmat_run.parsers.epoch import promote_epochs

pytestmark = pytest.mark.integration


# ISO-8601 epoch text that EphemerisFile (CCSDS-OEM) produces and the
# ephemeris parser consumes. Used when serialising and re-reading goldens
# whose epoch column came from the ephemeris writer (not the report writer).
_ISO8601_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


@dataclass(frozen=True)
class Sample:
    """One round-trip test case.

    Attributes:
        script_name: Filename of the script. When ``script_root='install'``
            it is resolved under ``<install.root>/samples/``; when
            ``script_root='fixtures'`` it is resolved under
            ``tests/integration/fixtures/``.
        script_root: Where to find ``script_name``.
        reports: ``{report-resource-name: golden-file-stem}``. Maps to the
            CSV golden under ``tests/integration/golden/``.
        ephemerides: ``{ephemeris-resource-name: golden-file-stem}``. Same
            convention; epoch column is ISO-8601 in the golden.
        rtol: Relative tolerance handed to ``assert_frame_equal``.
        atol: Absolute tolerance handed to ``assert_frame_equal``.
    """

    script_name: str
    reports: Mapping[str, str] = field(default_factory=dict)
    ephemerides: Mapping[str, str] = field(default_factory=dict)
    script_root: Literal["install", "fixtures"] = "install"
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
    Sample(
        # In-repo fixture; no stock sample emits an EphemerisFile without
        # also pulling in OpenFramesInterface (which the API runtime can't
        # load). Exercises the ephemeris parser end-to-end.
        script_name="Ex_LEOEphemeris.script",
        script_root="fixtures",
        reports={"RF": "Ex_LEOEphemeris__RF"},
        ephemerides={"EF": "Ex_LEOEphemeris__EF"},
    ),
]


def _ids(sample: Sample) -> str:
    return sample.script_name.removesuffix(".script")


@pytest.fixture(params=SAMPLES, ids=_ids)
def sample(request: pytest.FixtureRequest) -> Sample:
    param: Sample = request.param
    return param


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Where in-repo integration fixture scripts live."""
    return Path(__file__).parent / "fixtures"


def _run_sample(
    sample: Sample,
    samples_dir: Path,
    fixtures_dir: Path,
    tmp_path: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load and run a sample under an isolated workspace.

    Returns ``({report-name: DataFrame}, {ephemeris-name: DataFrame})`` for
    every output the sample asked for. Surfaces parse errors and run failures
    as the underlying exception so pytest's traceback names the actual
    culprit.

    Stock samples are loaded *in place* from ``samples/`` because they often
    carry relative-path references into the install's ``data/`` tree that
    resolve only when GMAT parses the script from its original location.
    Output redirection is unaffected: :meth:`Mission.run` rewrites every
    relative output ``Filename`` to land under ``working_dir``, so we still
    get the per-test isolation the suite needs. In-repo fixtures don't have
    that constraint but use the same loading path for consistency.
    """
    root = samples_dir if sample.script_root == "install" else fixtures_dir
    src = root / sample.script_name
    if not src.is_file():
        pytest.skip(f"script not present at {src}")

    mission = Mission.load(src)
    result = mission.run(working_dir=tmp_path / "run")
    reports = {name: result.reports[name] for name in sample.reports}
    ephemerides = {name: result.ephemerides[name] for name in sample.ephemerides}
    return reports, ephemerides


def _read_golden(path: Path, *, epoch_format: Literal["report", "ephemeris"]) -> pd.DataFrame:
    """Load a golden CSV and re-promote its epoch columns.

    ``report`` goldens use GMAT's printed Gregorian / ModJulian convention and
    delegate to :func:`gmat_run.parsers.epoch.promote_epochs` so the comparison
    matches what :meth:`Mission.run` returns. ``ephemeris`` goldens carry a
    plain ISO-8601 ``Epoch`` column written by the ephemeris parser.
    """
    df = pd.read_csv(path)
    if epoch_format == "report":
        return promote_epochs(df)
    # CCSDS-OEM ephemerides: a single ``Epoch`` column in ISO-8601.
    df["Epoch"] = pd.to_datetime(df["Epoch"], format=_ISO8601_FORMAT).astype("datetime64[ns]")
    return df


def _write_golden(
    df: pd.DataFrame, path: Path, *, epoch_format: Literal["report", "ephemeris"]
) -> None:
    """Serialise ``df`` to a golden CSV. Epoch text shape mirrors the source."""
    out = df.copy()
    fmt = "%d %b %Y %H:%M:%S.%f" if epoch_format == "report" else _ISO8601_FORMAT
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            # Strip the trailing three microsecond digits so the printed
            # text matches GMAT's millisecond-precision emitter exactly.
            out[col] = out[col].dt.strftime(fmt).str.slice(stop=-3)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.15g")


def test_sample_round_trip(
    sample: Sample,
    gmat_available: None,
    samples_dir: Path,
    fixtures_dir: Path,
    golden_dir: Path,
    tmp_path: Path,
    regenerate_golden: bool,
) -> None:
    actual_reports, actual_eph = _run_sample(sample, samples_dir, fixtures_dir, tmp_path)
    assert set(actual_reports) == set(sample.reports), (
        f"GMAT wrote reports {sorted(actual_reports)} "
        f"but the test expected {sorted(sample.reports)}"
    )
    assert set(actual_eph) == set(sample.ephemerides), (
        f"GMAT wrote ephemerides {sorted(actual_eph)} "
        f"but the test expected {sorted(sample.ephemerides)}"
    )

    regenerated: list[Path] = []
    cases: list[tuple[pd.DataFrame, str, Literal["report", "ephemeris"]]] = []
    for name, stem in sample.reports.items():
        cases.append((actual_reports[name], stem, "report"))
    for name, stem in sample.ephemerides.items():
        cases.append((actual_eph[name], stem, "ephemeris"))

    for df, stem, epoch_format in cases:
        golden_path = golden_dir / f"{stem}.csv"

        if regenerate_golden:
            _write_golden(df, golden_path, epoch_format=epoch_format)
            regenerated.append(golden_path)
            continue

        if not golden_path.is_file():
            pytest.fail(
                f"missing golden {golden_path} — run "
                f"`pytest --regenerate-golden tests/integration/` to create it"
            )
        expected = _read_golden(golden_path, epoch_format=epoch_format)
        assert_frame_equal(
            df,
            expected,
            rtol=sample.rtol,
            atol=sample.atol,
            check_dtype=True,
            check_like=False,
        )

    if regenerated:
        names = ", ".join(str(p.relative_to(golden_dir.parent.parent)) for p in regenerated)
        pytest.skip(f"regenerated goldens: {names}")
