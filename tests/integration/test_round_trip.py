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

# ContactLocator parsers eagerly promote every time column to datetime64;
# goldens carry them in the same Gregorian millisecond text the report writer
# uses. Names match :mod:`gmat_run.parsers.contact`.
_CONTACT_GREGORIAN_FORMAT = "%d %b %Y %H:%M:%S.%f"
_CONTACT_TIME_COLUMNS = frozenset({"Start", "Stop", "Time", "MaxElevationTime"})
_CONTACT_DURATION_COLUMN = "Duration"


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
        contacts: ``{contact-resource-name: golden-file-stem}``. Each
            golden carries the parsed ContactLocator DataFrame; time columns
            are written as Gregorian millisecond text, ``Duration`` as float
            seconds, ``Pass`` and ``Observer`` verbatim.
        rtol: Relative tolerance handed to ``assert_frame_equal``.
        atol: Absolute tolerance handed to ``assert_frame_equal``.
    """

    script_name: str
    reports: Mapping[str, str] = field(default_factory=dict)
    ephemerides: Mapping[str, str] = field(default_factory=dict)
    contacts: Mapping[str, str] = field(default_factory=dict)
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
        # load). Exercises the ephemeris parser end-to-end. EphemerisFile
        # only — the ReportFile path is covered by Ex_TLE_Propagation.
        # ``EF.StepSize = 60`` interpolates onto fixed-cadence timestamps
        # so the Epoch column is exact across platforms, but the integrated
        # state vectors still drift ~1e-5 km between Linux and Windows
        # libm over the 6 h propagation. Tolerance matches
        # Ex_AnalyticMassProperties for the same reason: tight enough to
        # catch parser bugs (which move values by orders of magnitude),
        # loose enough to absorb cross-platform integrator drift.
        script_name="Ex_LEOEphemeris.script",
        script_root="fixtures",
        ephemerides={"EF": "Ex_LEOEphemeris__EF"},
        rtol=1e-6,
        atol=1e-6,
    ),
    Sample(
        # Mirrors Ex_LEOEphemeris but emits STK-TimePosVel instead of
        # CCSDS-OEM. Same fixed-cadence reasoning, plus an Epoch
        # truncation step in the comparator: STK's writer keeps integrator
        # drift in the offset column at sub-microsecond precision, so the
        # parsed Epoch carries ~20 ns of platform-libm jitter per record.
        # Goldens are ms-truncated on serialise (same as OEM); the actual
        # frame is ms-truncated below before compare.
        script_name="Ex_STKEphemeris.script",
        script_root="fixtures",
        ephemerides={"EF": "Ex_STKEphemeris__EF"},
        rtol=1e-6,
        atol=1e-6,
    ),
    Sample(
        # Six ContactLocators in one mission cover all six ReportFormat
        # variants (Legacy + the five tabular formats). LegacyCL pins
        # the per-observer-block parser; ContactRangeCL / MaxElevCL /
        # MaxElevRangeCL pin the per-event tabular parsers; AzElRangeCL
        # (ReportTimeFormat=ISOYD) and AzElRangeRRCL (ReportTimeFormat=UTCMJD)
        # pin the per-(Pass, tick) grain plus the two non-Gregorian
        # ReportTimeFormat axes. Tolerance matches the ephemeris cases:
        # tight enough to flag a parser regression, loose enough to absorb
        # the sub-millisecond MJD round-trip drift the comparator already
        # truncates away.
        script_name="Ex_ContactLocatorAllFormats.script",
        script_root="fixtures",
        contacts={
            "LegacyCL": "Ex_ContactLocatorAllFormats__Legacy",
            "ContactRangeCL": "Ex_ContactLocatorAllFormats__ContactRange",
            "MaxElevCL": "Ex_ContactLocatorAllFormats__MaxElev",
            "MaxElevRangeCL": "Ex_ContactLocatorAllFormats__MaxElevRange",
            "AzElRangeCL": "Ex_ContactLocatorAllFormats__AzElRange",
            "AzElRangeRRCL": "Ex_ContactLocatorAllFormats__AzElRangeRR",
        },
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


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Where in-repo integration fixture scripts live."""
    return Path(__file__).parent / "fixtures"


def _run_sample(
    sample: Sample,
    samples_dir: Path,
    fixtures_dir: Path,
    tmp_path: Path,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """Load and run a sample under an isolated workspace.

    Returns ``(reports, ephemerides, contacts)`` for every output the sample
    asked for. Surfaces parse errors and run failures as the underlying
    exception so pytest's traceback names the actual culprit.

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
    contacts = {name: result.contacts[name] for name in sample.contacts}
    return reports, ephemerides, contacts


def _read_golden(
    path: Path, *, epoch_format: Literal["report", "ephemeris", "contact"]
) -> pd.DataFrame:
    """Load a golden CSV and re-promote its epoch columns.

    ``report`` goldens use GMAT's printed Gregorian / ModJulian convention and
    delegate to :func:`gmat_run.parsers.epoch.promote_epochs` so the comparison
    matches what :meth:`Mission.run` returns. ``ephemeris`` goldens carry a
    plain ISO-8601 ``Epoch`` column written by the ephemeris parser. ``contact``
    goldens carry one or more named time columns plus a numeric ``Duration``
    column written as float seconds.
    """
    df = pd.read_csv(path)
    if epoch_format == "report":
        return promote_epochs(df)
    if epoch_format == "ephemeris":
        # CCSDS-OEM ephemerides: a single ``Epoch`` column in ISO-8601.
        df["Epoch"] = pd.to_datetime(df["Epoch"], format=_ISO8601_FORMAT).astype("datetime64[ns]")
        return df
    # ``contact``: time columns are Gregorian; Duration is seconds. The parser
    # surfaces ``Observer`` as ``object`` (matching every other gmat-run
    # frame); pandas 3.x ``read_csv`` infers string columns as ``str``, so
    # downcast back to ``object`` here for dtype-strict compare.
    for col in df.columns:
        if col in _CONTACT_TIME_COLUMNS:
            df[col] = pd.to_datetime(df[col], format=_CONTACT_GREGORIAN_FORMAT).astype(
                "datetime64[ns]"
            )
        elif col == _CONTACT_DURATION_COLUMN:
            df[col] = pd.to_timedelta(df[col], unit="s").astype("timedelta64[ns]")
        elif df[col].dtype.name == "str":
            df[col] = df[col].astype("object")
    return df


def _truncate_datetime_to_ms(df: pd.DataFrame) -> pd.DataFrame:
    """Floor every datetime64 column to millisecond resolution.

    Goldens round-trip through millisecond-precision text on serialise; the
    actual frame must match that resolution before :func:`assert_frame_equal`
    can compare them with strict dtype checks.

    Timedelta columns are NOT truncated — the contact-format goldens write
    duration as float seconds via ``dt.total_seconds()`` and ``%.15g``, which
    is round-trip-symmetric back through ``pd.to_timedelta(unit='s')``.
    """
    out: pd.DataFrame = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.floor("ms").astype("datetime64[ns]")
    return out


def _write_golden(
    df: pd.DataFrame,
    path: Path,
    *,
    epoch_format: Literal["report", "ephemeris", "contact"],
) -> None:
    """Serialise ``df`` to a golden CSV. Epoch text shape mirrors the source."""
    out = df.copy()
    fmt = (
        _CONTACT_GREGORIAN_FORMAT
        if epoch_format == "contact"
        else "%d %b %Y %H:%M:%S.%f"
        if epoch_format == "report"
        else _ISO8601_FORMAT
    )
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            # Strip the trailing three microsecond digits so the printed
            # text matches GMAT's millisecond-precision emitter exactly.
            out[col] = out[col].dt.strftime(fmt).str.slice(stop=-3)
        elif pd.api.types.is_timedelta64_dtype(out[col]):
            # Goldens carry duration as float seconds — round-trips through
            # CSV without dragging a pandas-specific repr in.
            out[col] = out[col].dt.total_seconds()
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
    actual_reports, actual_eph, actual_contacts = _run_sample(
        sample, samples_dir, fixtures_dir, tmp_path
    )
    assert set(actual_reports) == set(sample.reports), (
        f"GMAT wrote reports {sorted(actual_reports)} "
        f"but the test expected {sorted(sample.reports)}"
    )
    assert set(actual_eph) == set(sample.ephemerides), (
        f"GMAT wrote ephemerides {sorted(actual_eph)} "
        f"but the test expected {sorted(sample.ephemerides)}"
    )
    assert set(actual_contacts) == set(sample.contacts), (
        f"GMAT wrote contacts {sorted(actual_contacts)} "
        f"but the test expected {sorted(sample.contacts)}"
    )

    regenerated: list[Path] = []
    cases: list[tuple[pd.DataFrame, str, Literal["report", "ephemeris", "contact"]]] = []
    for name, stem in sample.reports.items():
        cases.append((actual_reports[name], stem, "report"))
    for name, stem in sample.ephemerides.items():
        cases.append((actual_eph[name], stem, "ephemeris"))
    for name, stem in sample.contacts.items():
        cases.append((actual_contacts[name], stem, "contact"))

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
        # Goldens are serialised with millisecond-precision epoch text and
        # re-parsed back; truncate the actual frame's datetime columns to
        # match. For the OEM path this is a no-op (the file format is
        # already ms-precise); for STK it absorbs the sub-microsecond
        # integrator drift that surfaces in offset-based epoch reconstruction.
        actual = _truncate_datetime_to_ms(df)
        assert_frame_equal(
            actual,
            expected,
            rtol=sample.rtol,
            atol=sample.atol,
            check_dtype=True,
            check_like=False,
        )

    if regenerated:
        names = ", ".join(str(p.relative_to(golden_dir.parent.parent)) for p in regenerated)
        pytest.skip(f"regenerated goldens: {names}")
