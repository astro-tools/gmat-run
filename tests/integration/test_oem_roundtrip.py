"""Writer round-trip regression: ``Results.write_oem`` against committed goldens.

Companion to :mod:`tests.integration.test_round_trip`. Where that suite pins
the *parser* by diffing GMAT's stock output against committed CSVs, this one
pins the *writer*: it takes a GMAT-emitted CCSDS-OEM ephemeris, re-emits it
through :func:`gmat_run.writers.oem.write_oem`, and asserts both that the
parse → write → parse loop is numerically stable and that the emitted file
text matches a committed golden.

Two assertions per case:

1. **DataFrame round-trip.** Parse the GMAT-emitted OEM, re-emit it through
   the writer, parse the re-emission, and compare the two frames with
   :func:`pandas.testing.assert_frame_equal`. The writer's
   :func:`gmat_run.writers.oem._format_epoch` truncates to milliseconds, so
   both sides are ms-floored before compare; the per-column tolerance is
   ``rtol=atol=1e-9`` (no integrator drift in this loop — the only source of
   loss is float ↔ text serialisation).
2. **Text golden.** Read the re-emitted OEM as text, strip the volatile
   ``CREATION_DATE`` header (the writer stamps :func:`_now_iso` at emit
   time), and compare to ``tests/integration/golden/<stem>__roundtrip.oem``.
   Pins the file format itself — line ordering, units, comment placement —
   so a ``ccsds-ndm`` upgrade that re-orders the KVN output trips CI.

Three cases together cover the writer's branch points:

* ``Ex_LEOEphemeris`` — ``EarthMJ2000Eq`` + ``UTC``: the GMAT-internal frame
  name aliased to ``EME2000`` on the way out. Public-API path
  (``Results.write_oem``).
* ``Ex_ICRFEphemeris`` — ``EarthICRF`` + ``UTC``: the second alias-table
  entry. Public-API path. GMAT's CCSDS-OEM emitter writes ``TIME_SYSTEM =
  UTC`` regardless of the spacecraft's ``EpochFormat``, so a real
  GMAT-driven mission cannot reach the writer's non-UTC branches; that is
  what the third case is for.
* ``Ex_LEOEphemeris[TAI]`` — same source mission, but the parsed DataFrame
  is converted UTC → TAI via :func:`gmat_run.parsers.ephemeris.parse`'s
  ``convert_to`` argument before being handed to the writer. Reaches the
  writer's ``TIME_SYSTEM = TAI`` branch with real GMAT-derived state. Uses
  the function-level writer because :meth:`Results.write_oem` would emit
  the cached UTC parse instead. Time-scale unit coverage for the writer's
  full A1/TAI/TT/TDB matrix lives in ``tests/test_writers_oem.py``.

Regenerate goldens with ``pytest --regenerate-golden tests/integration/`` on
a machine with the supported GMAT version installed and the ``[ccsds-ndm]``
and ``[astropy]`` extras; the run rewrites the ``.oem`` files and skips
each case so the result is loud about *not* having compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from gmat_run import Mission
from gmat_run.parsers.ephemeris import parse as parse_oem
from gmat_run.writers.oem import write_oem

from ._compare import truncate_datetime_to_ms

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class Sample:
    """One writer round-trip case.

    Attributes:
        script_name: Filename under ``tests/integration/fixtures/`` for the
            mission that produces the source OEM.
        ephemeris_key: Resource name on :attr:`Results.ephemerides`.
        golden_stem: Stem (no extension) of the committed re-emission golden
            under ``tests/integration/golden/``.
        convert_to_scale: When set, the source OEM is re-parsed with this
            target time scale (UTC → TAI etc., delegated to
            :func:`gmat_run.parsers.ephemeris.parse`) before being handed to
            the writer. Forces the test off ``Results.write_oem`` (which
            would emit the cached untransformed parse) and onto the
            function-level :func:`gmat_run.writers.oem.write_oem`. ``None``
            uses the public-API path with no transform.
    """

    script_name: str
    ephemeris_key: str
    golden_stem: str
    convert_to_scale: str | None = None


SAMPLES = [
    Sample(
        script_name="Ex_LEOEphemeris.script",
        ephemeris_key="EF",
        golden_stem="Ex_LEOEphemeris__roundtrip",
    ),
    Sample(
        script_name="Ex_ICRFEphemeris.script",
        ephemeris_key="EF",
        golden_stem="Ex_ICRFEphemeris__roundtrip",
    ),
    Sample(
        script_name="Ex_LEOEphemeris.script",
        ephemeris_key="EF",
        golden_stem="Ex_LEOEphemeris__roundtrip_tai",
        convert_to_scale="TAI",
    ),
]


def _ids(sample: Sample) -> str:
    base = sample.script_name.removesuffix(".script")
    return f"{base}[{sample.convert_to_scale}]" if sample.convert_to_scale else base


@pytest.fixture(params=SAMPLES, ids=_ids)
def sample(request: pytest.FixtureRequest) -> Sample:
    param: Sample = request.param
    return param


def _strip_volatile(text: str) -> str:
    """Drop header lines that change between emissions of the same data.

    The writer stamps ``CREATION_DATE`` from :func:`datetime.now` so two runs
    of the same input produce textually-distinct files. Strip that line on
    both sides before compare. Line endings are normalised to ``\\n`` so the
    golden survives Linux ↔ Windows ↔ macOS check-outs.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(
        line for line in text.split("\n") if not line.lstrip().startswith("CREATION_DATE")
    )


def test_oem_roundtrip(
    sample: Sample,
    gmat_available: None,
    fixtures_dir: Path,
    golden_dir: Path,
    tmp_path: Path,
    regenerate_golden: bool,
) -> None:
    src = fixtures_dir / sample.script_name
    if not src.is_file():
        pytest.skip(f"script not present at {src}")

    mission = Mission.load(src)
    result = mission.run(working_dir=tmp_path / "run")

    reemitted = tmp_path / "reemitted.oem"
    if sample.convert_to_scale is None:
        # Public-API path: parse-once-and-cache via Results.ephemerides, emit
        # via Results.write_oem.
        df_source = result.ephemerides[sample.ephemeris_key]
        result.write_oem(sample.ephemeris_key, reemitted)
    else:
        # Function-level path with an in-DataFrame time-scale conversion.
        # Bypasses Results.ephemerides because that returns the cached UTC
        # parse with no convert_to applied.
        source_path = result.ephemeris_paths[sample.ephemeris_key]
        df_source = parse_oem(source_path, convert_to=sample.convert_to_scale)
        write_oem(df_source, reemitted)

    df_reparsed = parse_oem(reemitted)
    assert_frame_equal(
        truncate_datetime_to_ms(df_reparsed),
        truncate_datetime_to_ms(df_source),
        rtol=1e-9,
        atol=1e-9,
        check_dtype=True,
        check_like=False,
    )

    # Text golden: pins the file format itself. CREATION_DATE is filtered
    # because the writer stamps "now" each time.
    reemitted_text = _strip_volatile(reemitted.read_text(encoding="utf-8-sig"))
    golden_path = golden_dir / f"{sample.golden_stem}.oem"

    if regenerate_golden:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(reemitted_text + "\n", encoding="utf-8")
        rel = golden_path.relative_to(golden_dir.parent.parent)
        pytest.skip(f"regenerated golden: {rel}")

    if not golden_path.is_file():
        pytest.fail(
            f"missing golden {golden_path} — run "
            f"`pytest --regenerate-golden tests/integration/` to create it"
        )

    expected_text = _strip_volatile(golden_path.read_text(encoding="utf-8-sig"))
    assert reemitted_text.rstrip() == expected_text.rstrip(), (
        "re-emitted OEM differs from committed golden — "
        "either the writer changed or ccsds-ndm output drifted"
    )
