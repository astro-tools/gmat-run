"""GMAT consumes-what-we-wrote regression for ``Results.write_oem``.

The writer round-trip in :mod:`tests.integration.test_oem_roundtrip` pins the
``parse → write → parse`` loop and the file-format text. This module pins the
other half: GMAT itself accepts the re-emitted OEM as a propagator input.

Per fixture, the test:

1. Runs the source mission and captures the GMAT-emitted ephemeris.
2. Re-emits it through :meth:`gmat_run.Results.write_oem` to a tmp path.
3. Synthesises a tiny loopback ``.script`` whose ``Spacecraft`` points at the
   re-emitted OEM via ``EphemerisName`` and whose ``Propagator`` has
   ``Type = 'CCSDS-OEM'`` (the pattern from ``samples/Ex_OEMPropagation.script``
   in the GMAT install).
4. Runs the loopback mission and asserts the resulting ``ReportFile`` is
   non-empty, well-shaped, and carries finite state vectors with magnitudes
   consistent with a LEO orbit. A run that "accepts" the file but emits
   garbage would slip past a non-emptiness check; the magnitude bound makes
   sure GMAT actually read the data.

GMAT's CCSDS-OEM propagator does not extrapolate beyond the file's epoch
window — propagating past ``stop_time`` raises a runtime error. The loopback
mission stays well inside the source OEM's 6 h window (the source mission
propagates 21600 s; the loopback propagates 21000 s).

Single fixture: only the ``Ex_LEOEphemeris`` (Earth + EME2000) case runs
here. GMAT R2026a's CCSDS-OEM reader rejects the ``ICRF`` frame for Earth
("not supported for the central body 'Earth' ... in a CCSDS-OEM ephemeris
file"), so an Earth + ICRF loopback is a GMAT limitation, not a writer one.
Frame variation lives in :mod:`tests.integration.test_oem_roundtrip` where
GMAT is not in the loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from gmat_run import Mission

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class Sample:
    """One loopback case.

    Attributes:
        script_name: Source mission filename under
            ``tests/integration/fixtures/``.
        ephemeris_key: Resource name on :attr:`Results.ephemerides`.
        coordinate_system: GMAT coordinate system name matching the source
            OEM. Used both for the loopback ``Spacecraft.CoordinateSystem``
            and the ``ReportFile`` parameter axes.
    """

    script_name: str
    ephemeris_key: str
    coordinate_system: str


SAMPLES = [
    # Only one fixture: GMAT R2026a's CCSDS-OEM reader accepts a narrow set
    # of (frame, central body) pairs. Earth + ICRF, which the parallel
    # round-trip test exercises through the writer, is rejected here with
    # "coordinate system 'ICRF' ... is not supported for the central body
    # 'Earth' ... in a CCSDS-OEM ephemeris file". The loopback test's job is
    # narrowly "GMAT accepts what we wrote" — the EME2000 case is enough to
    # prove that round-trip; frame variation is covered upstream in
    # test_oem_roundtrip.
    Sample(
        script_name="Ex_LEOEphemeris.script",
        ephemeris_key="EF",
        coordinate_system="EarthMJ2000Eq",
    ),
]


def _ids(sample: Sample) -> str:
    return sample.script_name.removesuffix(".script")


@pytest.fixture(params=SAMPLES, ids=_ids)
def sample(request: pytest.FixtureRequest) -> Sample:
    param: Sample = request.param
    return param


# Loopback mission template. Single-quoted GMAT string literals do not need
# escaping for forward-slash paths, which is what ``Path.as_posix()`` returns
# on every platform GMAT supports.
_LOOPBACK_TEMPLATE = """\
%  Auto-generated loopback mission for the OEM round-trip integration test.
%  Reads a gmat-run-emitted CCSDS-OEM file via Spacecraft.EphemerisName and
%  the CCSDS-OEM propagator type, then logs the propagated state.

Create Spacecraft EphSat
EphSat.DateFormat = UTCGregorian
EphSat.CoordinateSystem = {coordinate_system}
EphSat.EphemerisName = '{ephemeris_path}'

Create Propagator EphProp
EphProp.Type = 'CCSDS-OEM'
EphProp.StepSize = 60

Create ReportFile RF
RF.Filename = 'loopback.txt'
RF.WriteHeaders = true
RF.Add = {{EphSat.{coordinate_system}.X, EphSat.{coordinate_system}.Y, ...
    EphSat.{coordinate_system}.Z, EphSat.{coordinate_system}.VX, ...
    EphSat.{coordinate_system}.VY, EphSat.{coordinate_system}.VZ}}

BeginMissionSequence
While EphSat.ElapsedSecs < 21000
   Propagate EphProp(EphSat) {{EphSat.ElapsedSecs = 600}}
EndWhile
"""


def test_oem_loopback(
    sample: Sample,
    gmat_available: None,
    fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    src_script = fixtures_dir / sample.script_name
    if not src_script.is_file():
        pytest.skip(f"script not present at {src_script}")

    # Source run: produce the OEM we want GMAT to read back.
    source_mission = Mission.load(src_script)
    source_result = source_mission.run(working_dir=tmp_path / "source")

    reemitted = tmp_path / "loopback_input.oem"
    source_result.write_oem(sample.ephemeris_key, reemitted)

    # Loopback run: GMAT reads the re-emitted OEM.
    loopback_script = tmp_path / "loopback.script"
    loopback_script.write_text(
        _LOOPBACK_TEMPLATE.format(
            coordinate_system=sample.coordinate_system,
            ephemeris_path=reemitted.resolve().as_posix(),
        ),
        encoding="utf-8",
    )

    loopback_mission = Mission.load(loopback_script)
    loopback_result = loopback_mission.run(working_dir=tmp_path / "loopback")

    report = loopback_result.reports["RF"]
    assert len(report) > 0, "GMAT accepted the OEM but produced an empty ReportFile"
    expected_columns = {"X", "Y", "Z", "VX", "VY", "VZ"}
    actual_columns = {col.split(".")[-1] for col in report.columns if col != "Epoch"}
    assert expected_columns <= actual_columns, (
        f"loopback ReportFile is missing state columns; got {sorted(report.columns)}"
    )

    # Sanity bound: source missions are LEO at ~7000 km altitude. A
    # propagator that "accepts" the OEM but emits zeros or NaN would still
    # produce a populated DataFrame; this catches that.
    radii = [
        math.sqrt(
            row[f"EphSat.{sample.coordinate_system}.X"] ** 2
            + row[f"EphSat.{sample.coordinate_system}.Y"] ** 2
            + row[f"EphSat.{sample.coordinate_system}.Z"] ** 2
        )
        for row in report.to_dict(orient="records")
    ]
    assert all(6000.0 < r < 8000.0 for r in radii), (
        f"loopback state vectors look wrong; sample radii = {radii[:3]}..."
    )
