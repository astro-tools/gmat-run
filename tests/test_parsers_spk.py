"""Unit tests for :func:`gmat_run.parsers.spk.parse`.

The committed ``Ex_SPK.bsp`` fixture under ``tests/fixtures/ephemeris/`` is
a single-segment Type 13 (Hermite) kernel built by ``_make_spk.py`` —
ten synthetic equatorial-plane states sampled at 60-second cadence.
Multi-target and unhappy-path tests build their kernels in ``tmp_path``
on the fly using :mod:`spiceypy` so the committed fixture set stays
small.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

# spiceypy is required to build the synthetic kernels these tests use.
# It is a dev-group dependency and a documented runtime extra
# (gmat-run[spiceypy]); if it is missing locally we cannot exercise the
# parser end-to-end and skipping is more useful than a confusing
# ImportError on collection.
spiceypy = pytest.importorskip("spiceypy")

from gmat_run.errors import GmatOutputParseError  # noqa: E402
from gmat_run.parsers.spk import is_spk_ephemeris, parse  # noqa: E402

_FIXTURE = Path(__file__).parent / "fixtures" / "ephemeris" / "Ex_SPK.bsp"

# Match _make_spk.py — these tests reach into the kernel's known shape.
_FIXTURE_TARGET_ID = -10000
_FIXTURE_OBSERVER_ID = 399  # Earth
_FIXTURE_FRAME = "J2000"
_FIXTURE_RADIUS_KM = 6678.0
_FIXTURE_OMEGA = 2.0 * np.pi / 5400.0
_FIXTURE_STEP = 60.0
_FIXTURE_N_STATES = 10
_FIXTURE_DEGREE = 7

# --- helpers ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_kernel_pool() -> Iterator[None]:
    """Ensure the SPICE kernel pool is empty before and after each test.

    The parser does its own kclear() bookkeeping; this fixture protects
    against bleed-through if a test directly calls spiceypy outside of
    parse(), which several do when constructing synthetic kernels.
    """
    spiceypy.kclear()
    try:
        yield
    finally:
        spiceypy.kclear()


def _make_single_segment_spk(
    path: Path,
    *,
    target: int = _FIXTURE_TARGET_ID,
    observer: int = _FIXTURE_OBSERVER_ID,
    frame: str = _FIXTURE_FRAME,
    n: int = _FIXTURE_N_STATES,
    degree: int = _FIXTURE_DEGREE,
    step: float = _FIXTURE_STEP,
    et0: float = 0.0,
    internal_name: str = "test SPK",
    segment_id: str = "synthetic LEO arc",
) -> Path:
    """Write a one-segment Type 13 SPK with a synthetic circular orbit."""
    if path.exists():
        path.unlink()
    ets = np.arange(n, dtype="float64") * step + et0
    states = np.zeros((n, 6), dtype="float64")
    for i, et in enumerate(ets):
        angle = _FIXTURE_OMEGA * et
        states[i, 0] = _FIXTURE_RADIUS_KM * np.cos(angle)
        states[i, 1] = _FIXTURE_RADIUS_KM * np.sin(angle)
        states[i, 3] = -_FIXTURE_RADIUS_KM * _FIXTURE_OMEGA * np.sin(angle)
        states[i, 4] = _FIXTURE_RADIUS_KM * _FIXTURE_OMEGA * np.cos(angle)
    handle = spiceypy.spkopn(str(path), internal_name, 4096)
    try:
        spiceypy.spkw13(
            handle,
            target,
            observer,
            frame,
            float(ets[0]),
            float(ets[-1]),
            segment_id,
            degree,
            n,
            states,
            ets,
        )
    finally:
        spiceypy.spkcls(handle)
    return path


# --- format detection -------------------------------------------------------


def test_is_spk_ephemeris_recognises_fixture() -> None:
    assert is_spk_ephemeris(_FIXTURE) is True


def test_is_spk_ephemeris_rejects_text(tmp_path: Path) -> None:
    text = tmp_path / "looks_like_oem.txt"
    text.write_text("CCSDS_OEM_VERS = 2.0\n", encoding="utf-8")
    assert is_spk_ephemeris(text) is False


def test_is_spk_ephemeris_rejects_short_file(tmp_path: Path) -> None:
    short = tmp_path / "short.bsp"
    short.write_bytes(b"DAF/SPK")  # missing trailing space
    assert is_spk_ephemeris(short) is False


def test_is_spk_ephemeris_returns_false_for_missing_file(tmp_path: Path) -> None:
    """A missing path returns False rather than raising — same as the STK sniffer."""
    assert is_spk_ephemeris(tmp_path / "does_not_exist.bsp") is False


# --- happy path: native mode (sampling_step=None) ---------------------------


def test_parses_expected_columns() -> None:
    df = parse(_FIXTURE)
    assert list(df.columns) == ["Epoch", "X", "Y", "Z", "VX", "VY", "VZ"]


def test_native_mode_returns_segment_endpoints() -> None:
    """One segment → start + stop, deduplicated."""
    df = parse(_FIXTURE)
    assert len(df) == 2


def test_state_columns_are_float64() -> None:
    df = parse(_FIXTURE)
    for column in ["X", "Y", "Z", "VX", "VY", "VZ"]:
        assert df[column].dtype == np.float64, column


def test_epoch_column_is_datetime64_ns() -> None:
    df = parse(_FIXTURE)
    assert df["Epoch"].dtype == np.dtype("datetime64[ns]")


def test_native_mode_round_trips_circular_orbit_radius() -> None:
    """Synthetic orbit is at constant radius 6678 km in the equatorial plane."""
    df = parse(_FIXTURE)
    radius = np.sqrt((df["X"] ** 2 + df["Y"] ** 2 + df["Z"] ** 2).to_numpy(dtype="float64"))
    assert radius[0] == pytest.approx(_FIXTURE_RADIUS_KM, rel=1e-9)
    assert radius[-1] == pytest.approx(_FIXTURE_RADIUS_KM, rel=1e-9)
    # Z and VZ are exactly zero in the equatorial test orbit.
    assert df["Z"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert df["VZ"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_native_mode_dedupes_butting_segments(tmp_path: Path) -> None:
    """Adjacent segments share endpoints; the parser must not emit duplicates."""
    path = tmp_path / "two_segments.bsp"
    if path.exists():
        path.unlink()
    n = _FIXTURE_N_STATES
    step = _FIXTURE_STEP
    ets1 = np.arange(n, dtype="float64") * step
    ets2 = np.arange(n, dtype="float64") * step + ets1[-1]  # starts where ets1 stops
    states1 = np.zeros((n, 6), dtype="float64")
    states2 = np.zeros((n, 6), dtype="float64")
    for i in range(n):
        a1 = _FIXTURE_OMEGA * ets1[i]
        a2 = _FIXTURE_OMEGA * ets2[i]
        states1[i] = [
            _FIXTURE_RADIUS_KM * np.cos(a1),
            _FIXTURE_RADIUS_KM * np.sin(a1),
            0.0,
            -_FIXTURE_RADIUS_KM * _FIXTURE_OMEGA * np.sin(a1),
            _FIXTURE_RADIUS_KM * _FIXTURE_OMEGA * np.cos(a1),
            0.0,
        ]
        states2[i] = [
            _FIXTURE_RADIUS_KM * np.cos(a2),
            _FIXTURE_RADIUS_KM * np.sin(a2),
            0.0,
            -_FIXTURE_RADIUS_KM * _FIXTURE_OMEGA * np.sin(a2),
            _FIXTURE_RADIUS_KM * _FIXTURE_OMEGA * np.cos(a2),
            0.0,
        ]
    handle = spiceypy.spkopn(str(path), "two-segment test", 4096)
    try:
        spiceypy.spkw13(
            handle,
            _FIXTURE_TARGET_ID,
            _FIXTURE_OBSERVER_ID,
            _FIXTURE_FRAME,
            float(ets1[0]),
            float(ets1[-1]),
            "seg1",
            _FIXTURE_DEGREE,
            n,
            states1,
            ets1,
        )
        spiceypy.spkw13(
            handle,
            _FIXTURE_TARGET_ID,
            _FIXTURE_OBSERVER_ID,
            _FIXTURE_FRAME,
            float(ets2[0]),
            float(ets2[-1]),
            "seg2",
            _FIXTURE_DEGREE,
            n,
            states2,
            ets2,
        )
    finally:
        spiceypy.spkcls(handle)
    df = parse(path)
    # 3 distinct endpoints: ets1[0], shared boundary (ets1[-1] == ets2[0]), ets2[-1].
    assert len(df) == 3
    assert df["Epoch"].is_monotonic_increasing


# --- happy path: sampled mode -----------------------------------------------


def test_sampled_mode_uses_step() -> None:
    """120s step over 540s coverage → start, 4 interior, stop."""
    df = parse(_FIXTURE, sampling_step=120.0)
    assert len(df) == 6
    epochs = df["Epoch"].astype("int64").to_numpy()
    diffs_seconds = np.diff(epochs) / 1e9
    # Five intervals between six rows. The first four are at the requested
    # 120s cadence; the last is whatever clamps to coverage_stop.
    assert diffs_seconds[:4] == pytest.approx([120.0, 120.0, 120.0, 120.0])


def test_sampled_mode_clamps_final_to_coverage_stop() -> None:
    """When sampling_step does not divide coverage evenly, last row pins to stop."""
    df = parse(_FIXTURE, sampling_step=120.0)
    # Coverage is [0, 540] s past J2000 TDB → stop is at the same UTC as
    # the last row of native mode.
    native = parse(_FIXTURE)
    assert df["Epoch"].iloc[-1] == native["Epoch"].iloc[-1]


def test_sampled_mode_rejects_non_positive_step() -> None:
    for bad in (0.0, -1.0, -0.001):
        with pytest.raises(GmatOutputParseError) as exc:
            parse(_FIXTURE, sampling_step=bad)
        assert "sampling_step" in str(exc.value)


def test_sampled_mode_step_smaller_than_coverage_dense_sampling() -> None:
    """A step that divides coverage evenly produces no extra clamp row."""
    # Coverage is 540 s; 60 s step → 10 samples (start + 9 steps).
    df = parse(_FIXTURE, sampling_step=60.0)
    assert len(df) == 10


# --- attrs ------------------------------------------------------------------


def test_attrs_epoch_scale_is_utc() -> None:
    df = parse(_FIXTURE)
    assert df.attrs["epoch_scales"] == {"Epoch": "UTC"}


def test_attrs_time_scale_records_source_as_tdb() -> None:
    """SPK is stored in TDB; the column is converted to UTC, attr records source."""
    df = parse(_FIXTURE)
    assert df.attrs["time_scale"] == "TDB"


def test_attrs_observer_resolves_to_earth() -> None:
    df = parse(_FIXTURE)
    assert df.attrs["observer_body"] == "EARTH"


def test_attrs_target_falls_back_to_id_string() -> None:
    """NAIF cannot resolve -10000 to a name; bodc2s returns the stringified ID."""
    df = parse(_FIXTURE)
    assert df.attrs["target_body"] == str(_FIXTURE_TARGET_ID)


def test_attrs_coordinate_system_is_frame_name() -> None:
    df = parse(_FIXTURE)
    assert df.attrs["coordinate_system"] == _FIXTURE_FRAME


def test_attrs_coverage_window() -> None:
    df = parse(_FIXTURE)
    coverage_seconds = (df.attrs["coverage_stop"] - df.attrs["coverage_start"]).total_seconds()
    assert coverage_seconds == pytest.approx((_FIXTURE_N_STATES - 1) * _FIXTURE_STEP, abs=1e-3)


def test_attrs_sampling_step_records_passed_value() -> None:
    """Attr should be the value passed in, not coerced to float."""
    assert parse(_FIXTURE).attrs["sampling_step"] is None
    assert parse(_FIXTURE, sampling_step=120.0).attrs["sampling_step"] == 120.0


def test_attrs_file_header() -> None:
    df = parse(_FIXTURE)
    fh = df.attrs["file_header"]
    assert fh["daf_id"] == "DAF/SPK"
    # _make_spk.py writes this exact internal name.
    assert fh["internal_filename"] == "gmat-run test SPK fixture"


# --- error paths ------------------------------------------------------------


def test_rejects_non_spk_file(tmp_path: Path) -> None:
    text = tmp_path / "fake.bsp"
    text.write_text("not an SPK", encoding="utf-8")
    with pytest.raises(GmatOutputParseError, match="DAF/SPK"):
        parse(text)


def test_rejects_multi_target_kernel(tmp_path: Path) -> None:
    """A kernel with two distinct (target, observer, frame) tuples is rejected."""
    path = tmp_path / "multi.bsp"
    if path.exists():
        path.unlink()
    n = _FIXTURE_N_STATES
    ets = np.arange(n, dtype="float64") * _FIXTURE_STEP
    states = np.zeros((n, 6), dtype="float64")
    for i, et in enumerate(ets):
        angle = _FIXTURE_OMEGA * et
        states[i, 0] = _FIXTURE_RADIUS_KM * np.cos(angle)
        states[i, 1] = _FIXTURE_RADIUS_KM * np.sin(angle)
        states[i, 3] = -_FIXTURE_RADIUS_KM * _FIXTURE_OMEGA * np.sin(angle)
        states[i, 4] = _FIXTURE_RADIUS_KM * _FIXTURE_OMEGA * np.cos(angle)
    handle = spiceypy.spkopn(str(path), "multi-target test", 4096)
    try:
        spiceypy.spkw13(
            handle,
            -10000,
            _FIXTURE_OBSERVER_ID,
            _FIXTURE_FRAME,
            float(ets[0]),
            float(ets[-1]),
            "sat A",
            _FIXTURE_DEGREE,
            n,
            states,
            ets,
        )
        spiceypy.spkw13(
            handle,
            -20000,
            _FIXTURE_OBSERVER_ID,
            _FIXTURE_FRAME,
            float(ets[0]),
            float(ets[-1]),
            "sat B",
            _FIXTURE_DEGREE,
            n,
            states,
            ets,
        )
    finally:
        spiceypy.spkcls(handle)
    with pytest.raises(GmatOutputParseError, match=r"multiple .* tuples"):
        parse(path)


def test_rejects_when_spiceypy_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the spiceypy import to verify the friendly error message."""
    spk_path = _make_single_segment_spk(tmp_path / "ok.bsp")

    # Block the import: drop the cached module, then make sys.modules
    # resolve to None so `import spiceypy` raises ImportError.
    monkeypatch.setitem(sys.modules, "spiceypy", None)
    # The parser caches spiceypy at the module level only inside parse();
    # importlib lookups will see sys.modules['spiceypy'] is None and raise.

    with pytest.raises(GmatOutputParseError, match=r"spiceypy.*\[spiceypy\]"):
        parse(spk_path)


# --- dispatch wiring sanity --------------------------------------------------


def test_dataframe_lengths_match_uniform_sampling_request() -> None:
    """Spot-check sampled-mode row count against the analytic expectation."""
    # Coverage 540s; step 90s → samples at 0, 90, 180, 270, 360, 450, plus
    # clamp at 540 → 7 rows.
    df = parse(_FIXTURE, sampling_step=90.0)
    assert len(df) == 7
