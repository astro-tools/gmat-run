"""Parse a GMAT ``EphemerisFile`` (NASA SPICE binary kernel) into a DataFrame.

GMAT's third ephemeris writer (selected with
``EphemerisFile.FileFormat = SPK``) emits a NASA SPICE Spacecraft and Planet
Kernel: a DAF/SPK binary file with one or more segments holding interpolated
trajectories that SPICE can evaluate at any time within each segment's
coverage. The CCSDS-OEM and STK-TimePosVel text flavours are handled by
:mod:`gmat_run.parsers.ephemeris` and :mod:`gmat_run.parsers.stk_ephemeris`
respectively; CCSDS-AEM (attitude-only) is handled by
:mod:`gmat_run.parsers.aem_ephemeris`; Code-500 (binary) is out of scope.

Unlike the text formats, SPK does not store a fixed list of state records:
it carries Hermite (or Chebyshev) coefficients that SPICE evaluates on
demand. Callers pick between two output modes via the ``sampling_step``
parameter:

* ``sampling_step=None`` (default): one row per segment endpoint
  (``et_start`` and ``et_stop`` for every segment, deduplicated). This is
  honest to what the file actually stores at the kernel level — no
  interior interpolation.
* ``sampling_step=<seconds>``: uniform sampling of the union coverage
  window at the given cadence, with :func:`spiceypy.spkez` evaluated at
  each timestamp.

GMAT's SPK writer emits one spacecraft per file, so the parser assumes a
single ``(target, observer, frame)`` per kernel. Multi-target kernels —
e.g. JPL planetary ephemerides like DE440 — are rejected with a typed
error rather than collapsed into a long-format DataFrame.

Times in SPK are TDB seconds past J2000. The ``Epoch`` column is
converted to UTC via :func:`spiceypy.et2utc` against a vendored NAIF
leap-seconds kernel (``naif0012.tls`` under
:mod:`gmat_run.parsers._kernels`), matching the OEM/STK parsers' UTC
default. ``df.attrs["time_scale"] = "TDB"`` records the source scale so
callers who want the un-converted ET can recover it.

:mod:`spiceypy` is required at parse time. If the package is not
installed, :func:`parse` raises :class:`gmat_run.errors.GmatOutputParseError`
with a hint pointing at the ``[spiceypy]`` extra. :func:`is_spk_ephemeris`
is a pure-bytes magic check and works without spiceypy.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from gmat_run.errors import GmatOutputParseError

__all__ = ["is_spk_ephemeris", "parse"]

# DAF/SPK file-record magic. The leading 8 bytes of every DAF-encoded SPK
# file. ASCII, identical across native-endian variants.
_DAF_SPK_MAGIC: Final = b"DAF/SPK "

# The SPK summary layout NAIF guarantees for SPK files: 2 doubles
# (et_start, et_stop) and 6 ints (target, center, frame_id, type, baddr,
# eaddr). Captured as constants so the daffna/dafgs/dafus loop reads.
_SPK_ND: Final = 2
_SPK_NI: Final = 6

_STATE_COLUMNS: Final = ["X", "Y", "Z", "VX", "VY", "VZ"]

# UTC is the surfaced scale on the Epoch column; matches OEM and STK.
_TARGET_TIME_SCALE: Final = "UTC"

# TDB is the kernel's native scale; surfaced on attrs so callers know
# what was converted from.
_SOURCE_TIME_SCALE: Final = "TDB"

# Vendored leap-seconds kernel — required for the TDB → UTC conversion.
_LSK_FILENAME: Final = "naif0012.tls"

# et2utc("ISOC") format: "YYYY-MM-DDTHH:MM:SS.ffffff". Microsecond
# precision keeps the string under 27 chars and within datetime64[ns] range.
_ET2UTC_FORMAT: Final = "ISOC"
_ET2UTC_PRECISION: Final = 6
_ISOC_LEN: Final = 32

# spkez aberration-correction flag — geometric state, no light-time or
# stellar aberration. Matches what GMAT's writer produced.
_ABCORR: Final = "NONE"

# ifname strings GMAT writes have trailing whitespace padding in the DAF
# file record; strip it before surfacing.
_IFNAME_STRIP_RE: Final = re.compile(r"\s+$")


@dataclass(frozen=True)
class _Segment:
    """One DAF/SPK segment summary."""

    target_id: int
    observer_id: int
    frame_id: int
    frame_name: str
    et_start: float
    et_stop: float


def parse(
    path: str | os.PathLike[str],
    *,
    sampling_step: float | None = None,
) -> pd.DataFrame:
    """Parse an SPK ephemeris file into a :class:`pandas.DataFrame`.

    Args:
        path: Path to the ``.bsp`` file on disk.
        sampling_step: Output cadence in seconds. ``None`` (default)
            returns one row per segment endpoint; a positive float
            uniformly samples the union coverage window at that cadence.

    Returns:
        A DataFrame with one row per state. Columns are ``Epoch``
        (``datetime64[ns]``, UTC) plus ``X``, ``Y``, ``Z``, ``VX``,
        ``VY``, ``VZ`` (all ``float64``, kilometres / km·s⁻¹ — SPICE's
        native units, matching the OEM/STK parsers' km defaults).

        Metadata surfaces on ``df.attrs``:

        * ``epoch_scales = {"Epoch": "UTC"}`` — see module docstring.
        * ``time_scale = "TDB"`` — the kernel's native scale; the
          ``Epoch`` column is converted from this to UTC.
        * ``target_body``, ``observer_body`` — common names where SPICE
          can resolve them, otherwise the integer NAIF IDs as strings.
        * ``coordinate_system`` — the segment's reference frame
          (e.g. ``"J2000"``).
        * ``coverage_start``, ``coverage_stop`` — UTC ``Timestamp``s
          spanning the kernel's union coverage.
        * ``sampling_step`` — the value passed in (``None`` or float).
        * ``file_header = {"daf_id": "DAF/SPK", "internal_filename": "..."}``.

    Raises:
        GmatOutputParseError: ``spiceypy`` is not installed; the file is
            unreadable or not a DAF/SPK kernel; the kernel has no
            segments; the kernel mixes ``(target, observer, frame)``
            tuples; or ``sampling_step`` is non-positive.
    """
    path = Path(path)

    if sampling_step is not None and not (sampling_step > 0):
        raise GmatOutputParseError(
            f"sampling_step must be positive, got {sampling_step!r}",
            path,
        )

    spice = _import_spiceypy(path)

    if not is_spk_ephemeris(path):
        raise GmatOutputParseError(
            "file is not a DAF/SPK kernel (magic mismatch); not an SPK ephemeris?",
            path,
        )

    # SPICE keeps a single global kernel pool. We bracket the parse with
    # kclear() to leave that pool the way we found it — back-to-back
    # parse() calls in one process must not see each other's kernels.
    spice.kclear()
    try:
        with as_file(files("gmat_run.parsers._kernels").joinpath(_LSK_FILENAME)) as lsk:
            spice.furnsh(str(lsk))
            spice.furnsh(str(path))

            segments = _enumerate_segments(spice, path)
            if not segments:
                raise GmatOutputParseError(
                    "DAF/SPK kernel contains no segments",
                    path,
                )
            target_id, observer_id, frame_name = _validate_uniform(segments, path)

            cov_start, cov_stop = _union_coverage(segments)

            if sampling_step is None:
                ets = _segment_endpoint_times(segments)
            else:
                ets = _uniform_sample_times(cov_start, cov_stop, sampling_step)

            states = _evaluate_states(spice, ets, target_id, frame_name, observer_id)
            epochs = _ets_to_utc(spice, ets)

            df = pd.DataFrame({"Epoch": epochs})
            for index, column in enumerate(_STATE_COLUMNS):
                df[column] = states[:, index].astype("float64")

            df.attrs["epoch_scales"] = {"Epoch": _TARGET_TIME_SCALE}
            df.attrs["time_scale"] = _SOURCE_TIME_SCALE
            df.attrs["target_body"] = _body_name(spice, target_id)
            df.attrs["observer_body"] = _body_name(spice, observer_id)
            df.attrs["coordinate_system"] = frame_name
            df.attrs["coverage_start"] = _ets_to_utc(spice, np.array([cov_start]))[0]
            df.attrs["coverage_stop"] = _ets_to_utc(spice, np.array([cov_stop]))[0]
            df.attrs["sampling_step"] = sampling_step
            df.attrs["file_header"] = _file_header(spice, path)
            return df
    finally:
        spice.kclear()


# --- format detection -------------------------------------------------------


def is_spk_ephemeris(path: str | os.PathLike[str]) -> bool:
    """Return ``True`` if ``path`` looks like a DAF/SPK kernel.

    Reads the first eight bytes and matches the ASCII ``"DAF/SPK "``
    magic. Used by :mod:`gmat_run.results` to dispatch by content rather
    than by extension. Pure-bytes — does not import ``spiceypy``.
    """
    try:
        with Path(path).open("rb") as fh:
            magic = fh.read(len(_DAF_SPK_MAGIC))
    except OSError:
        return False
    return magic == _DAF_SPK_MAGIC


# --- spiceypy import guard --------------------------------------------------


def _import_spiceypy(path: Path) -> Any:
    """Import :mod:`spiceypy` lazily, with a friendly error on failure.

    SPK parsing pulls in the CSPICE shared library, which is a heavy
    optional dependency. Importing here (rather than at module top) keeps
    ``import gmat_run.parsers.spk`` cheap for callers who only want the
    pure-bytes :func:`is_spk_ephemeris` check.
    """
    try:
        import spiceypy
    except ImportError as exc:
        raise GmatOutputParseError(
            "SPK parsing requires the 'spiceypy' extra: pip install gmat-run[spiceypy]",
            path,
        ) from exc
    return spiceypy


# --- segment enumeration & validation ---------------------------------------


def _enumerate_segments(spice: Any, path: Path) -> list[_Segment]:
    """Walk a DAF/SPK file and return one summary per segment.

    Uses the low-level DAF API (``dafopr`` / ``dafbfs`` / ``daffna`` /
    ``dafgs`` / ``dafus``) rather than the higher-level ``spkobj`` /
    ``spkcov`` pair because the latter does not expose per-segment
    observer / frame metadata.
    """
    handle = spice.dafopr(str(path))
    try:
        spice.dafbfs(handle)
        segments: list[_Segment] = []
        while spice.daffna():
            packed = spice.dafgs()
            dc, ic = spice.dafus(packed, _SPK_ND, _SPK_NI)
            target_id = int(ic[0])
            observer_id = int(ic[1])
            frame_id = int(ic[2])
            try:
                frame_name = spice.frmnam(frame_id)
            except Exception:
                # spiceypy raises a SpiceyError subclass on unknown IDs;
                # catch broadly to avoid taking on a typed import dependency.
                frame_name = ""
            if not frame_name:
                # frmnam returns an empty string for IDs the kernel pool
                # cannot resolve. Fall back to the numeric ID so the
                # surfaced metadata is at least round-trippable.
                frame_name = f"FRAME_{frame_id}"
            segments.append(
                _Segment(
                    target_id=target_id,
                    observer_id=observer_id,
                    frame_id=frame_id,
                    frame_name=frame_name,
                    et_start=float(dc[0]),
                    et_stop=float(dc[1]),
                )
            )
    finally:
        spice.dafcls(handle)
    return segments


def _validate_uniform(segments: list[_Segment], path: Path) -> tuple[int, int, str]:
    """Reject SPK files that mix targets, observers, or frames.

    GMAT's SPK writer emits one ``(target, observer, frame)`` per file.
    Generic multi-target kernels (planetary ephemerides, etc.) are out of
    scope — see issue #49. Surfacing the mismatch with the offending
    tuples is more useful than collapsing the kernel into a long-format
    DataFrame.
    """
    tuples = {(s.target_id, s.observer_id, s.frame_name) for s in segments}
    if len(tuples) > 1:
        rendered = ", ".join(
            f"(target={t}, observer={o}, frame={f!r})" for t, o, f in sorted(tuples)
        )
        raise GmatOutputParseError(
            "DAF/SPK kernel contains multiple (target, observer, frame) "
            f"tuples; only single-spacecraft kernels are supported: {rendered}",
            path,
        )
    sole = next(iter(tuples))
    return sole[0], sole[1], sole[2]


# --- time arrays ------------------------------------------------------------


def _union_coverage(segments: list[_Segment]) -> tuple[float, float]:
    """Earliest start and latest stop across every segment, in TDB ET seconds."""
    start = min(s.et_start for s in segments)
    stop = max(s.et_stop for s in segments)
    return start, stop


def _segment_endpoint_times(segments: list[_Segment]) -> np.ndarray:
    """Return one ET per segment endpoint, sorted and deduplicated.

    Segments commonly butt against each other (segment N's stop equals
    segment N+1's start); deduplicating avoids a near-zero-cadence row
    that confuses downstream consumers.
    """
    raw: list[float] = []
    for segment in segments:
        raw.append(segment.et_start)
        raw.append(segment.et_stop)
    return np.array(sorted(set(raw)), dtype="float64")


def _uniform_sample_times(cov_start: float, cov_stop: float, sampling_step: float) -> np.ndarray:
    """Uniformly sampled ETs spanning ``[cov_start, cov_stop]`` inclusive.

    The final timestamp is clamped to ``cov_stop`` so the caller always
    sees the kernel's full coverage end, even when ``sampling_step`` does
    not divide the coverage span evenly.
    """
    if cov_stop <= cov_start:
        return np.array([cov_start], dtype="float64")
    n = int(np.floor((cov_stop - cov_start) / sampling_step)) + 1
    samples = cov_start + sampling_step * np.arange(n, dtype="float64")
    if samples[-1] < cov_stop:
        samples = np.append(samples, cov_stop)
    return samples


def _evaluate_states(
    spice: Any,
    ets: np.ndarray,
    target_id: int,
    frame_name: str,
    observer_id: int,
) -> np.ndarray:
    """Evaluate position+velocity at every ET via :func:`spiceypy.spkez`.

    spiceypy's ``spkez`` is single-epoch — there is no batched call in
    the public API — so we loop in Python. The cost is dominated by the
    underlying CSPICE evaluation, not the loop overhead.
    """
    out = np.empty((len(ets), 6), dtype="float64")
    for i, et in enumerate(ets):
        state, _light_time = spice.spkez(target_id, float(et), frame_name, _ABCORR, observer_id)
        out[i] = state
    return out


def _ets_to_utc(spice: Any, ets: np.ndarray) -> pd.Series:
    """Convert TDB ETs to UTC ``datetime64[ns]`` via :func:`spiceypy.et2utc`.

    ``et2utc`` accepts an iterable input and returns a numpy array of
    fixed-byte ISO-8601 strings (``"YYYY-MM-DDTHH:MM:SS.ffffff"``); we
    parse those strings into ``datetime64[ns]`` with pandas to align
    with the OEM and STK parsers' Epoch dtype.
    """
    if len(ets) == 0:
        return pd.Series(dtype="datetime64[ns]")
    raw = spice.et2utc(ets, _ET2UTC_FORMAT, _ET2UTC_PRECISION, _ISOC_LEN)
    # spiceypy returns a numpy array of bytes (older builds) or strs
    # (newer builds); coerce both to a Python list before parsing.
    if isinstance(raw, np.ndarray):
        decoded = [s.decode("ascii") if isinstance(s, bytes) else str(s) for s in raw]
    else:
        decoded = [raw if isinstance(raw, str) else raw.decode("ascii")]
    return pd.Series(pd.to_datetime(decoded, format="ISO8601")).astype("datetime64[ns]")


# --- metadata helpers -------------------------------------------------------


def _body_name(spice: Any, code: int) -> str:
    """NAIF body name for ``code``, falling back to the stringified ID.

    ``bodc2s`` always returns a string — either the resolved name or the
    numeric ID rendered as text — so this is a thin wrapper that just
    pins the result type for the type checker.
    """
    return str(spice.bodc2s(code))


def _file_header(spice: Any, path: Path) -> dict[str, str]:
    """Extract the DAF file-record metadata GMAT-aware callers care about.

    ``dafrfr`` returns the internal filename (a 60-char text field GMAT's
    writer fills with the spacecraft name and a timestamp) and the
    pointer fields needed to traverse the DAF. We surface only ``daf_id``
    (the magic) and ``internal_filename`` for parity with the OEM/STK
    parsers' ``file_header`` shape.
    """
    handle = spice.dafopr(str(path))
    try:
        # dafrfr → (nd, ni, ifname, fward, bward, free)
        _nd, _ni, ifname, *_rest = spice.dafrfr(handle)
    finally:
        spice.dafcls(handle)
    return {
        "daf_id": _DAF_SPK_MAGIC.decode("ascii").strip(),
        "internal_filename": _IFNAME_STRIP_RE.sub("", str(ifname)),
    }
