"""Emit a CCSDS-OEM file from an ephemeris :class:`pandas.DataFrame`.

Round-trip companion to :mod:`gmat_run.parsers.ephemeris`. The DataFrame shape
expected here is exactly the one that parser produces — columns
``Epoch`` plus ``X``, ``Y``, ``Z``, ``VX``, ``VY``, ``VZ``, with the source
metadata surfaced on ``df.attrs`` (``coordinate_system``, ``central_body``,
``time_scale`` / ``epoch_scales``, ``object_name``, ``interpolation``,
``interpolation_degree``).

The writer is gated behind the ``[ccsds-ndm]`` extra; importing this module
without ``ccsds-ndm`` installed surfaces an :class:`ImportError` with a hint
the moment :func:`write_oem` is called.

Multi-segment OEM output is deliberately not supported — every call emits a
single segment containing every row of the DataFrame in file order.
``df.attrs["segments"]`` is ignored. Covariance blocks are also out of scope.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any, Final

import pandas as pd

__all__ = ["write_oem"]

# CCSDS 502.0-B-2 §A.5 reference frames the writer accepts as canonical.
# Anything in this set passes through to ``REF_FRAME`` unchanged. The set is
# intentionally narrow — additions should be made when a real GMAT output
# surfaces them, not speculatively.
_CCSDS_FRAMES: Final = frozenset(
    {
        "EME2000",
        "GCRF",
        "GRC",
        "ICRF",
        "ITRF-93",
        "ITRF-97",
        "ITRF2000",
        "MCI",
        "TDR",
        "TEME",
        "TOD",
        "EFG",
        "GTOD",
    }
)

# Aliases from the names other parsers / GMAT internals surface to the CCSDS
# canonical name. The right-hand side must be a member of ``_CCSDS_FRAMES``.
_FRAME_ALIASES: Final = {
    # GMAT-internal coordinate-system names — the OEM text writer normally
    # already translates these, but a user feeding a hand-edited or non-OEM
    # DataFrame may still see them.
    "EarthMJ2000Eq": "EME2000",
    "EarthICRF": "ICRF",
    # STK's ``CoordinateSystem`` token for inertial J2000.
    "J2000": "EME2000",
}

# Time scales we accept on output. CCSDS-OEM permits more (UT1, GPS, …) but
# the writer is scoped to the five GMAT recognises end-to-end so a typo in
# ``df.attrs`` surfaces here instead of inside a downstream consumer.
_ALLOWED_TIME_SCALES: Final = frozenset({"A1", "TAI", "UTC", "TT", "TDB"})


def write_oem(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    originator: str = "gmat-run",
    object_name: str | None = None,
) -> Path:
    """Write ``df`` to ``path`` as a CCSDS-OEM (KVN) file.

    Args:
        df: Ephemeris DataFrame in the shape produced by
            :func:`gmat_run.parsers.ephemeris.parse` — columns ``Epoch``,
            ``X``, ``Y``, ``Z``, ``VX``, ``VY``, ``VZ`` and the metadata
            attrs the parser surfaces (``coordinate_system``,
            ``central_body``, ``time_scale`` / ``epoch_scales``, optionally
            ``object_name``, ``interpolation``, ``interpolation_degree``).
        path: Destination ``.oem`` file. Parent directories are created.
        originator: Value for the ``ORIGINATOR`` header field. Defaults to
            ``"gmat-run"`` (the file header from the source ephemeris is
            *not* preserved — the new file is a new artefact).
        object_name: Override for the ``OBJECT_NAME`` meta field. When
            ``None``, falls back to ``df.attrs["object_name"]``, then to
            ``"UNKNOWN"`` if neither is set.

    Returns:
        The destination ``Path``.

    Raises:
        ImportError: ``ccsds-ndm`` is not installed.
        ValueError: a required attr (``coordinate_system``, ``central_body``,
            time scale) is missing, the time scale is not one of
            ``A1`` / ``TAI`` / ``UTC`` / ``TT`` / ``TDB``, or the
            coordinate system is neither a CCSDS canonical name nor a known
            alias (see the module-level frame mapping table).
    """
    try:
        from ccsds_ndm.models.ndmxml4 import (
            OdmHeader,
            Oem,
            OemBody,
            OemData,
            OemMetadata,
            OemSegment,
            PositionTypeUo,
            PositionUnits,
            StateVectorAccType,
            VelocityTypeUo,
            VelocityUnits,
        )
        from ccsds_ndm.ndm_kvn_io import NdmKvnIo
    except ImportError as exc:
        raise ImportError(
            "Results.write_oem requires the [ccsds-ndm] extra: pip install gmat-run[ccsds-ndm]"
        ) from exc

    ref_frame = _resolve_frame(df.attrs.get("coordinate_system"))
    central_body = _require_attr(df, "central_body")
    time_system = _resolve_time_scale(df)
    object_name_resolved = object_name or df.attrs.get("object_name") or "UNKNOWN"
    object_id = df.attrs.get("object_id") or object_name_resolved

    if not len(df):
        raise ValueError("cannot write an OEM from an empty DataFrame")

    epochs = [_format_epoch(ts) for ts in df["Epoch"]]
    xs = df["X"].to_numpy(dtype=float)
    ys = df["Y"].to_numpy(dtype=float)
    zs = df["Z"].to_numpy(dtype=float)
    vxs = df["VX"].to_numpy(dtype=float)
    vys = df["VY"].to_numpy(dtype=float)
    vzs = df["VZ"].to_numpy(dtype=float)
    state_vectors = [
        StateVectorAccType(
            epoch=epoch,
            x=PositionTypeUo(value=float(xs[i]), units=PositionUnits.KM),
            y=PositionTypeUo(value=float(ys[i]), units=PositionUnits.KM),
            z=PositionTypeUo(value=float(zs[i]), units=PositionUnits.KM),
            x_dot=VelocityTypeUo(value=float(vxs[i]), units=VelocityUnits.KM_S),
            y_dot=VelocityTypeUo(value=float(vys[i]), units=VelocityUnits.KM_S),
            z_dot=VelocityTypeUo(value=float(vzs[i]), units=VelocityUnits.KM_S),
            x_ddot=None,
            y_ddot=None,
            z_ddot=None,
        )
        for i, epoch in enumerate(epochs)
    ]

    interpolation = df.attrs.get("interpolation")
    interpolation_degree = df.attrs.get("interpolation_degree")

    metadata = OemMetadata(
        comment=[],
        object_name=str(object_name_resolved),
        object_id=str(object_id),
        center_name=str(central_body),
        ref_frame=ref_frame,
        ref_frame_epoch=None,
        time_system=time_system,
        start_time=epochs[0],
        useable_start_time=None,
        useable_stop_time=None,
        stop_time=epochs[-1],
        interpolation=str(interpolation) if interpolation is not None else None,
        interpolation_degree=(
            int(interpolation_degree) if interpolation_degree is not None else None
        ),
    )
    oem = Oem(
        header=OdmHeader(
            comment=[],
            classification=None,
            creation_date=_now_iso(),
            originator=originator,
            message_id=None,
        ),
        body=OemBody(
            segment=[
                OemSegment(
                    metadata=metadata,
                    data=OemData(comment=[], state_vector=state_vectors, covariance_matrix=[]),
                )
            ]
        ),
    )
    # GMAT R2026a's CCSDS-OEM reader is strict about the version line:
    # ``3.0`` (the ccsds-ndm default) is rejected outright; ``2.0`` is gated
    # behind a "TESTING mode" runtime flag and unusable in production; only
    # ``1.0`` is accepted as a propagator input. ccsds-ndm 3.x can't write
    # ``1.0`` directly (its mapping table registers v2/v3 only), so we let
    # it serialise as ``2.0`` and rewrite the version line in the emitted
    # text. The KVN payload that follows is identical between v1.0 and v2.0
    # for single-segment, covariance-free OEMs (the spec changes between
    # those versions are confined to optional features the writer does not
    # emit).
    oem.version = "2.0"

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    NdmKvnIo().to_file(oem, dest)
    _rewrite_oem_version_line(dest)
    return dest


def _rewrite_oem_version_line(path: Path) -> None:
    """Patch ``CCSDS_OEM_VERS = 2.0`` → ``= 1.0`` in place. See ``write_oem``."""
    text = path.read_text(encoding="utf-8")
    patched = text.replace(
        "CCSDS_OEM_VERS           = 2.0",
        "CCSDS_OEM_VERS           = 1.0",
        1,
    )
    if patched == text:
        raise RuntimeError(
            "ccsds-ndm did not emit the expected 'CCSDS_OEM_VERS = 2.0' header; "
            "the writer can no longer guarantee GMAT-compatible output"
        )
    path.write_text(patched, encoding="utf-8")


def _resolve_frame(value: Any) -> str:
    if value is None:
        raise ValueError(
            "df.attrs['coordinate_system'] is required to write an OEM "
            "(set it to a CCSDS-OEM frame name or a recognised GMAT alias)"
        )
    name = str(value)
    if name in _CCSDS_FRAMES:
        return name
    if name in _FRAME_ALIASES:
        return _FRAME_ALIASES[name]
    canonical = sorted(_CCSDS_FRAMES)
    aliases = sorted(_FRAME_ALIASES)
    raise ValueError(
        f"unknown coordinate system {name!r}; supply a CCSDS-OEM frame "
        f"({', '.join(canonical)}) or a known alias ({', '.join(aliases)})"
    )


def _resolve_time_scale(df: pd.DataFrame) -> str:
    epoch_scales = df.attrs.get("epoch_scales")
    if isinstance(epoch_scales, dict) and "Epoch" in epoch_scales:
        scale = str(epoch_scales["Epoch"])
    elif "time_scale" in df.attrs:
        scale = str(df.attrs["time_scale"])
    else:
        raise ValueError(
            "df.attrs is missing the time scale "
            "(set df.attrs['epoch_scales'] = {'Epoch': '<scale>'} "
            "or df.attrs['time_scale'] = '<scale>')"
        )
    if scale not in _ALLOWED_TIME_SCALES:
        raise ValueError(
            f"unsupported TIME_SYSTEM {scale!r}; expected one of {sorted(_ALLOWED_TIME_SCALES)}"
        )
    return scale


def _require_attr(df: pd.DataFrame, key: str) -> Any:
    if key not in df.attrs:
        raise ValueError(f"df.attrs[{key!r}] is required to write an OEM")
    return df.attrs[key]


def _format_epoch(ts: Any) -> str:
    """ISO-8601 with millisecond precision, matching GMAT's stock OEM format."""
    return pd.Timestamp(ts).isoformat(timespec="milliseconds")


def _now_iso() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")
    )
