"""Regenerate the ``Ex_SPK.bsp`` fixture used by the SPK parser tests.

Run from a checkout root with ``spiceypy`` on the path::

    uv run python tests/fixtures/ephemeris/_make_spk.py

Output is a small DAF/SPK Type 13 (Hermite, unequal time steps) kernel
with one segment, a ten-state synthetic circular LEO trajectory, and the
NAIF body ID GMAT uses for arbitrary spacecraft. Matching the writer
GMAT actually produces (Type 13 at degree 7) keeps the fixture
representative without committing a full GMAT-emitted ``.bsp`` to the
repo.

Output values are deterministic — running this script a second time
should produce a byte-identical file.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import spiceypy as spice

# Synthetic spacecraft NAIF ID. GMAT writes spacecraft to negative IDs;
# -10000 is unassigned in NAIF's body lists and easy to spot in tests.
_TARGET_ID = -10000
# Earth as the central body matches GMAT's default for LEO ephemerides.
_OBSERVER_ID = 399
_FRAME = "J2000"
_DEGREE = 7
# Minimum sample count for a degree-7 Hermite is degree+1 = 8; 10 keeps
# us comfortably above the floor.
_N_STATES = 10
_STEP_SECONDS = 60.0
_OUT = Path(__file__).with_name("Ex_SPK.bsp")
_INTERNAL_NAME = "gmat-run test SPK fixture"
_SEGMENT_ID = "Ex_SPK synthetic LEO arc"


def _synthetic_states() -> tuple[np.ndarray, np.ndarray]:
    """Ten states on a circular LEO at 6678 km, 90-minute period.

    Deterministic; no random component. The orbit is in the equatorial
    plane so the Z and VZ components are exactly zero, which makes
    inspecting golden values by eye trivial.
    """
    omega = 2.0 * np.pi / 5400.0  # rad/s; ~90-min orbit
    radius = 6678.0  # km
    ets = np.arange(_N_STATES, dtype="float64") * _STEP_SECONDS
    states = np.zeros((_N_STATES, 6), dtype="float64")
    for index, et in enumerate(ets):
        angle = omega * et
        states[index, 0] = radius * np.cos(angle)
        states[index, 1] = radius * np.sin(angle)
        states[index, 3] = -radius * omega * np.sin(angle)
        states[index, 4] = radius * omega * np.cos(angle)
    return ets, states


def main() -> None:
    if _OUT.exists():
        _OUT.unlink()
    ets, states = _synthetic_states()
    handle = spice.spkopn(str(_OUT), _INTERNAL_NAME, 4096)
    try:
        spice.spkw13(
            handle,
            _TARGET_ID,
            _OBSERVER_ID,
            _FRAME,
            float(ets[0]),
            float(ets[-1]),
            _SEGMENT_ID,
            _DEGREE,
            _N_STATES,
            states,
            ets,
        )
    finally:
        spice.spkcls(handle)
    size = os.path.getsize(_OUT)
    print(f"wrote {_OUT} ({size} bytes, {_N_STATES} states, degree {_DEGREE})")


if __name__ == "__main__":
    main()
