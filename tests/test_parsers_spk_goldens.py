"""Golden-file tests for :func:`gmat_run.parsers.spk.parse`.

Pins the structure of the DataFrame the parser produces from the
committed ``Ex_SPK.bsp`` fixture. The fixture is a synthetic Type 13
(Hermite) kernel built by ``tests/fixtures/ephemeris/_make_spk.py`` —
ten states on a circular equatorial-plane orbit at radius 6678 km,
60-second cadence, J2000 frame. Regenerate by running the
``_make_spk.py`` script.

The synthetic orbit is chosen so the analytic state at any ET is
``X = R cos(omega t)``, ``Y = R sin(omega t)``, ``Z = 0``,
``VX = -R omega sin(omega t)``, ``VY = R omega cos(omega t)``,
``VZ = 0`` — letting the assertions below check absolute values rather
than relying on opaque saved-from-disk numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# spiceypy is required by the parser; importorskip lets `uv run pytest`
# survive cleanly on a contributor box that has not synced the dev group.
pytest.importorskip("spiceypy")

from gmat_run.parsers.spk import parse

FIXTURES = Path(__file__).parent / "fixtures" / "ephemeris"


class TestExSPK:
    """``Ex_SPK.bsp`` -> single-segment Type 13 LEO arc, 60 s cadence."""

    FIXTURE = FIXTURES / "Ex_SPK.bsp"
    RADIUS_KM = 6678.0
    OMEGA = 2.0 * np.pi / 5400.0
    STEP = 60.0
    N_STATES = 10
    # Coverage in TDB ET starts at 0 (J2000); first row in UTC trails by
    # leap seconds + TDB-TT periodic correction. Pin the exact moment via
    # the parser's own coverage attr so any conversion drift is caught.
    EXPECTED_COVERAGE_SECONDS = (N_STATES - 1) * STEP

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return parse(self.FIXTURE)

    @pytest.fixture(scope="class")
    def df_sampled(self) -> pd.DataFrame:
        return parse(self.FIXTURE, sampling_step=120.0)

    def test_columns(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == ["Epoch", "X", "Y", "Z", "VX", "VY", "VZ"]

    def test_native_mode_row_count(self, df: pd.DataFrame) -> None:
        # Single segment → start + stop, deduplicated.
        assert len(df) == 2

    def test_sampled_mode_row_count(self, df_sampled: pd.DataFrame) -> None:
        # Coverage is 540 s; 120 s step → 5 cadenced rows + clamp at stop = 6.
        assert len(df_sampled) == 6

    def test_epoch_dtype_and_scale(self, df: pd.DataFrame) -> None:
        assert df["Epoch"].dtype == np.dtype("datetime64[ns]")
        assert df.attrs["epoch_scales"] == {"Epoch": "UTC"}

    def test_state_columns_are_float64(self, df: pd.DataFrame) -> None:
        for column in ["X", "Y", "Z", "VX", "VY", "VZ"]:
            assert df[column].dtype == np.float64, column

    def test_first_record_matches_analytic_state(self, df: pd.DataFrame) -> None:
        # ET 0 (J2000), so angle = 0 and the synthetic orbit puts the
        # spacecraft on +X with +Y velocity.
        assert df["X"].iloc[0] == pytest.approx(self.RADIUS_KM, rel=1e-12)
        assert df["Y"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert df["Z"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert df["VX"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert df["VY"].iloc[0] == pytest.approx(self.RADIUS_KM * self.OMEGA, rel=1e-9)
        assert df["VZ"].iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_last_record_matches_analytic_state(self, df: pd.DataFrame) -> None:
        # ET = 540 s past J2000.
        angle = self.OMEGA * self.EXPECTED_COVERAGE_SECONDS
        assert df["X"].iloc[-1] == pytest.approx(self.RADIUS_KM * np.cos(angle), rel=1e-9)
        assert df["Y"].iloc[-1] == pytest.approx(self.RADIUS_KM * np.sin(angle), rel=1e-9)
        assert df["Z"].iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_circular_orbit_radius_is_constant_across_rows(self, df_sampled: pd.DataFrame) -> None:
        radius = np.sqrt(df_sampled["X"] ** 2 + df_sampled["Y"] ** 2 + df_sampled["Z"] ** 2)
        # Hermite interpolation is exact on the sample points and tiny
        # off them; pin the constant-radius invariant to within 1e-6 km.
        assert np.allclose(radius, self.RADIUS_KM, atol=1e-6)

    def test_metadata_attrs(self, df: pd.DataFrame) -> None:
        assert df.attrs["coordinate_system"] == "J2000"
        # Synthetic spacecraft NAIF ID has no name in NAIF's body lists.
        assert df.attrs["target_body"] == "-10000"
        assert df.attrs["observer_body"] == "EARTH"
        assert df.attrs["time_scale"] == "TDB"
        assert df.attrs["sampling_step"] is None

    def test_coverage_window_attrs(self, df: pd.DataFrame) -> None:
        delta = df.attrs["coverage_stop"] - df.attrs["coverage_start"]
        assert delta.total_seconds() == pytest.approx(self.EXPECTED_COVERAGE_SECONDS, abs=1e-3)

    def test_file_header_attrs(self, df: pd.DataFrame) -> None:
        fh = df.attrs["file_header"]
        assert fh["daf_id"] == "DAF/SPK"
        # _make_spk.py writes this exact internal name; a future
        # regeneration that changes it should fail this assertion.
        assert fh["internal_filename"] == "gmat-run test SPK fixture"
