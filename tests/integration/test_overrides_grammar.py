"""Integration coverage for the v0.5 override-grammar extensions.

Two flavours added to ``Mission.__setitem__``:

* ``Resource.SubResource.Field`` (multi-dot) — exercised via
  ``FM.Drag.CSSISpaceWeatherFile`` against a drag-enabled LEO fixture.
* ``Variable.Value`` — exercised against a fixture that declares
  ``Create Variable`` and propagates for the variable's value.

The fixtures live next to the existing minimal-LEO smoke fixture so the
test is independent of which stock samples ship with a given GMAT
release.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from gmat_run import Mission
from gmat_run.errors import GmatFieldError

pytestmark = pytest.mark.integration


_DRAG_FIXTURE = Path(__file__).parent / "fixtures" / "Ex_DragLEO.script"
_VAR_FIXTURE = Path(__file__).parent / "fixtures" / "Ex_VariableOverride.script"


@pytest.fixture
def drag_script(tmp_path: Path) -> Path:
    dst = tmp_path / "drag_leo.script"
    shutil.copyfile(_DRAG_FIXTURE, dst)
    return dst


@pytest.fixture
def variable_script(tmp_path: Path) -> Path:
    dst = tmp_path / "variable.script"
    shutil.copyfile(_VAR_FIXTURE, dst)
    return dst


def test_multi_dot_override_round_trips_through_real_gmat(
    gmat_available: None,
    drag_script: Path,
) -> None:
    """Write to ``FM.Drag.CSSISpaceWeatherFile`` and read it back.

    The motivating use case is ``paper-tle-divergence-atlas``'s ~24k
    Starlink propagations swapping the weather file per run. Before v0.5
    the paper carried a regex helper that rewrote the script text; this
    pin is the public contract that lets that helper retire.
    """
    target = "/tmp/test_cssi.txt"
    mission = Mission.load(drag_script)

    mission["FM.Drag.CSSISpaceWeatherFile"] = target

    assert mission["FM.Drag.CSSISpaceWeatherFile"] == target
    # AtmosphereModel reads through the same sub-resource path too — proves
    # the dotted-tail routing isn't specific to one field.
    assert mission["FM.Drag.AtmosphereModel"] == "JacchiaRoberts"


def test_multi_dot_unknown_field_raises_typed_error(
    gmat_available: None,
    drag_script: Path,
) -> None:
    """A sub-resource path with a wrong leaf surfaces ``GmatFieldError``.

    The error preserves the dotted tail in the message so a caller spots
    whether the typo is in the parent or the leaf without re-parsing the
    underlying GMAT message.
    """
    mission = Mission.load(drag_script)
    with pytest.raises(GmatFieldError) as excinfo:
        _ = mission["FM.Drag.NotARealField"]
    assert "unknown field 'Drag.NotARealField'" in str(excinfo.value)


def test_variable_value_override_takes_effect_at_runtime(
    gmat_available: None,
    variable_script: Path,
) -> None:
    """``Variable.Value`` is the documented contract for script-Variable overrides.

    The fixture script propagates for ``propagate_seconds`` seconds, where
    that bound is itself a ``Create Variable`` block. Overriding the
    variable before run() shortens the propagation; the captured
    ReportFile column for the variable carries the override value.
    """
    overridden = 90.0  # seconds — well under the script's default of 600
    mission = Mission.load(variable_script)
    mission["propagate_seconds.Value"] = overridden

    # Read-back: the override is visible on the live Variable.
    assert mission["propagate_seconds.Value"] == overridden

    result = mission.run()
    df = result.reports["RF"]
    assert isinstance(df, pd.DataFrame)
    # The Variable column tracks the overridden value across the run.
    assert df["propagate_seconds"].iloc[0] == pytest.approx(overridden)
    assert df["propagate_seconds"].iloc[-1] == pytest.approx(overridden)
