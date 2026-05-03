"""Standalone smoke runner for the ``canonical-image-smoke`` CI cell.

The cell installs gmat-run with no extras (mirroring ``minimal-install``) and
runs this script inside ``ghcr.io/astro-tools/gmat:Rxxxxa``. Because the cell
has no dev dependencies, this is a plain ``python <script>`` runnable rather
than a pytest test — pytest is not on the install path here.

What it catches: drift between the canonical image and what gmat-run expects.
A missing propagator integrator, a mangled ``gmat_startup_file.txt``, a
dropped Python ABI from the bundled gmatpy set, or a tag that silently moves
to a different GMAT upload all break the load → run → DataFrame round-trip
this script exercises.

Runnable locally for debugging against any GMAT install that
``gmat_run.locate_gmat`` can find::

    python tests/canonical_image_smoke.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from gmat_run import Mission

FIXTURE = Path(__file__).parent / "integration" / "fixtures" / "Ex_MinimalLEO.script"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "minimal_leo.script"
        shutil.copyfile(FIXTURE, script)

        mission = Mission.load(script)
        result = mission.run()

        df = result.reports["RF"]
        assert isinstance(df, pd.DataFrame), f"expected DataFrame, got {type(df)!r}"
        assert not df.empty, "ReportFile DataFrame is empty"
        assert "Sat.SMA" in df.columns, f"missing Sat.SMA column; got {list(df.columns)}"

        print(f"OK: ReportFile 'RF' parsed as {len(df)}-row DataFrame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
