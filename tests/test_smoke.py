"""Smoke tests — the package imports and exposes its version."""

import gmat_run


def test_import() -> None:
    assert gmat_run.__version__ == "0.0.0"
