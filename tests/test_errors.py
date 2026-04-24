"""Unit tests for gmat_run.errors.

Raise-site tests for ``GmatNotFoundError`` and ``GmatLoadError`` live with
their respective entry points in ``test_install.py`` and ``test_runtime.py``.
Raise-site tests for the remaining three classes will be added alongside
their entry points (mission / parsers / field access), which are separate
issues. These tests cover the hierarchy shape and the attribute contract.
"""

from pathlib import Path

import pytest

from gmat_run.errors import (
    GmatError,
    GmatFieldError,
    GmatLoadError,
    GmatNotFoundError,
    GmatOutputParseError,
    GmatRunError,
)

_CONCRETE_CLASSES = [
    GmatNotFoundError,
    GmatLoadError,
    GmatRunError,
    GmatOutputParseError,
    GmatFieldError,
]


# --- hierarchy invariants -----------------------------------------------------


def test_base_is_exception_subclass() -> None:
    assert issubclass(GmatError, Exception)


@pytest.mark.parametrize("cls", _CONCRETE_CLASSES)
def test_every_concrete_class_subclasses_gmat_error(cls: type[GmatError]) -> None:
    assert issubclass(cls, GmatError)
    assert issubclass(cls, Exception)


@pytest.mark.parametrize("cls", [GmatError, *_CONCRETE_CLASSES])
def test_every_class_has_a_docstring(cls: type[GmatError]) -> None:
    assert cls.__doc__ is not None
    assert cls.__doc__.strip()


# --- GmatRunError -------------------------------------------------------------


def test_gmat_run_error_preserves_log_attribute() -> None:
    log = "GMAT: targeter diverged\n  last DV = 1.234 km/s\n"
    exc = GmatRunError("mission sequence failed", log)
    assert exc.log == log
    assert str(exc) == "mission sequence failed"


def test_gmat_run_error_accepts_empty_log() -> None:
    exc = GmatRunError("failed with no captured output", "")
    assert exc.log == ""


# --- GmatOutputParseError -----------------------------------------------------


def test_gmat_output_parse_error_preserves_path_attribute(tmp_path: Path) -> None:
    offending = tmp_path / "ReportFile1.txt"
    exc = GmatOutputParseError("unexpected column count on line 3", offending)
    assert exc.path == offending
    assert str(exc) == "unexpected column count on line 3"


# --- GmatFieldError -----------------------------------------------------------


def test_gmat_field_error_preserves_path_and_value() -> None:
    exc = GmatFieldError("unknown field", "Sat.NoSuchField", 7000.0)
    assert exc.path == "Sat.NoSuchField"
    assert exc.value == 7000.0
    assert str(exc) == "unknown field"


def test_gmat_field_error_defaults_value_to_none_for_reads() -> None:
    exc = GmatFieldError("unknown field on read", "Sat.NoSuchField")
    assert exc.path == "Sat.NoSuchField"
    assert exc.value is None


@pytest.mark.parametrize("value", ["a-string", 42, 3.14, True, None, [1, 2], {"k": "v"}])
def test_gmat_field_error_preserves_arbitrary_value_types(value: object) -> None:
    exc = GmatFieldError("type mismatch", "Sat.SMA", value)
    assert exc.value == value
