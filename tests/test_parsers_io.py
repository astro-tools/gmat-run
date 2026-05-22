"""The typed-error contract every output-file parser shares.

A missing or non-UTF-8 input must surface as ``GmatOutputParseError`` — never a
raw ``FileNotFoundError`` / ``UnicodeDecodeError`` — from both the shared reader
:func:`gmat_run.parsers._io.read_text_lines` and every parser's ``parse`` entry
point.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from gmat_run.errors import GmatOutputParseError
from gmat_run.parsers._io import read_text_lines
from gmat_run.parsers.aem_ephemeris import parse as parse_aem
from gmat_run.parsers.contact import parse as parse_contact
from gmat_run.parsers.ephemeris import parse as parse_oem
from gmat_run.parsers.reportfile import parse as parse_reportfile
from gmat_run.parsers.solver_log import parse as parse_solver_log
from gmat_run.parsers.stk_ephemeris import parse as parse_stk

# Every byte value, so a strict UTF-8 decode fails. Stands in for a binary
# ephemeris (e.g. GMAT's Code-500 format) reaching a text parser.
_BINARY = bytes(range(256)) * 8


# --- read_text_lines ---------------------------------------------------------


def test_read_text_lines_universal_newline_split(tmp_path: Path) -> None:
    path = tmp_path / "ok.txt"
    path.write_bytes(b"alpha\r\nbeta\n")
    assert read_text_lines(path) == ["alpha", "beta"]


def test_read_text_lines_strips_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfalpha\n")
    assert read_text_lines(path) == ["alpha"]


def test_read_text_lines_missing_file_raises_typed(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.txt"
    with pytest.raises(GmatOutputParseError) as excinfo:
        read_text_lines(missing)
    assert excinfo.value.path == missing
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_read_text_lines_binary_file_raises_typed(tmp_path: Path) -> None:
    path = tmp_path / "binary.bin"
    path.write_bytes(_BINARY)
    with pytest.raises(GmatOutputParseError) as excinfo:
        read_text_lines(path)
    assert excinfo.value.path == path
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


# --- every parser surfaces the typed error -----------------------------------

# One entry per text parser that reads via read_text_lines. spk.parse is
# excluded — it is a binary format with its own typed missing-file handling.
_PARSERS = [
    pytest.param(parse_reportfile, id="reportfile"),
    pytest.param(parse_oem, id="ephemeris"),
    pytest.param(parse_stk, id="stk_ephemeris"),
    pytest.param(parse_aem, id="aem_ephemeris"),
    pytest.param(parse_contact, id="contact"),
    pytest.param(parse_solver_log, id="solver_log"),
]


@pytest.mark.parametrize("parse_fn", _PARSERS)
def test_parser_missing_file_raises_typed_error(
    parse_fn: Callable[..., pd.DataFrame], tmp_path: Path
) -> None:
    """A missing path surfaces GmatOutputParseError, not raw FileNotFoundError."""
    with pytest.raises(GmatOutputParseError):
        parse_fn(tmp_path / "missing-output.txt")


@pytest.mark.parametrize("parse_fn", _PARSERS)
def test_parser_binary_file_raises_typed_error(
    parse_fn: Callable[..., pd.DataFrame], tmp_path: Path
) -> None:
    """A binary file surfaces GmatOutputParseError, not raw UnicodeDecodeError."""
    path = tmp_path / "binary-output.bin"
    path.write_bytes(_BINARY)
    with pytest.raises(GmatOutputParseError):
        parse_fn(path)
