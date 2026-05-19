"""Unit tests for :mod:`gmat_run._path_utils`.

The helper centralises the path-normalisation rule applied at every
public path-shaped boundary (``Mission.load``, ``Mission.run``,
``Results.persist``, ``write_oem``). The tests pin its contract directly
so any future tightening or loosening has a single place to assert.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gmat_run._path_utils import resolve_user_path


def test_resolves_relative_against_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = resolve_user_path("outputs/run_0")
    assert result == (tmp_path / "outputs/run_0").resolve()
    assert result.is_absolute()


def test_absolute_path_is_noop(tmp_path: Path) -> None:
    abs_input = tmp_path / "abs"
    result = resolve_user_path(abs_input)
    assert result == abs_input.resolve()
    assert result.is_absolute()


def test_tilde_is_expanded() -> None:
    result = resolve_user_path("~/some-target")
    assert result == (Path.home() / "some-target").resolve()
    assert "~" not in str(result)


def test_pathlike_input_accepted(tmp_path: Path) -> None:
    result = resolve_user_path(tmp_path / "sub")
    assert result == (tmp_path / "sub").resolve()


def test_empty_string_resolves_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = resolve_user_path("")
    assert result == tmp_path.resolve()


def test_dot_resolves_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = resolve_user_path(".")
    assert result == tmp_path.resolve()
