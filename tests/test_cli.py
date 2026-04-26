"""Unit tests for :mod:`gmat_run.cli`.

A handwritten fake :class:`Mission` stands in for the real one. The fake
records the arguments it was constructed with, returns a configurable
:class:`_FakeResult`, and is patched into the CLI module via monkeypatch.
This keeps the suite GMAT-free; the end-to-end smoke against a real GMAT
install lives in ``tests/integration/test_cli.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from gmat_run import __version__, cli
from gmat_run.errors import (
    GmatFieldError,
    GmatLoadError,
    GmatNotFoundError,
    GmatOutputParseError,
    GmatRunError,
)

# --- fakes -------------------------------------------------------------------


class _FakeResult:
    def __init__(
        self,
        output_dir: Path,
        *,
        reports: dict[str, pd.DataFrame] | None = None,
        ephemerides: dict[str, pd.DataFrame] | None = None,
        contacts: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.reports = reports or {}
        self.ephemerides = ephemerides or {}
        self.contacts = contacts or {}
        self.persisted_to: Path | None = None

    def persist(self, path: str) -> _FakeResult:
        self.persisted_to = Path(path)
        return self


class _FakeMission:
    """Mission stand-in that records ``load`` arguments and returns a result."""

    last_load: tuple[str, str | None] | None = None
    next_result: _FakeResult | None = None
    load_raises: BaseException | None = None
    run_raises: BaseException | None = None

    def __init__(self, script_path: str) -> None:
        self.script_path = script_path

    @classmethod
    def load(cls, path: str, *, gmat_root: str | None = None) -> _FakeMission:
        cls.last_load = (path, gmat_root)
        load_raises = cls.load_raises
        if load_raises is not None:
            raise load_raises
        return cls(path)

    def run(self) -> _FakeResult:
        run_raises = type(self).run_raises
        if run_raises is not None:
            raise run_raises
        result = type(self).next_result
        assert result is not None, "test must set _FakeMission.next_result"
        return result


@pytest.fixture(autouse=True)
def _reset_fake() -> Iterator[None]:
    _FakeMission.last_load = None
    _FakeMission.next_result = None
    _FakeMission.load_raises = None
    _FakeMission.run_raises = None
    yield
    _FakeMission.last_load = None
    _FakeMission.next_result = None
    _FakeMission.load_raises = None
    _FakeMission.run_raises = None


@pytest.fixture
def patch_mission(monkeypatch: pytest.MonkeyPatch) -> type[_FakeMission]:
    monkeypatch.setattr(cli, "Mission", _FakeMission)
    return _FakeMission


# --- argparse-driven flags ---------------------------------------------------


def test_version_prints_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_top_level_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
    assert "run" in capsys.readouterr().out


def test_run_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["run", "--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--out" in out
    assert "--gmat-root" in out


def test_no_subcommand_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main([])
    # argparse's own error path; we don't pin the code beyond "non-zero".
    assert info.value.code != 0
    assert "required" in capsys.readouterr().err.lower()


def test_run_missing_script_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["run"])
    assert info.value.code != 0
    assert "script" in capsys.readouterr().err.lower()


# --- happy path --------------------------------------------------------------


def test_run_happy_path_summary(
    patch_mission: type[_FakeMission],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_mission.next_result = _FakeResult(
        tmp_path,
        reports={"ReportFile1": pd.DataFrame({"a": range(7)})},
        ephemerides={"EphemerisFile1": pd.DataFrame({"t": range(3)})},
    )

    code = cli.main(["run", "flyby.script"])

    assert code == cli.EXIT_OK
    out = capsys.readouterr().out
    assert f"Output directory: {tmp_path}" in out
    assert "Reports:" in out
    assert "ReportFile1: 7 rows" in out
    assert "Ephemerides:" in out
    assert "EphemerisFile1: 3 rows" in out
    assert "Contacts: (none)" in out
    assert patch_mission.last_load == ("flyby.script", None)


def test_run_with_no_outputs_prints_none_for_each_section(
    patch_mission: type[_FakeMission],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_mission.next_result = _FakeResult(tmp_path)

    code = cli.main(["run", "flyby.script"])

    assert code == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Reports: (none)" in out
    assert "Ephemerides: (none)" in out
    assert "Contacts: (none)" in out


def test_run_out_calls_persist(
    patch_mission: type[_FakeMission],
    tmp_path: Path,
) -> None:
    result = _FakeResult(tmp_path)
    patch_mission.next_result = result
    out_dir = tmp_path / "persisted"

    code = cli.main(["run", "flyby.script", "--out", str(out_dir)])

    assert code == cli.EXIT_OK
    assert result.persisted_to == out_dir


def test_run_without_out_does_not_persist(
    patch_mission: type[_FakeMission],
    tmp_path: Path,
) -> None:
    result = _FakeResult(tmp_path)
    patch_mission.next_result = result

    code = cli.main(["run", "flyby.script"])

    assert code == cli.EXIT_OK
    assert result.persisted_to is None


def test_run_passes_gmat_root_to_load(
    patch_mission: type[_FakeMission],
    tmp_path: Path,
) -> None:
    patch_mission.next_result = _FakeResult(tmp_path)

    code = cli.main(["run", "flyby.script", "--gmat-root", "/opt/gmat-R2026a"])

    assert code == cli.EXIT_OK
    assert patch_mission.last_load == ("flyby.script", "/opt/gmat-R2026a")


# --- exit-code mapping -------------------------------------------------------


def test_gmat_not_found_maps_to_exit_2(
    patch_mission: type[_FakeMission],
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_mission.load_raises = GmatNotFoundError([("env", None, "GMAT_ROOT not set")])

    code = cli.main(["run", "flyby.script"])

    assert code == cli.EXIT_NOT_FOUND == 2
    assert "gmat-run:" in capsys.readouterr().err


def test_gmat_load_error_maps_to_exit_3(
    patch_mission: type[_FakeMission],
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_mission.load_raises = GmatLoadError("could not import gmatpy")

    code = cli.main(["run", "flyby.script"])

    assert code == cli.EXIT_LOAD == 3
    assert "could not import gmatpy" in capsys.readouterr().err


def test_gmat_run_error_maps_to_exit_4(
    patch_mission: type[_FakeMission],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_mission.run_raises = GmatRunError("solver did not converge", log="...")

    code = cli.main(["run", "flyby.script"])

    assert code == cli.EXIT_RUN == 4
    assert "solver did not converge" in capsys.readouterr().err


def test_gmat_output_parse_error_maps_to_exit_5(
    patch_mission: type[_FakeMission],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_path = tmp_path / "bad.txt"

    class _BadFrame:
        def __len__(self) -> int:
            raise GmatOutputParseError("malformed report header", bad_path)

    patch_mission.next_result = _FakeResult(
        tmp_path,
        reports={"ReportFile1": _BadFrame()},  # type: ignore[dict-item]
    )

    code = cli.main(["run", "flyby.script"])

    assert code == cli.EXIT_PARSE == 5
    assert "malformed report header" in capsys.readouterr().err


def test_unexpected_exception_maps_to_exit_1(
    patch_mission: type[_FakeMission],
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_mission.run_raises = RuntimeError("boom")

    code = cli.main(["run", "flyby.script"])

    assert code == cli.EXIT_UNEXPECTED == 1
    assert "unexpected error" in capsys.readouterr().err


def test_field_error_falls_through_to_unexpected(
    patch_mission: type[_FakeMission],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # GmatFieldError is a typed gmat-run error but is not in the issue's
    # exit-code map — it can only happen from explicit get/set, which the
    # CLI never does. Falls through to the generic 1.
    patch_mission.run_raises = GmatFieldError("bogus", "Sat.Nope", value=1)

    code = cli.main(["run", "flyby.script"])

    assert code == cli.EXIT_UNEXPECTED == 1
    err = capsys.readouterr().err
    assert "unexpected error" in err
    assert "bogus" in err


def test_entrypoint_module_target() -> None:
    # The pyproject [project.scripts] entry resolves to gmat_run.cli:main —
    # guard against accidental rename without a corresponding pyproject edit.
    import gmat_run.cli

    assert callable(gmat_run.cli.main)
