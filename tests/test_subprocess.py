"""Unit tests for :mod:`gmat_run._subprocess`.

The driver is exercised with a fake ``Popen`` so tests can drive the
parent through the timeout, non-zero-exit, malformed-stdout, and
ok-False branches deterministically. ``child_main`` is exercised by
calling it directly with ``BytesIO`` stdin/stdout — the real subprocess
end-to-end path is covered in ``tests/integration/test_timeout.py``.
"""

from __future__ import annotations

import io
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gmat_run._subprocess import child_main, run_in_subprocess
from gmat_run.errors import GmatRunError, GmatTimeoutError

# --- fake Popen ---------------------------------------------------------------


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` controllable by the test.

    ``communicate`` returns ``(stdout, stderr)`` after setting
    ``returncode`` if ``timeout_after`` is ``None``; otherwise it raises
    :class:`subprocess.TimeoutExpired`. ``terminate`` / ``wait`` /
    ``kill`` track call counts so the kill-ladder branches are
    observable.
    """

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        raise_timeout: bool = False,
        terminate_actually_kills: bool = True,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._raise_timeout = raise_timeout
        self._terminate_kills = terminate_actually_kills
        self.returncode: int | None = None
        self.pid = 12345
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self.received_stdin: bytes | None = None

    def communicate(
        self,
        *,
        input: bytes | None = None,  # noqa: A002 — match Popen signature
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        self.received_stdin = input
        if self._raise_timeout:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0.0)
        self.returncode = self._returncode
        return self._stdout, self._stderr

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._terminate_kills:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0.0)
        return self.returncode


def _make_spawn(proc: _FakePopen) -> Any:
    """Wrap a single fake Popen as a ``spawn(cmd) -> proc`` callable."""

    def _spawn(_cmd: list[str]) -> _FakePopen:
        return proc

    return _spawn


# --- run_in_subprocess: happy path --------------------------------------------


def test_run_in_subprocess_decodes_status_dict(tmp_path: Path) -> None:
    payload = {
        "ok": True,
        "log": "GMAT execution complete\n",
        "report_paths": {"RF": str(tmp_path / "rf.txt")},
        "ephemeris_paths": {},
        "contact_paths": {},
        "output_dir": str(tmp_path),
    }
    proc = _FakePopen(stdout=json.dumps(payload).encode("utf-8"))
    status = run_in_subprocess(
        script_path=tmp_path / "x.script",
        overrides={},
        workspace_path=tmp_path,
        overwrite=False,
        gmat_root=None,
        timeout=10.0,
        _popen=_make_spawn(proc),
    )
    assert status == payload


def test_run_in_subprocess_serialises_overrides_with_allow_nan_false(
    tmp_path: Path,
) -> None:
    proc = _FakePopen(stdout=b'{"ok": true, "log": "", "report_paths": {}, '
                             b'"ephemeris_paths": {}, "contact_paths": {}, '
                             b'"output_dir": ""}')
    run_in_subprocess(
        script_path=tmp_path / "x.script",
        overrides={"Sat.SMA": 7000.0, "Sat.ECC": 0.0},
        workspace_path=tmp_path,
        overwrite=True,
        gmat_root=None,
        timeout=5.0,
        _popen=_make_spawn(proc),
    )
    assert proc.received_stdin is not None
    sent = json.loads(proc.received_stdin.decode("utf-8"))
    assert sent["overrides"] == {"Sat.SMA": 7000.0, "Sat.ECC": 0.0}
    assert sent["working_dir"] == str(tmp_path)
    assert sent["overwrite"] is True


def test_run_in_subprocess_rejects_nan_override(tmp_path: Path) -> None:
    # allow_nan=False on the encode side is the parent-stack-trace guarantee
    # — a NaN/Inf must not silently round-trip into the child's SetField.
    with pytest.raises(ValueError):  # json raises ValueError on NaN with allow_nan=False
        run_in_subprocess(
            script_path=tmp_path / "x.script",
            overrides={"Sat.SMA": math.nan},
            workspace_path=tmp_path,
            overwrite=False,
            gmat_root=None,
            timeout=5.0,
            _popen=_make_spawn(_FakePopen()),
        )


def test_run_in_subprocess_passes_gmat_root(tmp_path: Path) -> None:
    proc = _FakePopen(stdout=b'{"ok": true, "log": "", "report_paths": {}, '
                             b'"ephemeris_paths": {}, "contact_paths": {}, '
                             b'"output_dir": ""}')
    root = tmp_path / "gmat-install"
    run_in_subprocess(
        script_path=tmp_path / "x.script",
        overrides={},
        workspace_path=tmp_path,
        overwrite=False,
        gmat_root=root,
        timeout=5.0,
        _popen=_make_spawn(proc),
    )
    sent = json.loads((proc.received_stdin or b"").decode("utf-8"))
    assert sent["gmat_root"] == str(root)


# --- run_in_subprocess: failure paths -----------------------------------------


def test_run_in_subprocess_raises_gmat_timeout_error_on_timeout(tmp_path: Path) -> None:
    proc = _FakePopen(raise_timeout=True)
    log_path = tmp_path / "GmatLog.txt"
    log_path.write_text("partial GMAT log content\n", encoding="utf-8")

    with pytest.raises(GmatTimeoutError) as excinfo:
        run_in_subprocess(
            script_path=tmp_path / "x.script",
            overrides={},
            workspace_path=tmp_path,
            overwrite=False,
            gmat_root=None,
            timeout=2.0,
            _popen=_make_spawn(proc),
        )
    err = excinfo.value
    assert err.requested_timeout == 2.0
    assert err.elapsed >= 0.0
    assert "partial GMAT log" in err.log


def test_run_in_subprocess_kills_child_on_timeout(tmp_path: Path) -> None:
    proc = _FakePopen(raise_timeout=True)
    with pytest.raises(GmatTimeoutError):
        run_in_subprocess(
            script_path=tmp_path / "x.script",
            overrides={},
            workspace_path=tmp_path,
            overwrite=False,
            gmat_root=None,
            timeout=1.0,
            _popen=_make_spawn(proc),
        )
    # On POSIX the kill ladder uses os.killpg, not proc.terminate; on Windows
    # it would call proc.terminate. Either way wait() is called at least once
    # to give the child a chance to exit.
    assert proc.wait_calls >= 1


def test_run_in_subprocess_raises_gmat_run_error_on_nonzero_exit(tmp_path: Path) -> None:
    proc = _FakePopen(stdout=b"", stderr=b"child crashed\n", returncode=1)
    with pytest.raises(GmatRunError) as excinfo:
        run_in_subprocess(
            script_path=tmp_path / "x.script",
            overrides={},
            workspace_path=tmp_path,
            overwrite=False,
            gmat_root=None,
            timeout=5.0,
            _popen=_make_spawn(proc),
        )
    assert "exited with code 1" in str(excinfo.value)
    assert "child crashed" in str(excinfo.value)


def test_run_in_subprocess_raises_gmat_run_error_on_malformed_stdout(tmp_path: Path) -> None:
    proc = _FakePopen(stdout=b"not json at all", returncode=0)
    with pytest.raises(GmatRunError) as excinfo:
        run_in_subprocess(
            script_path=tmp_path / "x.script",
            overrides={},
            workspace_path=tmp_path,
            overwrite=False,
            gmat_root=None,
            timeout=5.0,
            _popen=_make_spawn(proc),
        )
    assert "non-JSON" in str(excinfo.value)


def test_run_in_subprocess_raises_gmat_run_error_on_ok_false(tmp_path: Path) -> None:
    payload = {"ok": False, "error": "GMAT rejected the script", "log": "engine error\n"}
    proc = _FakePopen(stdout=json.dumps(payload).encode("utf-8"), returncode=0)
    with pytest.raises(GmatRunError) as excinfo:
        run_in_subprocess(
            script_path=tmp_path / "x.script",
            overrides={},
            workspace_path=tmp_path,
            overwrite=False,
            gmat_root=None,
            timeout=5.0,
            _popen=_make_spawn(proc),
        )
    assert "GMAT rejected" in str(excinfo.value)
    assert excinfo.value.log == "engine error\n"


# --- child_main ---------------------------------------------------------------


def test_child_main_emits_error_on_invalid_stdin() -> None:
    stdin = io.BytesIO(b"this is not json")
    stdout = io.BytesIO()
    rc = child_main(stdin=stdin, stdout=stdout)
    assert rc == 1
    parsed = json.loads(stdout.getvalue().decode("utf-8"))
    assert parsed["ok"] is False
    assert "invalid stdin payload" in parsed["error"]


def test_child_main_surfaces_handler_exception_on_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force _run_child to raise; the handler must catch it and emit a
    # structured error on stdout rather than a Python traceback on stderr.
    from gmat_run import _subprocess as sub

    def _boom(_payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(sub, "_run_child", _boom)

    stdin = io.BytesIO(json.dumps({"script": "/x.script"}).encode("utf-8"))
    stdout = io.BytesIO()
    rc = child_main(stdin=stdin, stdout=stdout)
    assert rc == 1
    parsed = json.loads(stdout.getvalue().decode("utf-8"))
    assert parsed["ok"] is False
    assert "RuntimeError" in parsed["error"]
    assert "kaboom" in parsed["error"]


def test_child_main_passes_through_run_child_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from gmat_run import _subprocess as sub

    def _ok(payload: dict[str, Any]) -> dict[str, Any]:
        # Echo a deterministic status so the test asserts the wire format.
        return {
            "ok": True,
            "log": "ran fine\n",
            "report_paths": {"RF": "/tmp/rf.txt"},
            "ephemeris_paths": {},
            "contact_paths": {},
            "output_dir": payload.get("working_dir", ""),
        }

    monkeypatch.setattr(sub, "_run_child", _ok)

    stdin = io.BytesIO(
        json.dumps(
            {
                "script": "/x.script",
                "overrides": {"Sat.SMA": 7000.0},
                "working_dir": "/tmp/work",
                "overwrite": False,
                "gmat_root": None,
            }
        ).encode("utf-8")
    )
    stdout = io.BytesIO()
    rc = child_main(stdin=stdin, stdout=stdout)
    assert rc == 0
    parsed = json.loads(stdout.getvalue().decode("utf-8"))
    assert parsed["ok"] is True
    assert parsed["log"] == "ran fine\n"
    assert parsed["report_paths"] == {"RF": "/tmp/rf.txt"}
    assert parsed["output_dir"] == "/tmp/work"
