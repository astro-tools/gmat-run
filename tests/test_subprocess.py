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
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, ClassVar

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
        input: bytes | None = None,
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
    proc = _FakePopen(
        stdout=b'{"ok": true, "log": "", "report_paths": {}, '
        b'"ephemeris_paths": {}, "contact_paths": {}, '
        b'"output_dir": ""}'
    )
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
    proc = _FakePopen(
        stdout=b'{"ok": true, "log": "", "report_paths": {}, '
        b'"ephemeris_paths": {}, "contact_paths": {}, '
        b'"output_dir": ""}'
    )
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


# --- _kill_child --------------------------------------------------------------


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX-only ladder")
def test_kill_child_posix_sigterm_then_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fast path: SIGTERM, child exits within the grace window, no SIGKILL."""
    from gmat_run import _subprocess as sub

    proc = _FakePopen()
    proc.returncode = -15  # already exited from the SIGTERM caller's perspective

    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(sub, "_try_get_pgid", lambda _pid: 99)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))

    sub._kill_child(proc)  # type: ignore[arg-type]

    # SIGTERM was sent; SIGKILL was not (child exited inside the grace window).
    assert (99, signal.SIGTERM) in sent
    assert (99, signal.SIGKILL) not in sent


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX-only ladder")
def test_kill_child_posix_escalates_to_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    """When SIGTERM is ignored past the grace window, the ladder escalates."""
    from gmat_run import _subprocess as sub

    proc = _FakePopen()  # returncode stays None — proc.wait raises TimeoutExpired

    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(sub, "_try_get_pgid", lambda _pid: 99)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))

    sub._kill_child(proc)  # type: ignore[arg-type]

    assert (99, signal.SIGTERM) in sent
    assert (99, signal.SIGKILL) in sent


# --- _run_child ---------------------------------------------------------------


def test_run_child_loads_applies_overrides_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child handler walks load → __setitem__ → run with the parent's payload."""
    from gmat_run import _subprocess as sub

    set_calls: list[tuple[str, Any]] = []

    # Use a tmp_path-style location so `str(Path(...))` round-trips
    # natively on whatever platform the test runs (Windows turns "/tmp/x"
    # into "\\tmp\\x" and the assertion would diverge).
    work = Path("work")
    rf_path = work / "rf.txt"
    eph_path = work / "e1.eph"

    class _FakeReports:
        _paths: ClassVar[dict[str, Path]] = {"RF": rf_path}

    class _FakeResult:
        log = "child ran fine\n"
        reports = _FakeReports()
        ephemeris_paths: ClassVar[dict[str, Path]] = {"E1": eph_path}
        contact_paths: ClassVar[dict[str, Path]] = {}
        output_dir = work

    class _FakeMission:
        @classmethod
        def load(cls, path: str, *, gmat_root: str | None = None) -> _FakeMission:
            cls.last_load = (path, gmat_root)  # type: ignore[attr-defined]
            return cls()

        def __setitem__(self, key: str, value: Any) -> None:
            set_calls.append((key, value))

        def run(self, *, working_dir: Any = None, overwrite: bool = False) -> _FakeResult:
            type(self).last_run = (working_dir, overwrite)  # type: ignore[attr-defined]
            return _FakeResult()

    # Patch the late-imported Mission inside _run_child by stubbing the
    # module attribute on `gmat_run.mission` before the function imports it.
    import gmat_run.mission as mission_module

    monkeypatch.setattr(mission_module, "Mission", _FakeMission)

    payload = {
        "script": "/path/to/x.script",
        "overrides": {"Sat.SMA": 7100.0, "Sat.ECC": 0.01},
        "working_dir": str(work),
        "overwrite": True,
        "gmat_root": "/opt/gmat",
    }
    status = sub._run_child(payload)

    assert _FakeMission.last_load == ("/path/to/x.script", "/opt/gmat")  # type: ignore[attr-defined]
    assert ("Sat.SMA", 7100.0) in set_calls
    assert ("Sat.ECC", 0.01) in set_calls
    assert _FakeMission.last_run == (str(work), True)  # type: ignore[attr-defined]
    assert status == {
        "ok": True,
        "log": "child ran fine\n",
        "report_paths": {"RF": str(rf_path)},
        "ephemeris_paths": {"E1": str(eph_path)},
        "contact_paths": {},
        "output_dir": str(work),
    }


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


def test_child_main_propagates_gmat_run_error_log(monkeypatch: pytest.MonkeyPatch) -> None:
    # When _run_child raises a GmatRunError-shaped exception, its `log`
    # attribute must surface on the wire so the parent can propagate it
    # through to the caller's GmatRunError.log.
    from gmat_run import _subprocess as sub

    def _raise_with_log(_payload: dict[str, Any]) -> dict[str, Any]:
        raise GmatRunError("engine failed", log="GMAT: integrator diverged\n")

    monkeypatch.setattr(sub, "_run_child", _raise_with_log)

    stdin = io.BytesIO(json.dumps({"script": "/x.script"}).encode("utf-8"))
    stdout = io.BytesIO()
    rc = child_main(stdin=stdin, stdout=stdout)
    assert rc == 1
    parsed = json.loads(stdout.getvalue().decode("utf-8"))
    assert parsed["ok"] is False
    assert "GmatRunError" in parsed["error"]
    assert parsed["log"] == "GMAT: integrator diverged\n"


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
