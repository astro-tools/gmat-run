"""Subprocess execution for ``Mission.run(timeout=...)``.

Parent-side driver and child-side handler for the wall-clock-cap path. The
public surface is :meth:`gmat_run.mission.Mission.run` (with ``timeout=``)
and :class:`~gmat_run.errors.GmatTimeoutError`; the JSON wire format
between parent and child is intentionally private and not part of the
package's stable API.

The driver lives here rather than in ``mission.py`` to keep the platform-
specific process management (``start_new_session`` / ``CREATE_NEW_PROCESS_
GROUP``, the SIGTERM/SIGKILL ladder) isolated from the in-process run path.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import IO, Any

from gmat_run.errors import GmatRunError, GmatTimeoutError

__all__ = ["child_main", "run_in_subprocess"]

# Grace period between SIGTERM and SIGKILL on POSIX. The child gets this
# much wall-clock time to clean up after the first signal before the
# parent escalates to an unrecoverable kill.
_KILL_GRACE_SECONDS = 5.0


def run_in_subprocess(
    *,
    script_path: Path,
    overrides: dict[str, Any],
    workspace_path: Path,
    overwrite: bool,
    gmat_root: Path | None,
    timeout: float,
    _popen: Any = None,
) -> dict[str, Any]:
    """Run a mission in a child Python process and return its status payload.

    The parent serialises the inputs as JSON (``allow_nan=False`` so a NaN
    or Inf override surfaces here with a usable stack instead of silently
    landing in the child's ``SetField``), spawns the child via ``-m
    gmat_run.cli _internal-run``, and waits up to ``timeout`` seconds for it
    to return. On timeout, the parent kills the entire child process group
    (POSIX) or the child itself (Windows) and raises
    :class:`~gmat_run.errors.GmatTimeoutError`. On any other failure mode
    (non-zero exit, malformed stdout, ``ok=False`` status) the parent
    raises :class:`~gmat_run.errors.GmatRunError` with the salvaged log.

    ``_popen`` is a test seam — pass a callable matching ``subprocess.Popen``
    to drive the parent without spawning real processes.

    Returns:
        Parsed status dict ``{ok, log, report_paths, ephemeris_paths,
        contact_paths, output_dir}``. The caller (``Mission.run``) builds
        the :class:`~gmat_run.results.Results`.
    """
    payload = {
        "script": str(script_path),
        "overrides": overrides,
        "working_dir": str(workspace_path),
        "overwrite": overwrite,
        "gmat_root": str(gmat_root) if gmat_root is not None else None,
    }
    payload_bytes = json.dumps(payload, allow_nan=False).encode("utf-8")

    cmd = [sys.executable, "-m", "gmat_run.cli", "_internal-run"]
    spawn = _popen if _popen is not None else _spawn
    proc = spawn(cmd)

    started = time.monotonic()
    try:
        stdout, stderr = proc.communicate(input=payload_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        _kill_child(proc)
        log = _salvage_log(workspace_path)
        raise GmatTimeoutError(
            f"mission run exceeded {timeout} s timeout (elapsed {elapsed:.2f} s)",
            log=log,
            requested_timeout=timeout,
            elapsed=elapsed,
        ) from None

    if proc.returncode != 0:
        log = _salvage_log(workspace_path)
        message = stderr.decode("utf-8", errors="replace").strip() or "no stderr"
        raise GmatRunError(
            f"subprocess exited with code {proc.returncode}: {message}",
            log=log,
        )

    try:
        status: dict[str, Any] = json.loads(stdout.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log = _salvage_log(workspace_path)
        raise GmatRunError(
            f"subprocess returned non-JSON output: {exc}",
            log=log,
        ) from exc

    if not status.get("ok", False):
        raise GmatRunError(
            status.get("error", "subprocess returned ok=False without an error message"),
            log=status.get("log", ""),
        )
    return status


def _spawn(cmd: list[str]) -> subprocess.Popen[bytes]:
    """Spawn the child with platform-appropriate process-isolation flags.

    POSIX: ``start_new_session=True`` puts the child in a new session so the
    parent can signal the entire process group together via ``killpg``.
    Windows: ``CREATE_NEW_PROCESS_GROUP`` allows group-level termination
    via ``TerminateProcess``; per the issue we accept that buffer flushing
    may be skipped on the hard-kill path.
    """
    if sys.platform == "win32":
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
        )
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _kill_child(proc: subprocess.Popen[bytes]) -> None:
    """Escalating-signal kill ladder.

    POSIX: ``SIGTERM`` the process group, wait :data:`_KILL_GRACE_SECONDS`,
    then ``SIGKILL`` if anything is still alive. Windows: ``terminate()``
    (= ``TerminateProcess``) — the kernel does not offer a graceful
    escalation that would respect ``CREATE_NEW_PROCESS_GROUP``, so this is
    the unrecoverable kill from the start.
    """
    if sys.platform == "win32":
        with suppress(OSError):
            proc.terminate()
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_KILL_GRACE_SECONDS)
        return

    pgid = _try_get_pgid(proc.pid)
    if pgid is not None:
        with suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=_KILL_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    if pgid is not None:
        with suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_KILL_GRACE_SECONDS)


def _try_get_pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, OSError):
        return None


def _salvage_log(workspace_path: Path) -> str:
    """Best-effort read of the child's GmatLog.txt; empty string on any failure.

    GMAT writes its diagnostic output here as soon as ``UseLogFile`` is
    pointed at the workspace, so even a child killed mid-propagation has
    likely flushed *something* useful. A missing file (child died before
    the log handle was redirected) or any I/O error returns ``""`` so the
    caller can still build a :class:`GmatTimeoutError` without choking.
    """
    log_path = workspace_path / "GmatLog.txt"
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# --- child-side handler -------------------------------------------------------


def child_main(
    stdin: IO[bytes] | None = None,
    stdout: IO[bytes] | None = None,
) -> int:
    """Read JSON payload from stdin, run the mission, write status to stdout.

    Returns an exit code intended for the process entry point. ``stdin`` /
    ``stdout`` parameters exist for testability — the default uses the
    real process streams via ``sys.stdin.buffer`` / ``sys.stdout.buffer``.
    Failures in the handler itself (malformed payload, exception during
    load/run) are reported as ``{"ok": false, "error": ..., "log": ...}``
    on stdout with a non-zero exit code, so the parent can surface them
    in :class:`~gmat_run.errors.GmatRunError`.
    """
    if stdin is None:
        stdin = sys.stdin.buffer
    if stdout is None:
        stdout = sys.stdout.buffer

    payload_bytes = stdin.read()
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _emit_error(stdout, f"invalid stdin payload: {exc}", "")
        return 1

    try:
        status = _run_child(payload)
    except Exception as exc:  # noqa: BLE001 — top-level handler, surface anything
        log = ""
        # GmatRunError carries a log; surface it so the parent can show it.
        if hasattr(exc, "log"):
            log_attr = getattr(exc, "log", "")
            if isinstance(log_attr, str):
                log = log_attr
        _emit_error(stdout, f"{type(exc).__name__}: {exc}", log)
        return 1

    stdout.write(json.dumps(status, allow_nan=False).encode("utf-8"))
    stdout.flush()
    return 0


def _run_child(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply ``payload`` to a fresh :class:`Mission` and return a status dict.

    The handler walks the in-process ``Mission.load`` → ``__setitem__`` →
    ``run`` pipeline exactly as a notebook user would; the wall-clock cap
    is enforced by the parent, not here.
    """
    # Late import — keeps the parent process from paying the gmatpy
    # bootstrap cost just because it imported the driver.
    from gmat_run.mission import Mission

    script = payload["script"]
    overrides: dict[str, Any] = payload.get("overrides", {})
    working_dir = payload.get("working_dir")
    overwrite: bool = bool(payload.get("overwrite", False))
    gmat_root = payload.get("gmat_root")

    mission = Mission.load(script, gmat_root=gmat_root)
    for dotted, value in overrides.items():
        mission[dotted] = value
    result = mission.run(working_dir=working_dir, overwrite=overwrite)

    # `Results` exposes ephemeris_paths / contact_paths publicly; the
    # report-path mapping lives on the lazy `reports` view as `_paths`.
    # Asymmetric, but matches the existing public surface.
    report_paths: dict[str, Path] = result.reports._paths  # type: ignore[attr-defined]
    return {
        "ok": True,
        "log": result.log,
        "report_paths": {k: str(v) for k, v in report_paths.items()},
        "ephemeris_paths": {k: str(v) for k, v in result.ephemeris_paths.items()},
        "contact_paths": {k: str(v) for k, v in result.contact_paths.items()},
        "output_dir": str(result.output_dir),
    }


def _emit_error(stdout: IO[bytes], error: str, log: str) -> None:
    payload = {"ok": False, "error": error, "log": log}
    stdout.write(json.dumps(payload, allow_nan=False).encode("utf-8"))
    stdout.flush()
