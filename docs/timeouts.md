# Timeouts and subprocess execution

Long-running propagations, divergent solvers, and hung subscribers can block
the calling Python process indefinitely. A notebook user has no way out short
of restarting the kernel — Python signal handlers do not fire inside the
SWIG-wrapped `gmat.RunScript()` call (which holds the GIL), and Python threads
cannot be safely killed mid-`gmatpy`.

[`Mission.run`][gmat_run.Mission.run] accepts an opt-in `timeout=` keyword
that solves this by running the mission in a child Python process and killing
it if it exceeds the cap.

## Usage

```python
from gmat_run import Mission, GmatTimeoutError

mission = Mission.load("flyby.script")
mission["Sat.SMA"] = 7100.0  # overrides applied as usual

try:
    result = mission.run(timeout=30.0)
except GmatTimeoutError as exc:
    print(f"killed after {exc.elapsed:.1f} s; partial log:\n{exc.log}")
```

`timeout=None` (the default) runs in-process with no cap — no behavior
change for callers who don't ask for one.

The CLI exposes the same:

```bash
gmat-run run flyby.script --timeout 30
```

[Exit code `6`](cli.md#exit-codes) signals a timeout.

## How it works

The parent forwards three things to the child:

- The `script_path` of the loaded `Mission`.
- Every recorded `__setitem__` write, in dotted-path form, as a JSON object.
- The resolved workspace path (so `ReportFile`, `EphemerisFile`, and
  `ContactLocator` outputs land where the parent expects).

The child re-runs `Mission.load`, replays the override writes, and invokes
the in-process `Mission.run`. The parent waits up to `timeout` seconds via
`Popen.communicate`. On timeout the parent kills the child's process group
(POSIX) or the child itself (Windows) and raises
[`GmatTimeoutError`][gmat_run.GmatTimeoutError]. The error carries the
requested timeout, the wall-clock elapsed, and any partial GMAT log content
salvageable from the workspace at kill time.

## Limitations

### Escape-hatch mutations are not replayed

[`Mission.gmat`][gmat_run.Mission.gmat] is an escape hatch for callers who
need raw `gmatpy` access. Mutations made through it (calling `SetField` on
a resource directly, walking the registry to flip a flag, etc.) are
**not** recorded in the override set the parent ships to the child.

If you have touched `mission.gmat` and then call `mission.run(timeout=...)`,
a `UserWarning` fires:

```text
UserWarning: Mission.gmat was accessed before run(timeout=...). Mutations
made through that escape hatch are not replayed in the child process; only
writes that went through __setitem__ are shipped. The child run may diverge
from what the in-process path would produce.
```

If your mutations matter for the run, either move them into `__setitem__`
calls (the supported path), or run in-process with no `timeout=`.

### Bootstrap cost paid twice

The child re-imports `gmatpy` and re-parses the script. On R2026a this adds
roughly one to two seconds to the run on top of the timeout itself. Acceptable
for the timeout case (the alternative is no timeout at all); something to
keep in mind when picking a `timeout=` value for short-running missions.

### Override values must be JSON-encodable

The parent serialises overrides with `json.dumps(allow_nan=False)`. Every
value [`Mission.__setitem__`][gmat_run.Mission] accepts (real, integer,
boolean, string, string-array, real-vector, real-matrix) is already
JSON-native after the in-process `_coerce` step, so this is transparent
in practice. `NaN` or `Inf` overrides are rejected at the parent encode
call rather than silently round-tripping into the child's `SetField`.

### Windows kill is unbuffered

`TerminateProcess` does not flush GMAT's C++ I/O buffers; orphaned files
in the workspace tempdir are possible after a Windows timeout. The parent
wipes the workspace anyway, so a hung run does not leak a directory.

## When *not* to use `timeout=`

The in-process path is faster, simpler, and easier to debug. Skip
`timeout=` when:

- You're running a known-finite mission whose worst-case wall-clock is
  bounded by something other than a parent-side cap.
- You need raw `gmatpy` access via [`Mission.gmat`][gmat_run.Mission.gmat]
  for graph-level mutations that don't go through `__setitem__`.
- You're driving the engine through your own `gmat.RunScript()` calls
  outside `Mission.run`.
