# Wall-clock timeouts

[`Mission.run`][gmat_run.Mission.run] does not take a `timeout=` keyword. If
you need a wall-clock cap on a mission run — for example to bound a divergent
solver or a hung subscriber — run the mission in a child process and let
`subprocess.run(timeout=...)` enforce the cap.

## Worked example

Put the mission in its own file:

```python
# run_flyby.py
from gmat_run import Mission

mission = Mission.load("flyby.script")
mission["Sat.SMA"] = 7000
mission.run(working_dir="out/")
```

Run it from the parent process under a timeout:

```python
import subprocess

try:
    subprocess.run(
        ["python", "run_flyby.py"],
        timeout=60,
        check=True,
    )
except subprocess.TimeoutExpired:
    raise RuntimeError("mission exceeded 60 s") from None

# outputs are now in out/ — read them however you like
```

`subprocess.run(timeout=...)` does the kill cross-platform: `SIGTERM` on
POSIX, `TerminateProcess` on Windows.

To parameterise the child — different scripts, different overrides — pass
arguments via `argv` or read a small config file from inside `run_flyby.py`.

## Caveats

- **No partial results on timeout.** When the cap fires, the child is killed
  mid-run; output files in the workspace may be empty, truncated, or missing.
- **Bootstrap cost on every call.** The child pays the full
  `gmat_run.bootstrap()` import on every run, which adds noticeable startup
  time. For tight inner loops, run in-process without a timeout.
- **Threads can't substitute for a child process.** Wrapping `Mission.run` in
  an in-process thread does not give you a real cap — `gmat.RunScript` holds
  the GIL through a blocking C++ call, so Python signals never fire.
