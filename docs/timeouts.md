# Wall-clock timeouts

[`Mission.run`][gmat_run.Mission.run] does not take a `timeout=` keyword. A
divergent solver or a hung subscriber will block the calling Python process
until the interpreter is restarted.

The library doesn't ship a built-in cap because the use case is narrow — real
GMAT runs are slow, not hung — and a short subprocess recipe covers it
cleanly without dragging cross-platform process management into the library.
If you need a wall-clock cap on a mission run, lift the snippet below into
your own code.

## Recipe

```python
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

DRIVER = """
import json
import sys

from gmat_run import Mission

payload = json.loads(sys.stdin.read())
mission = Mission.load(payload["script"])
for k, v in payload["overrides"].items():
    mission[k] = v
mission.run(working_dir=payload["workspace"])
"""


@contextmanager
def run_with_timeout(script, overrides, timeout):
    """Run a GMAT mission in a child process under a wall-clock cap.

    Yields the workspace path. The temp dir is cleaned up on context exit,
    so read every output you need before leaving the ``with`` block.
    """
    with tempfile.TemporaryDirectory() as workspace:
        payload = json.dumps(
            {
                "script": str(script),
                "overrides": overrides,
                "workspace": workspace,
            }
        )
        try:
            subprocess.run(
                [sys.executable, "-c", DRIVER],
                input=payload,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"mission exceeded {timeout} s") from exc
        yield Path(workspace)
```

## Using it

```python
import pandas as pd

with run_with_timeout("flyby.script", {"Sat.SMA": 7000}, timeout=60) as ws:
    df = pd.read_csv(ws / "ReportFile1.txt", sep=r"\s+")
    # ...do whatever you need with df before leaving the block
```

## How it works

- `subprocess.run(timeout=...)` does the cross-platform kill: `SIGTERM` on
  POSIX, `TerminateProcess` on Windows. There is nothing for the library to
  add on top.
- `tempfile.TemporaryDirectory` owns the workspace lifetime; the `with`
  block on the caller side keeps it alive until you've finished reading.
- The driver uses `json` rather than `pickle` for the override payload — only
  JSON-encodable values cross the process boundary. If you need to pass numpy
  arrays, convert with `.tolist()` first.
- The child process pays the full `gmat_run.bootstrap()` cost on every call.
  That's the right price for the use case (wall-clock-bounded, occasionally
  killed); for tight inner loops, run in-process without the timeout.

## Limitations

- **No partial results on timeout.** When the cap fires, the child process is
  killed mid-run — output files in the workspace may be empty, truncated, or
  missing. The recipe surfaces only the timeout itself; recovering whatever
  GMAT had flushed to disk is on you.
- **One timeout per call.** Wrapping `Mission.run` from inside an in-process
  thread does not work — `gmat.RunScript` holds the GIL through a blocking
  C++ call, so Python signals never fire. The subprocess split is what makes
  the cap real.
- **JSON-only overrides.** `Mission["Sat.OrbitState"] = np.array([...])`
  works in-process, but the array has to be pre-converted to a list before
  crossing the subprocess boundary here.
