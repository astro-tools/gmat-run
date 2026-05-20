# Working directories

[`Mission.run`][gmat_run.Mission.run] writes every output file (and GMAT's own
log) into a single directory — the *workspace*. By default that workspace is a
fresh `tempfile.TemporaryDirectory` whose lifetime is tied to the returned
[`Results`][gmat_run.Results]; pass `working_dir=...` to write into a permanent
location instead.

```python
from gmat_run import Mission

# Default: throwaway workspace, cleaned up when `result` is dropped.
result = Mission.load("flyby.script").run()

# Explicit: outputs land in ./out/ and stay there.
result = Mission.load("flyby.script").run(working_dir="out")
```

## Filename rewrite (the rule that makes the workspace work)

GMAT subscribers (`ReportFile`, `EphemerisFile`, `ContactLocator`) cache their
output path internally at parse time, and `FileManager.OUTPUT_PATH` does not
override it after the fact. [`Mission.run`][gmat_run.Mission.run] sidesteps
the cache by rewriting each subscriber's `Filename` field to an absolute path
inside the workspace before invoking `RunScript`.

- A **relative** `Filename` (e.g. `RF.Filename = 'leo_state.txt'`) is rewritten
  to the matching path inside the workspace — `<working_dir>/leo_state.txt`. A
  relative path that carries directory components
  (`RF.Filename = 'reports/leo_state.txt'`) keeps its subdirectory structure:
  it resolves to `<working_dir>/reports/leo_state.txt`, and
  [`Mission.run`][gmat_run.Mission.run] pre-creates the nested parent
  directories that GMAT itself will not. Forward-slash and back-slash relative
  paths are treated identically on every OS. A relative `Filename` containing a
  `..` component is refused with [`GmatRunError`][gmat_run.GmatRunError] before
  the run — it could resolve outside the workspace gmat-run manages.
- An **absolute** `Filename` (e.g. `RF.Filename = '/data/runs/leo.txt'`) is
  honoured as-is. The user picked that destination; gmat-run does not relocate
  it.

If two of a run's outputs resolve to the same path — two subscribers declaring
an identical relative `Filename` — the run is refused with
[`GmatRunError`][gmat_run.GmatRunError], regardless of `overwrite`.

The rewrite does not persist on the `Mission`. A `Mission` stays a view of the
loaded script: reading a subscriber's `Filename` back after the run yields the
value declared in the script, not the resolved workspace path. The resolved
output locations live on the returned [`Results`][gmat_run.Results] —
`report_paths`, `ephemeris_paths`, `contact_paths`, and `solver_paths` each map
a resource name to the absolute path GMAT actually wrote:

```python
mission = Mission.load("flyby.script")
result = mission.run(working_dir="out")

mission["RF.Filename"]       # 'leo_state.txt' — the script's declared value
result.report_paths["RF"]    # Path('/abs/.../out/leo_state.txt')
```

## Out-of-workspace notice

Whenever an absolute `Filename` lands an output outside the workspace, a
single line is prepended to [`Results.log`][gmat_run.Results] so the trail is
visible without walking the path mappings:

```
[gmat-run] note: 1 output(s) landed outside working_dir: /data/runs/leo.txt
<original GMAT log content>
```

The notice is informational. Pinned-absolute paths are a deliberate user
choice and the run is not gated on them.

## Pre-existing artefacts

When `working_dir` already contains a file whose name matches a resolved
output path (the `Filename` rewrite landed it inside the workspace), the
default behaviour is to **refuse the run** rather than silently mix the prior
run's artefacts with the new run's:

```python
Mission.load("flyby.script").run(working_dir="out")
# raises GmatRunError: working_dir 'out' already contains output files: out/leo_state.txt;
# pass overwrite=True to clear them and re-run
```

The gate fires before `RunScript` is invoked, so the existing files are
untouched and the workspace is left in the same state the prior run left it
in. The exception's `.path` attribute carries the offending workspace
directory.

Two ways forward:

- Move the artefacts elsewhere (or run into a different `working_dir`) and
  retry.
- Re-run with `overwrite=True` to unlink the colliding files and proceed:
  ```python
  Mission.load("flyby.script").run(working_dir="out", overwrite=True)
  ```
  `overwrite=True` only removes files that match the run's resolved output
  paths. Stale unrelated files in `working_dir` are left alone, and absolute
  paths the script pinned outside `working_dir` are not touched either —
  the gate is scoped to files inside the workspace.

`overwrite` is ignored for the default-workspace case (`working_dir=None`)
since a fresh temporary directory cannot contain colliding artefacts.

## Permission errors

If `working_dir` cannot be created or the writability probe fails (no write
bit on the directory, full filesystem, ACL refusal on Windows, …)
[`Mission.run`][gmat_run.Mission.run] raises
[`GmatRunError`][gmat_run.GmatRunError] before invoking `RunScript`:

```python
Mission.load("flyby.script").run(working_dir="/proc/cannot-write-here")
# raises GmatRunError: working_dir '/proc/cannot-write-here' is not writable: ...
```

The exception's `.path` attribute points at the offending directory and the
underlying `OSError` is chained via `__cause__`, so callers can present a
diagnostic without re-probing.

## `working_dir` resolves to the script's own directory

Pointing `working_dir` at the directory that contains the loaded `.script`
emits a `UserWarning` — GMAT outputs there can clobber the script itself or
sibling source files when filenames coincide:

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error")
    Mission.load("scripts/flyby.script").run(working_dir="scripts")
    # UserWarning: working_dir 'scripts' is the script's own directory; ...
```

The warning is the only safety rail; the run still proceeds. Filter it
explicitly with `warnings.simplefilter("ignore")` if you have audited the
script's `Filename` declarations and know they cannot collide with sources.

## `Results.persist`

[`Results.persist`][gmat_run.Results.persist] is unaffected by the gates
above. It still copies every artefact under [`Results.output_dir`][gmat_run.Results]
into a destination directory and rewrites the path mappings to point at the
copies. Calling `persist` after an explicit `working_dir` run leaves the
original workspace intact; calling it after the default-workspace run releases
the temporary directory as part of the move.
