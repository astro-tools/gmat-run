# Getting started

## Requirements

- Python 3.10, 3.11, or 3.12.
- A local GMAT install. See [Install GMAT](install-gmat.md). gmat-run does not ship
  GMAT binaries.

### Supported GMAT versions

| GMAT release | Status                     | CI                                                 |
| ------------ | -------------------------- | -------------------------------------------------- |
| R2026a       | Primary development target | Exercised on every PR (Ubuntu + Windows + macOS)   |
| R2025a       | Expected to work           | Not exercised in CI                                |
| R2024a       | Expected to work           | Not exercised in CI                                |
| R2023a       | Expected to work           | Not exercised in CI                                |
| R2022a       | Expected to work           | Not exercised in CI                                |

A wider CI matrix is planned for a follow-up release; report any version-specific
breakage as an issue and we'll add a CI cell for it.

## Install gmat-run

```bash
pip install gmat-run
```

## Quick start

```python
from gmat_run import Mission

mission = Mission.load("flyby.script")
mission["Sat.SMA"] = 7000
result = mission.run()
result.reports["ReportFile1"].plot(x="UTCGregorian", y="Sat.Earth.Altitude")
```

[`Mission.load`][gmat_run.Mission.load] discovers a local GMAT install, bootstraps
`gmatpy`, and parses the `.script` into the live GMAT object graph.

Subscript access reads and writes fields against that graph. Reads return
Python-typed values; writes coerce to the GMAT-expected type:

```python
mission["Sat.SMA"]              # 7000.0  (read)
mission["Sat.SMA"] = 6878.0     # write through SetField
mission["DefaultProp.InitialStepSize"] = 60
```

[`mission.run()`][gmat_run.Mission.run] executes the mission sequence headlessly
and returns a [`Results`][gmat_run.Results] object keyed by the resource names
declared in the script. Parsing is lazy — a DataFrame is materialised on first
access and cached:

```python
df = result.reports["ReportFile1"]
df["UTCGregorian"]              # already parsed as datetime64[ns]
df.plot(x="UTCGregorian", y="Sat.Earth.Altitude")
```

## Pointing at a specific GMAT install

If you have multiple GMAT installs, or your install is in a non-standard location,
override discovery:

```python
mission = Mission.load("flyby.script", gmat_root="/opt/gmat-R2026a")
```

Or set `GMAT_ROOT` in the environment:

```bash
export GMAT_ROOT=/opt/gmat-R2026a
```

## Working directory

By default, `mission.run()` writes GMAT outputs into a fresh temporary directory whose
lifetime is tied to the returned [`Results`][gmat_run.Results] — the directory survives
until you drop the result, so lazy DataFrame access still works after the run returns.
Pass `working_dir=...` to write into a permanent location instead.

## Errors

Every exception gmat-run raises inherits from [`GmatError`][gmat_run.GmatError]. Branch
on the leaf type when you need to react to a specific failure:

| Exception                                                  | When                                                                |
| ---------------------------------------------------------- | ------------------------------------------------------------------- |
| [`GmatNotFoundError`][gmat_run.GmatNotFoundError]          | No usable GMAT install on this machine.                             |
| [`GmatLoadError`][gmat_run.GmatLoadError]                  | gmatpy could not be imported (e.g. wrong Python minor version).     |
| [`GmatRunError`][gmat_run.GmatRunError]                    | The mission sequence itself failed; GMAT's log is on `.log`.        |
| [`GmatFieldError`][gmat_run.GmatFieldError]                | A dotted-path field access failed (unknown path or type mismatch).  |
| [`GmatOutputParseError`][gmat_run.GmatOutputParseError]    | An output file could not be parsed into a DataFrame.                |
