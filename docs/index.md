# gmat-run

Run GMAT mission scripts from Python and get results as pandas DataFrames.

## What this is

A thin, Pythonic wrapper around NASA GMAT's own `gmatpy` runtime. You bring a working
`.script`; gmat-run loads it, lets you override fields from Python, runs the mission
headlessly, and returns `ReportFile` / ephemeris / `ContactLocator` output as pandas
DataFrames.

```python
from gmat_run import Mission

mission = Mission.load("flyby.script")
mission["Sat.SMA"] = 7000
result = mission.run()
result.reports["ReportFile1"].plot(x="UTCGregorian", y="Sat.Earth.Altitude")
```

## What this is not

- **Not** a way to build GMAT missions from scratch in Python — see
  [gmatpyplus](https://github.com/weasdown/gmatpyplus) for that.
- **Not** a `.script` text generator — see [pygmat](https://pypi.org/project/pygmat/).
- **Not** a parallel sweep runner — that's a future astro-tools project (`gmat-sweep`)
  built on top of gmat-run.

## Where to next

- [Getting started](getting-started.md) — install gmat-run and run your first mission.
- [Install GMAT](install-gmat.md) — get the GMAT engine on your machine.
- [API reference](reference/mission.md) — the public Python API.
- [Known limitations](known-limitations.md) — gmatpy single-init constraint and other gotchas.
