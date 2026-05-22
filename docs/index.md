# gmat-run

Run GMAT mission scripts from Python and get results as pandas DataFrames.

## What this is

gmat-run drives NASA's General Mission Analysis Tool (GMAT) from Python. You bring a
working `.script`; gmat-run discovers your GMAT install, loads the mission, lets you
override fields from Python with type coercion, runs it headlessly, and parses GMAT's
`ReportFile`, ephemeris, `ContactLocator`, and solver-log output into typed pandas
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
- **Not** a parallel sweep runner — see [gmat-sweep](https://github.com/astro-tools/gmat-sweep),
  an astro-tools project built on top of gmat-run.

## Where to next

- [Getting started](getting-started.md) — install gmat-run and run your first mission.
- [Install GMAT](install-gmat.md) — get the GMAT engine on your machine.
- [Run gmat-run in your CI](ci-with-setup-gmat.md) — wire `astro-tools/setup-gmat`
  into a GitHub Actions workflow.
- [Example notebooks](examples/index.md) — end-to-end Jupyter notebooks that
  exercise the API on stock GMAT missions.
- [API reference](reference/index.md) — the public Python API.
- [Known limitations](known-limitations.md) — gmatpy single-init constraint and other gotchas.
