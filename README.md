# gmat-run

Run GMAT mission scripts from Python and get results as pandas DataFrames.

> **Status:** pre-alpha. Scaffolding only — no public API yet.

## What this is

A thin, Pythonic wrapper around NASA GMAT's own `gmatpy` runtime. You bring a working `.script`;
gmat-run loads it, lets you override fields from Python, runs the mission headlessly, and returns
`ReportFile` / ephemeris / `ContactLocator` output as pandas DataFrames.

```python
from gmat_run import Mission

mission = Mission.load("flyby.script")
mission["Sat.SMA"] = 7000
result = mission.run()
result.reports["ReportFile1"].plot(x="UTCGregorian", y="Sat.Earth.Altitude")
```

## What this is not

- **Not** a way to build GMAT missions from scratch in Python — see
  [`gmatpyplus`](https://github.com/weasdown/gmatpyplus) for that.
- **Not** a `.script` text generator — see [`pygmat`](https://pypi.org/project/pygmat/).
- **Not** a parallel sweep runner — that's a future astro-tools project (`gmat-sweep`) built on top.

## Requirements

- Python 3.10, 3.11, or 3.12.
- A local GMAT install (R2022a or later; R2026a is the primary development target). gmat-run
  does not ship GMAT binaries — install GMAT separately from
  [gmat.gsfc.nasa.gov](https://gmat.gsfc.nasa.gov/).

## Installation

Not yet on PyPI. To work on gmat-run itself:

```bash
git clone https://github.com/astro-tools/gmat-run.git
cd gmat-run
uv sync --all-groups
```

## Licence

MIT. See [LICENSE](LICENSE).
