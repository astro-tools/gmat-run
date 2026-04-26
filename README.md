# gmat-run

[![CI](https://github.com/astro-tools/gmat-run/actions/workflows/ci.yml/badge.svg)](https://github.com/astro-tools/gmat-run/actions/workflows/ci.yml)
[![Docs](https://github.com/astro-tools/gmat-run/actions/workflows/docs.yml/badge.svg)](https://astro-tools.github.io/gmat-run/)
[![PyPI](https://img.shields.io/pypi/v/gmat-run.svg)](https://pypi.org/project/gmat-run/)
[![Python versions](https://img.shields.io/pypi/pyversions/gmat-run.svg)](https://pypi.org/project/gmat-run/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Run GMAT mission scripts from Python and get results as pandas DataFrames.

## What this is

A thin, Pythonic wrapper around NASA GMAT's own `gmatpy` runtime. You bring a working `.script`;
gmat-run loads it, lets you override fields from Python, runs the mission headlessly, and returns
`ReportFile` / ephemeris / `ContactLocator` output as pandas DataFrames.

## What this is not

- **Not** a way to build GMAT missions from scratch in Python — see
  [`gmatpyplus`](https://github.com/weasdown/gmatpyplus) for that.
- **Not** a `.script` text generator — see [`pygmat`](https://pypi.org/project/pygmat/).
- **Not** a parallel sweep runner — that's a future astro-tools project (`gmat-sweep`) built on top.

## Requirements

- Python 3.10, 3.11, or 3.12.
- A local GMAT install. gmat-run does not ship GMAT binaries — install GMAT separately from
  [gmat.gsfc.nasa.gov](https://gmat.gsfc.nasa.gov/).

### Supported GMAT versions

| GMAT release | Status | CI |
|---|---|---|
| R2026a | Primary development target | Exercised on every PR (Ubuntu + Windows + macOS) |
| R2025a | Expected to work | Not exercised in CI |
| R2024a | Expected to work | Not exercised in CI |
| R2023a | Expected to work | Not exercised in CI |
| R2022a | Expected to work | Not exercised in CI |

A wider CI matrix is planned for a follow-up release; report any version-specific breakage as
an issue and we'll add a CI cell for it.

## Installation

```bash
pip install gmat-run
```

For SPK ephemerides, install the `spiceypy` extra:

```bash
pip install gmat-run[spiceypy]
```

## Quick start

Load a script, override a field, run the mission, and read each output GMAT wrote as a
pandas DataFrame:

```python
from gmat_run import Mission

mission = Mission.load("flyby.script")
mission["Sat.SMA"] = 7000
result = mission.run()

# ReportFile → DataFrame, with UTCGregorian / *ModJulian epoch columns
# promoted to datetime64[ns].
result.reports["ReportFile1"].plot(x="UTCGregorian", y="Sat.Earth.Altitude")

# EphemerisFile → DataFrame, dispatching on file format.
ephem = result.ephemerides["EphemerisFile1"]

# ContactLocator → DataFrame; df.attrs["report_format"] carries the variant.
contacts = result.contacts["ContactLocator1"]
```

`Mission.load` discovers a local GMAT install (honouring the `GMAT_ROOT` environment variable
or a `gmat_root=` argument), bootstraps `gmatpy`, and parses the script into the live GMAT
object graph. Subscript access reads and writes fields against that graph with type coercion.
`mission.run()` executes the mission sequence headlessly, captures GMAT's log, and returns a
`Results` exposing three lazy mappings — `reports`, `ephemerides`, and `contacts` — each
keyed by the GMAT resource name and parsing to a DataFrame on first access. See
[Outputs](#outputs) below for the formats covered.

A `gmat-run` console script is also installed for shell-script and smoke-test use:

```bash
gmat-run run flyby.script --out results/
```

See the [CLI reference](https://astro-tools.github.io/gmat-run/cli/) for flags, exit codes,
and sample output.

## Outputs

`Results` exposes three mappings, each keyed by the GMAT resource name as declared in the
`.script`:

- **`ReportFile`** → DataFrame, with `UTCGregorian` and `*ModJulian` epoch columns promoted
  to `datetime64[ns]`.
- **`EphemerisFile`** → DataFrame, dispatching on file format: **CCSDS-OEM** and
  **STK-TimePosVel** are read out of the box; **SPK** (NASA SPICE binary) is read with the
  `[spiceypy]` extra installed. **Code-500** (GSFC binary) is tracked for v0.3 (#50).
- **`ContactLocator`** → DataFrame, supporting Legacy and the five tabular `ReportFormat`
  variants. `df.attrs["report_format"]` carries the variant name so downstream code can
  branch on it without inspecting the column set.

## Documentation

Full docs at **<https://astro-tools.github.io/gmat-run/>**, including a
[getting-started guide](https://astro-tools.github.io/gmat-run/getting-started/),
[GMAT install instructions](https://astro-tools.github.io/gmat-run/install-gmat/),
the [CLI reference](https://astro-tools.github.io/gmat-run/cli/),
and the [API reference](https://astro-tools.github.io/gmat-run/reference/).

Runnable example notebooks:

- [Load / run / plot](https://astro-tools.github.io/gmat-run/examples/01_load_run_plot/) —
  load a stock GMAT sample, run it, and plot altitude over time end-to-end.
- [Parameter sweep](https://astro-tools.github.io/gmat-run/examples/02_parameter_sweep/) —
  vary `Sat.SMA` across a range, run the same script for each, and overlay the resulting
  orbits.
- [Ground track](https://astro-tools.github.io/gmat-run/examples/03_ground_track/) — read an
  `EphemerisFile` from `Results.ephemerides` and plot the spacecraft's ground track on a
  Cartopy world map.

## Development

To work on gmat-run itself:

```bash
git clone https://github.com/astro-tools/gmat-run.git
cd gmat-run
uv sync --all-groups
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full branch / PR / test workflow.

## Licence

MIT. See [LICENSE](LICENSE).
