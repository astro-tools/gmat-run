# Examples

End-to-end Jupyter notebooks that exercise the `gmat-run` API on stock GMAT
sample missions. Each notebook is committed with cell outputs so you can read
through it on the docs site without running anything; you can also run them
locally after `pip install gmat-run[examples]` and the matplotlib dependency.

- [Load, run, and plot](01_load_run_plot.ipynb) — the canonical loop:
  [`Mission.load`][gmat_run.Mission.load] a stock sample, run it, pull the
  resulting `ReportFile` back as a DataFrame, derive altitude, plot.
- [Parameter sweep](02_parameter_sweep.ipynb) — vary `Sat.SMA` across a range
  of values, run the same script for each, and overlay the resulting orbits.
- [Ground track](03_ground_track.ipynb) — read an `EphemerisFile` from
  [`Results.ephemerides`][gmat_run.Results] and plot the spacecraft's ground
  track on a Cartopy world map.
- [Export to CCSDS-OEM](04_export_oem.ipynb) — run a stock GMAT sample that emits
  an STK ephemeris, convert it to a CCSDS-OEM file with
  [`Results.write_oem`][gmat_run.Results], re-parse the result, and visualise the
  trajectory in 3D.
- [Time-scale conversion](05_time_scales.ipynb) — propagate across the 2017-01-01
  leap-second boundary and convert the resulting ReportFile's epoch columns
  between A1, TAI, UTC, TT, and TDB with
  [`gmat_run.time.convert`][gmat_run.time.convert] and the parser-level
  `convert_to=` keyword.
