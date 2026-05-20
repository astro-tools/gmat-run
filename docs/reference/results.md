# Results

[`Results`][gmat_run.Results] is the return value of
[`Mission.run`][gmat_run.Mission.run]. It exposes four keyed views over GMAT's
output files — `reports`, `ephemerides`, `contacts`, and `solver_runs` — each
typed as `Mapping[str, pandas.DataFrame]` and keyed by the resource name as
declared in the `.script`.

Parsing is lazy: a `ReportFile` listed in `Results.reports` is read from disk
and converted to a DataFrame only on first access, then cached for the life of
the instance.

When [`Mission.run`][gmat_run.Mission.run] was called without a `working_dir`,
the artefacts live under a `tempfile.TemporaryDirectory` that is cleaned up
when the [`Results`][gmat_run.Results] is garbage-collected. Call
[`Results.persist`][gmat_run.Results.persist] to copy the artefacts to a
permanent location before the temp dir disappears.

[`Results.write_oem`][gmat_run.Results.write_oem] and
[`Results.write_oem_all`][gmat_run.Results.write_oem_all] re-emit
ephemerides as CCSDS-OEM files via `ccsds-ndm` (gated behind the
`[ccsds-ndm]` extra). Coordinate-system names are rewritten through a
small alias table on output:

| Source name | Written as | Notes |
|-------------|------------|-------|
| `EME2000`, `GCRF`, `ICRF`, `ITRF2000`, `TEME`, `TOD`, … | (passthrough) | CCSDS 502.0-B-2 §A.5 canonical frames |
| `EarthMJ2000Eq` | `EME2000` | GMAT internal |
| `EarthICRF` | `ICRF` | GMAT internal |
| `J2000` | `EME2000` | STK-TimePosVel convention |

Anything else raises `ValueError` rather than emitting a header
downstream tools cannot interpret. Time systems are restricted to
`A1`, `TAI`, `UTC`, `TT`, and `TDB` — the five GMAT speaks end-to-end.

## Solver runs

`Results.solver_runs` keys a DataFrame by each `Solver` resource a `Target` or
`Optimize` block exercised — `result.solver_runs["DC"]`, `result.solver_runs["Yukon1"]`.
A mission with no solver yields an empty mapping. Each DataFrame has one row
per iteration's nominal pass:

| Column | Meaning |
|--------|---------|
| `iteration` | Iteration number as GMAT reports it. |
| `<variable>` | One column per `Vary` variable, named verbatim from the script (`MOI.Element1`). |
| `status` | `"running"` on every row except the last, which carries `"converged"`, `"max_iter"`, or `"failed"`. |

A `DifferentialCorrector` adds, per `Achieve` goal, the quartet `<goal>` (the
achieved value), `<goal>_desired`, `<goal>_residual` (`achieved - desired`, so a
positive residual means the achieved value overshot the target), and
`<goal>_tolerance`. A `Yukon` optimizer instead adds `cost` (the cost-function
value) and `<constraint>_residual` per nonlinear constraint — its iteration log
records no achieved value, desired bound, or tolerance for a constraint.

`df.attrs` carries `solver_type`, `solver_mode`, `n_iterations`, `n_variables`,
`n_goals`, `converged`, and `source_path`.

```python
runs = result.solver_runs["DC"]
runs.plot(x="iteration", y="geoSat.Earth.SMA_residual")  # watch it home in
final = runs.iloc[-1]
```

`Results.converged` is a `{solver: bool}` shortcut over the same mapping, for
the common branch:

```python
if not result.converged["DC"]:
    raise RuntimeError("targeter did not reach its goal")
```

**Convergence detection.** A `Yukon` run is converged when its log stamps the
optimizer-converged marker. A `DifferentialCorrector` log carries no such
marker — the same "Targeting Completed" line ends a converged and a
non-converged run alike — so convergence is *derived*: the last iteration is
converged when every goal's `abs(residual) <= tolerance`. A non-converged run
that ran out of iterations (`n_iterations` reached the solver's
`MaximumIterations`) is reported as `"max_iter"`; any other non-converged
outcome is `"failed"`.

A run-time GMAT error inside a solver block still raises
[`GmatRunError`][gmat_run.GmatRunError] — `solver_runs` reports solver
*outcomes*, not underlying run failures.

::: gmat_run.Results
