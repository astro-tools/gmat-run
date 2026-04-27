# Results

[`Results`][gmat_run.Results] is the return value of
[`Mission.run`][gmat_run.Mission.run]. It exposes three keyed views over GMAT's
output files — `reports`, `ephemerides`, and `contacts` — each typed as
`Mapping[str, pandas.DataFrame]` and keyed by the resource name as declared in
the `.script`.

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

::: gmat_run.Results
