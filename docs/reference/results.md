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

::: gmat_run.Results
