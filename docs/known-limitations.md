# Known limitations

The list below captures behaviour that surprises users often enough to warrant
a heads-up. Most items are intentional — they fall out of the underlying
`gmatpy` runtime or the GMAT script semantics — and gmat-run does not try to
paper over them.

## gmatpy single-init constraint

`gmatpy` cannot be cleanly reinitialised once it has been loaded into a Python
interpreter. [`gmat_run.bootstrap`][gmat_run.bootstrap] (and therefore
[`Mission.load`][gmat_run.Mission.load]) caches the loaded module and the
[`GmatInstall`][gmat_run.GmatInstall] it was bound to; a second call requesting
a *different* install raises [`GmatLoadError`][gmat_run.GmatLoadError]. Calling
again with the *same* install is a no-op and returns the cached module.

Practical implications:

- A single Python process is bound to one GMAT install for its lifetime. If
  you need to compare missions across GMAT releases, run them in separate
  processes (subprocess, pytest workers, multiprocessing).
- `gmat-run` integration tests that touch the runtime cannot fork
  `xdist` workers within the same process; spawn them.

## Supported Python versions

`gmatpy` ships per-Python-minor shared libraries inside the GMAT install's
`bin/gmatpy/` directory. If your interpreter's minor version does not match
one of the prebuilt wheels in the install, [`bootstrap`][gmat_run.bootstrap]
raises [`GmatLoadError`][gmat_run.GmatLoadError] at import time.

R2026a ships builds for Python 3.10, 3.11, and 3.12. Older Python interpreters
or 3.13+ are not supported by the bundled `gmatpy` and cannot be made to work
without rebuilding GMAT from source.

## Output paths must be set via `Filename`, not `OUTPUT_PATH`

`FileManager.OUTPUT_PATH` and `GmatGlobal.SetOutputPath` look like the right
knobs for redirecting `ReportFile` / `EphemerisFile` / `ContactLocator`
output, but they only take effect at parse time. Once
[`Mission.load`][gmat_run.Mission.load] has parsed a script, every subscriber
has its absolute output path resolved and cached internally; later changes to
the global `OUTPUT_PATH` are ignored at write time.

[`Mission.run`][gmat_run.Mission.run] handles this for you by rewriting each
subscriber's `Filename` field to an absolute path inside the run workspace.
If you bypass [`Mission.run`][gmat_run.Mission.run] and drive
`gmat.RunScript()` yourself, you need to do the same rewrite — set
`<Subscriber>.Filename` to an absolute path before calling `RunScript`.

## GMAT R2026a does not write CCSDS-AEM

GMAT's `EphemerisFile` writer supports CCSDS-OEM, SPK, Code-500, and
STK-TimePosVel only — there is no `FileFormat = CCSDS-AEM`. Attitude history
in GMAT is a *reader-side* concept: a `Spacecraft` resource consumes an
external AEM file via its `AttitudeFileName` field. gmat-run surfaces those
input files via [`Mission.attitude_inputs`][gmat_run.Mission.attitude_inputs]
and parses them with [`gmat_run.parsers.aem_ephemeris.parse`][gmat_run.parsers.aem_ephemeris.parse],
but you cannot ask GMAT to *emit* an AEM trace of a propagated spacecraft.

## Parser format restrictions

The parsers in [`gmat_run.parsers`](reference/parsers.md) cover the formats
GMAT actually emits in v0.2; uncommon variants are rejected with
[`GmatOutputParseError`][gmat_run.GmatOutputParseError] rather than silently
guessed at.

- **CCSDS-OEM ephemeris**: covariance blocks (`COVARIANCE_START` …
  `COVARIANCE_STOP`) are skipped; acceleration columns past the mandatory six
  state components are rejected.
- **STK-TimePosVel ephemeris**: the `EphemerisTimePosVelAcc` (with
  acceleration) and `EphemerisTimePos` (position-only) data-section variants
  are rejected.
- **CCSDS-AEM attitude**: only `QUATERNION` and `EULER_ANGLE` segment types
  parse. Rate/derivative/spin variants
  (`QUATERNION/DERIVATIVE`, `QUATERNION/RATE`, `EULER_ANGLE/RATE`, `SPIN`,
  `SPIN/NUTATION`) are rejected, and multi-segment files mixing different
  `ATTITUDE_TYPE` values are rejected (the column shapes are not
  concatenable).
- **SPK ephemeris**: the parser assumes one spacecraft per file (which
  matches GMAT's writer behaviour). Multi-target SPKs are rejected.
- **Code-500 binary ephemeris**: not implemented. GMAT does not run any of its
  stock R2026a sample missions through this format, and no public tooling
  decodes it; out of scope unless a user need surfaces.

## Epoch promotion is not a time-scale conversion

[`gmat_run.parsers.epoch.promote_epochs`][gmat_run.parsers.epoch.promote_epochs]
turns the ten `{scale}{format}` GMAT epoch columns into
`datetime64[ns]` with the time scale recorded on `df.attrs["epoch_scales"]`.
It does **not** apply leap-second-correct conversion between scales: a
`TAIModJulian` column becomes a `datetime64[ns]` representing the TAI instant,
labelled `"TAI"`. Aligning across scales (UTC ↔ TAI ↔ TT ↔ TDB) is deferred
to a future astropy-extra release.
