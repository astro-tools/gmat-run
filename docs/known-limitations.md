# Known limitations

The list below captures behaviour that surprises users often enough to warrant
a heads-up. Most items fall out of the underlying `gmatpy` runtime or the GMAT
script semantics.

## No built-in wall-clock timeout

[`Mission.run`][gmat_run.Mission.run] does not take a `timeout=` keyword. If
you need a wall-clock cap on a mission run — for example to bound a divergent
solver or a hung subscriber — see [Wall-clock timeouts](timeouts.md) for a
subprocess recipe.

## gmatpy single-init constraint

`gmatpy` cannot be cleanly reinitialised once it has been loaded into a Python
interpreter. [`gmat_run.bootstrap`][gmat_run.bootstrap] (and therefore
[`Mission.load`][gmat_run.Mission.load]) caches the loaded module and the
[`GmatInstall`][gmat_run.GmatInstall] it was bound to; a second call requesting
a *different* install raises [`GmatLoadError`][gmat_run.GmatLoadError]. Calling
again with the *same* install is a no-op and returns the cached module.

Practical implication: a single Python process is bound to one GMAT install
for its lifetime. If you need to compare missions across GMAT releases, run
them in separate processes (subprocess, pytest workers, multiprocessing).

## Supported Python versions

`gmatpy` ships per-Python-minor shared libraries inside the GMAT install's
`bin/gmatpy/` directory. If your interpreter's minor version does not match
one of the prebuilt wheels in the install, [`bootstrap`][gmat_run.bootstrap]
raises [`GmatLoadError`][gmat_run.GmatLoadError] at import time.

R2026a ships builds for Python 3.10, 3.11, and 3.12. Older Python interpreters
or 3.13+ are not supported by the bundled `gmatpy` and cannot be made to work
without rebuilding GMAT from source.

## Canonical-image drift is covered by a single CI canary

The main `test` matrix runs in action-mode (`astro-tools/setup-gmat@v0`) on
bare runners — it does not exercise the canonical `ghcr.io/astro-tools/gmat`
container image. A separate `canonical-image-smoke` CI cell runs `pip install
gmat-run` and a small `Mission.load → run → assert non-empty ReportFile`
round-trip inside `ghcr.io/astro-tools/gmat:R2026a` on every PR and every
push to `main`. It is the canary for image-vs-action drift: a stripped
plugin, a mangled `gmat_startup_file.txt`, a dropped Python ABI from the
bundled `gmatpy` set, or a tag that silently retargets a different GMAT
upload all surface here before downstream consumers running container-mode
CI hit them. The cell is pinned to `:R2026a`; the matrix coverage of the
image across multiple GMAT releases lives in setup-gmat itself.

## R2022a is not exercised in CI

R2022a's bundled `gmatpy` tops out at Python 3.9 across all three OSes, while
gmat-run pins `python>=3.10` in `pyproject.toml`. The two constraints are
incompatible — a CI cell on R2022a would have to drop the Python floor for
gmat-run as a whole or pin a one-off interpreter just for that cell. Neither
trade-off pays for itself given how few users run R2022a today.

R2022a is therefore listed as "expected to work" but is not exercised in CI.
If you depend on R2022a and hit a regression, file an issue and we'll add a
targeted cell.

## Some integration tests are R2026a-only

The integration suite under `tests/integration/` runs across the full CI
matrix, but a small handful of tests skip on GMAT releases other than R2026a:

- `test_attitude_inputs_*` — `Mission.attitude_inputs` walks `Spacecraft`
  resources via gmatpy's `GetField("Attitude")` accessor, whose return shape
  has drifted between releases. The discovery path is exercised on R2026a;
  tracking every release's accessor isn't worth the test churn.
- `test_sample_round_trip[Ex_ContactLocatorAllFormats]` — contact start/stop
  epochs drift by ~1 ms between R2025a and R2026a, and
  `pandas.testing.assert_frame_equal` ignores `rtol`/`atol` on datetime
  columns. Skipped rather than papered over with a coarser comparator.

Position-only ephemeris round-trips (`Ex_LEOEphemeris`, `Ex_STKEphemeris`)
run on every release with looser tolerances on non-R2026a runs to absorb
integrator/ephemeris drift; R2026a keeps strict tolerances so the regression
signal there isn't slackened.

If you're running the suite locally against R2025a (or any non-primary
release), expect a handful of `SKIPPED` lines for these tests. Real
gmat-run regressions still surface elsewhere.

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
GMAT actually emits in v0.3; uncommon variants raise
[`GmatOutputParseError`][gmat_run.GmatOutputParseError].

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
- **Code-500 binary ephemeris**: not implemented. No public tooling decodes
  the format, and GMAT does not exercise it in its stock R2026a samples.

## Epoch promotion is not a time-scale conversion

[`gmat_run.parsers.epoch.promote_epochs`][gmat_run.parsers.epoch.promote_epochs]
turns the ten `{scale}{format}` GMAT epoch columns into
`datetime64[ns]` with the time scale recorded on `df.attrs["epoch_scales"]`.
It does **not** apply leap-second-correct conversion between scales: a
`TAIModJulian` column becomes a `datetime64[ns]` representing the TAI instant,
labelled `"TAI"`. For converting between scales after promotion, see
[`gmat_run.time`](reference/time.md), which routes A1/TAI/UTC/TT/TDB through
[`astropy.time.Time`](https://docs.astropy.org/en/stable/time/) and is gated
behind the `[astropy]` extra.
