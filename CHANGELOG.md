# Changelog

All notable changes to gmat-run are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-04-28

### Added

- CCSDS-OEM export — `Results.write_oem(name, path)` writes any ephemeris
  DataFrame in `Results.ephemerides` as a CCSDS-OEM file, gated behind the
  new `[ccsds-ndm]` extra (#67). A CI-enforced golden round-trip pins the
  writer against GMAT's own OEM output (#68).
- Time-scale conversion — `gmat_run.time` routes A1 / TAI / UTC / TT / TDB
  epochs through `astropy.time.Time` (leap-second-correct), gated behind
  the new `[astropy]` extra (#65). `promote_epochs(..., convert_to=...)`
  and the matching keyword on the four file-format parsers compose the
  conversion into a single DataFrame access (#66).
- Hardened explicit workspaces — `Mission.run(working_dir=...)` now handles
  pre-existing artefacts (with `overwrite=True` for re-runs into a populated
  workspace), absolute `Filename` fields in the script, and permission
  errors, surfacing each as a typed `GmatRunError` (#69).
- New optional extras `[ccsds-ndm]` and `[astropy]`, documented in the
  README install table (#64).
- Example notebooks, runnable end-to-end and rendered into the docs site:
  export an ephemeris to CCSDS-OEM (#70), and time-scale conversion across
  a leap-second boundary (#71).
- Wall-clock timeout cookbook page (`docs/timeouts.md`) — a subprocess
  recipe for capping `Mission.run` from caller code, with a cross-link
  from "Known limitations".

### Changed

- `Mission.__setitem__` accepts numpy scalars (`np.float64`, `np.int_`,
  `np.bool_`) and `np.ndarray` directly. Previously the call site had to
  do `float(...)` / `.tolist()` first (#85).

### Fixed

- `gmat_run.time.convert(..., to_scale="UTC")` (and transitively
  `convert_column`, `promote_epochs(..., convert_to=)`, and the parser-level
  `convert_to=` keyword) raised `ValueError` from astropy's
  `Time.utc.datetime64` accessor on epochs landing exactly on a leap-second
  instant. Non-leap rows keep nanosecond precision; leap-second rows fall
  back to microsecond precision and are pinned to the post-jump second to
  match GMAT's labelling convention (#80, #81).

## [0.2.0] — 2026-04-26

### Added

- `EphemerisFile` parser → `pandas.DataFrame`, dispatching on file format:
  CCSDS-OEM (#46), STK-TimePosVel (#51), and SPK via the `[spiceypy]` extra (#60).
- CCSDS-AEM attitude ephemeris parser, exposed through `Mission.attitude_inputs`
  for files referenced by a `Spacecraft.AttitudeFileName` (#52).
- `ContactLocator` parser → `Results.contacts`, supporting the Legacy report and
  the five tabular `ReportFormat` variants. `df.attrs["report_format"]` carries
  the variant so downstream code can branch without inspecting columns (#53).
- `gmat-run` CLI: `gmat-run run SCRIPT [--out DIR]` runs a script headlessly
  and writes outputs to a chosen directory, with documented exit codes (#54).
- Example notebooks, runnable end-to-end and rendered into the docs site:
  load / run / plot a stock GMAT sample (#55), parameter sweep across
  `Sat.SMA` (#56), and a ground-track plot from `EphemerisFile` (#57).
- macOS runner in the CI test matrix (#59).
- Coverage gating in CI: ≥ 80 % overall and ≥ 95 % on
  `src/gmat_run/parsers/`, both enforced on the Ubuntu / Python 3.12 cell (#59).
- API reference site polish: per-module navigation, a "Known limitations"
  page, and full coverage of the public API surface (#61).

### Changed

- README and getting-started docs refreshed for v0.2 — showcases ephemeris
  and contact support, links every example notebook, and updates the
  supported-GMAT / CI matrix tables (#62, #63).

## [0.1.1] — 2026-04-25

### Fixed

- Stale install instructions in README and the docs site (#36).

## [0.1.0] — 2026-04-24

Initial public release.

### Added

- `Mission.load` parses a `.script` into the live GMAT object graph;
  dotted-path subscript access reads and writes fields with type coercion (#26).
- `Mission.run` executes the mission sequence headlessly, captures GMAT's
  stdout/stderr, and surfaces failures as typed exceptions (#27).
- `Results` exposes lazy, name-keyed mappings over GMAT's output files
  (#25), with `Results.persist` to copy artefacts out of the temp
  workspace (#28).
- `ReportFile` parser → `DataFrame`, with `UTCGregorian` and `*ModJulian`
  epoch columns promoted to `datetime64[ns]` (#21, #22).
- `locate_gmat` cross-platform GMAT install discovery and `bootstrap` to
  load `gmatpy` for a resolved install (#18, #19).
- Typed exception hierarchy under `gmat_run.errors` rooted at
  `GmatError` (#20).
- CI on Ubuntu and Windows (Python 3.10 / 3.11 / 3.12), with integration
  tests against stock GMAT samples (#17, #29).
- MkDocs-Material documentation site, auto-deployed to GitHub Pages on
  tag pushes (#30).
- Release workflow: build, PyPI trusted publishing, and
  `gh release create --generate-notes` on `v*` tags (#31).

[0.3.0]: https://github.com/astro-tools/gmat-run/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/astro-tools/gmat-run/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/astro-tools/gmat-run/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/astro-tools/gmat-run/releases/tag/v0.1.0
