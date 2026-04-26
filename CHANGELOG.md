# Changelog

All notable changes to gmat-run are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/astro-tools/gmat-run/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/astro-tools/gmat-run/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/astro-tools/gmat-run/releases/tag/v0.1.0
