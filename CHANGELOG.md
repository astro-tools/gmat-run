# Changelog

All notable changes to gmat-run are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-05-20

### Added

- Solver/targeter introspection — [`Results.solver_runs`][gmat_run.Results]
  exposes a lazy mapping keyed by `Solver` resource name, one
  `pandas.DataFrame` per `Target` / `Optimize` run. Each frame carries one row
  per iteration with a column for every `Vary` variable, the goal/constraint
  residuals, and a terminal `status`; [`Results.converged`][gmat_run.Results]
  is the derived `{solver: bool}` shortcut. A new pure parser,
  `gmat_run.parsers.solver_log`, reads the per-`Solver` `.data` file for
  `DifferentialCorrector` and `Yukon` with no `gmatpy` import (#106, #112).
- Extended `mission[...]` override grammar — multi-dot sub-resource paths
  (`mission["FM.Drag.CSSISpaceWeatherFile"] = ...`) and script `Variable`
  values (`mission["elapsed_seconds.Value"] = ...`), alongside the existing
  `Resource.Field` form (#103, #104, #111).
- `Results.report_paths` — a read-only mapping of `ReportFile` resource name
  to its output path, completing the parser-free path accessors next to
  `ephemeris_paths`, `contact_paths`, and `solver_paths` (#117, #125).
- Example notebook *Target a Hohmann transfer and inspect solver iterations*
  — runs a two-burn transfer and reads the `DifferentialCorrector` history
  back through `Results.solver_runs`, including a `MaximumIterations`-capped
  run that ends non-converged without raising (#107, #113).

### Changed

- Every public path-shaped argument — [`Mission.load`][gmat_run.Mission.load],
  the `working_dir` of [`Mission.run`][gmat_run.Mission.run],
  [`Results.persist`][gmat_run.Results.persist], and `Results.write_oem` /
  `write_oem_all` — now expands `~` and resolves a relative path against the
  caller's working directory when the call is made. Stored path attributes
  (`Mission.script_path`, `Results.output_dir`) are always absolute. Callers
  already passing absolute paths are unaffected (#105, #110, #120).

### Fixed

- [`Mission.summary`][gmat_run.Mission.summary] (and the `Mission` / `Results`
  notebook reprs) crashed the interpreter on any mission containing a branch
  command — `Target`, `If`, `For`, and the like — and listed
  `BeginMissionSequence` as a spurious first command. The command-tree walk
  now stops at each `BranchEnd`, enumerates branches explicitly, and consumes
  GMAT's full `NoOp -> BeginMissionSequence` sentinel prefix (#114, #116, #123).
- Running the same `Mission` twice wrote the second run's output into the
  first run's directory: the output-path rewrite leaked onto the engine object
  between runs. [`Mission.run`][gmat_run.Mission.run] now restores every
  rewritten field after the run (including after a failure), so each run
  resolves its outputs afresh and a `Mission` stays a view of the loaded
  script — reading a subscriber's `Filename` back yields the value declared in
  the script, not the resolved workspace path (#115, #124).
- Relative subscriber and solver output paths were flattened to a bare
  filename, so two outputs declared with distinct relative paths that shared a
  basename collided onto one file. Resolution now preserves the subdirectory
  structure under the run workspace (nested parents are pre-created), rejects a
  relative `Filename` containing a `..` component, and raises when two outputs
  of one run resolve to the same path (#119, #127).
- `promote_epochs` re-tagged an already-converted epoch column with the scale
  derived from its name, so a promote → convert → promote sequence could
  silently mislabel the time scale. A scale recorded by an earlier pass or by
  `convert_column` is now preserved (#118, #126).
- [`bootstrap`][gmat_run.bootstrap] rejected a second load of the same GMAT
  install when it was reached through a different discovery route or a
  symlink; the install root is now canonicalised before the comparison.
  Separately, the [`Results.persist`][gmat_run.Results.persist] docstring no
  longer claims the method can "move" artefacts — it only ever copies
  (#121, #122, #128).
- `CONTRIBUTING.md` linked to a per-repo Discussions board that does not
  exist; it now points at the org-wide `astro-tools` Discussions space
  (#102, #109).

## [0.4.0] — 2026-05-03

### Added

- `Mission.summary()` returns a [`MissionSummary`][gmat_run.MissionSummary]
  with the resolved GMAT install, the parsed [`ResourceGroup`][gmat_run.ResourceGroup]s
  and [`CommandOutline`][gmat_run.CommandOutline] for the mission sequence.
  `Mission` and `Results` both gain `_repr_html_` so a notebook cell that
  ends in either renders a structured table instead of a `<…>` repr (#91, #99).
- `docs/ci-with-setup-gmat.md` — a cookbook page showing how to wire
  `astro-tools/setup-gmat@v0` into a downstream project's GitHub Actions
  workflow, including caching, optional extras, and a multi-version
  matrix (#89, #97).

### Changed

- CI now uses `astro-tools/setup-gmat@v0` instead of an inline GMAT install
  step. The action handles download, cache, `BuildApiStartupFile.py`, and
  `GMAT_ROOT` export — the workflow shrinks to one step (#87, #95).
- CI matrix expanded from one GMAT release to two: every PR runs the full
  cross-product of Ubuntu / Windows / macOS × Python 3.10 / 3.11 / 3.12 ×
  R2025a / R2026a (18 cells). Both releases are now first-class supported
  (#88, #96).
- Overall coverage gate raised from ≥ 80% to ≥ 90% on the
  Ubuntu / Python 3.12 / R2026a cell (#90, #98).
- New `canonical-image-smoke` CI cell exercises `pip install gmat-run`
  and a `Mission.load → run → assert non-empty ReportFile` round-trip
  inside `ghcr.io/astro-tools/gmat:R2026a` on every PR. The action and
  the canonical image can drift apart silently; this is the canary that
  catches it before downstream container-mode CI users do (#93, #100).

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

[0.5.0]: https://github.com/astro-tools/gmat-run/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/astro-tools/gmat-run/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/astro-tools/gmat-run/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/astro-tools/gmat-run/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/astro-tools/gmat-run/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/astro-tools/gmat-run/releases/tag/v0.1.0
