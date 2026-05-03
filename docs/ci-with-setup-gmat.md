# Run gmat-run in your CI

[`astro-tools/setup-gmat`](https://github.com/astro-tools/setup-gmat) installs
GMAT on the runner, caches it across runs, and exports `GMAT_ROOT` to the
workflow environment. With it, a complete CI workflow for a project that uses
gmat-run is four steps.

## Minimal workflow

Drop this into `.github/workflows/test.yml`:

```yaml
name: test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - uses: astro-tools/setup-gmat@v0
        with:
          version: R2026a
          cache: true

      - run: pip install gmat-run

      - run: pytest
```

`setup-gmat` runs `BuildApiStartupFile.py` and a one-line gmatpy smoke check
before returning, then exports `GMAT_ROOT` for downstream steps. gmat-run picks
up `GMAT_ROOT` automatically — no `--gmat-root` flag and no `Mission.load(...,
gmat_root=...)` argument needed in the test code.

If your project does not have a `pytest` suite yet, swap the last step for a
one-line discovery smoke that proves the install is wired up:

```yaml
      - run: python -c "from gmat_run import locate_gmat; print(locate_gmat().version)"
```

## Optional extras

Some gmat-run features are gated behind extras. Install the ones you need in
the same step as gmat-run itself:

```yaml
      - run: pip install 'gmat-run[spiceypy,ccsds-ndm,astropy]'
```

| Extra          | Pulls in                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `[spiceypy]`   | SPK ephemeris parsing in [`Results.ephemerides`][gmat_run.Results].         |
| `[ccsds-ndm]`  | CCSDS-OEM export via [`Results.write_oem`][gmat_run.Results.write_oem].     |
| `[astropy]`    | [`gmat_run.time`](reference/time.md) leap-second-correct time-scale conversion. |

The extras compose; pass several in one bracketed list as above.

## Caching

`cache: true` is the default and caches the resolved GMAT install across runs,
keyed on `(action major version, GMAT version, runner OS, runner arch,
installer SHA-256)`. First run on each `(OS, version)` pair is a cold download
+ extract (a few minutes); subsequent runs restore from cache in seconds. The
key includes the installer hash, so an upstream installer rebuild invalidates
the cache automatically.

## Multi-version testing

Add a `version:` axis to your matrix to exercise multiple GMAT releases:

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        gmat-version: [R2025a, R2026a]
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - uses: astro-tools/setup-gmat@v0
        with:
          version: ${{ matrix.gmat-version }}
          cache: true
      - run: pip install gmat-run
      - run: pytest
```

Each cell gets its own cache entry, so adding a version costs a one-time cold
install per OS.

The `(runner, GMAT version, Python version)` cells that setup-gmat actually
ships are listed in
[setup-gmat's README](https://github.com/astro-tools/setup-gmat#supported-versions),
along with the Python-ABI-per-release pinning rule — pin Python to a version
that the targeted GMAT release shipped gmatpy bindings for, or
`BuildApiStartupFile.py` will fail at install time with a missing `_pyXYZ`
module.
