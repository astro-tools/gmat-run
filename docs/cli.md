# Command-line interface

A minimal `gmat-run` console script ships alongside the Python API. The Python
API remains the primary interface; the CLI exists for shell-script and
smoke-test use — for example, running a saved `.script` from a Makefile or a CI
job and persisting the outputs to a known directory.

## Invocation

```bash
gmat-run --version
gmat-run --help
gmat-run run --help
```

`--version` prints the installed package version and exits. `--help` prints
the top-level usage; `run --help` prints the subcommand's flags.

## `gmat-run run`

```text
gmat-run run SCRIPT [--out DIR] [--gmat-root PATH]
```

Loads `SCRIPT`, runs its mission sequence headlessly, and prints a one-line
summary plus a per-output-type breakdown to stdout.

| Argument | Description |
| --- | --- |
| `SCRIPT` | Path to a GMAT `.script` file. |
| `--out DIR` | Persist the run's outputs into `DIR`. The directory is created if missing. Without this flag, GMAT writes into a temporary workspace that is cleaned up when the process exits. |
| `--gmat-root PATH` | Path to the GMAT install. Overrides discovery and `$GMAT_ROOT`. |

### Sample output

```text
$ gmat-run run flyby.script --out results/
Output directory: results
Reports:
  ReportFile1: 1441 rows
Ephemerides:
  EphemerisFile1: 1441 rows
Contacts: (none)
```

Row counts are computed by parsing each output file, the same way
[`Results`][gmat_run.Results] would in Python — accepted cost for the CLI's
summary. Use the Python API if you need only the file paths.

## Exit codes

Stable per the issue spec, so shell scripts can branch on failure mode without
parsing stderr:

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | Unexpected error. |
| `2` | [`GmatNotFoundError`][gmat_run.GmatNotFoundError] — no usable GMAT install. |
| `3` | [`GmatLoadError`][gmat_run.GmatLoadError] — `gmatpy` could not be loaded. |
| `4` | [`GmatRunError`][gmat_run.GmatRunError] — the mission sequence itself failed. |
| `5` | [`GmatOutputParseError`][gmat_run.GmatOutputParseError] — an output file could not be parsed. |

Argparse uses code `2` for argument errors (missing `SCRIPT`, unknown flag).
That overlaps with `GmatNotFoundError`; the script can disambiguate from
stderr if it cares.
