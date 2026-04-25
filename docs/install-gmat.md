# Install GMAT

gmat-run does not ship GMAT binaries. Install GMAT separately from
[gmat.gsfc.nasa.gov](https://gmat.gsfc.nasa.gov/) or directly from
[SourceForge](https://sourceforge.net/projects/gmat/files/GMAT/).

## Supported versions

| GMAT version    | Status                                                                  |
| --------------- | ----------------------------------------------------------------------- |
| R2026a          | Primary development target; exercised in CI on every PR.                |
| R2024a, R2025a  | Supported; exercised in release CI.                                     |
| R2022a, R2023a  | Supported on a best-effort basis.                                       |
| Earlier         | Not supported. gmatpy ships per-Python-minor shared libraries, and    |
|                 | older releases may not match Python 3.10+.                              |

## Windows

1. Download the Windows zip from
   [SourceForge](https://sourceforge.net/projects/gmat/files/GMAT/) (e.g.
   `gmat-win-R2026a.zip`).
2. Extract anywhere — `C:\Program Files\GMAT-R2026a\` is typical.
3. gmat-run discovers installs under `C:\Program Files\GMAT*` and
   `C:\Program Files (x86)\GMAT*` automatically.

## Linux

1. Download the Linux tarball (e.g. `gmat-ubuntu-x64-R2026a.tar.gz`).
2. Extract anywhere — `~/gmat-R2026a/` or `/opt/gmat-R2026a/` is typical.
3. gmat-run discovers installs matching `~/gmat-*` and `/opt/gmat-*` automatically.

## macOS

1. Download the macOS package from SourceForge.
2. Install into `/Applications/GMAT-R2026a/` or similar.
3. gmat-run discovers installs under `/Applications/GMAT*` automatically.

## Custom location

If your install lives elsewhere, point gmat-run at it explicitly:

```python
mission = Mission.load("flyby.script", gmat_root="/path/to/gmat-R2026a")
```

Or set the `GMAT_ROOT` environment variable, which gmat-run honours globally:

```bash
export GMAT_ROOT=/path/to/gmat-R2026a
```

## Verifying the install

```python
from gmat_run import locate_gmat

install = locate_gmat()
print(install.root, install.version)
```

If discovery fails, [`GmatNotFoundError`][gmat_run.GmatNotFoundError] lists every
search location it tried and why each was rejected — paste that output verbatim
when filing an install issue.

## API startup file

On first use, gmat-run generates `<install>/bin/api_startup_file.txt` by running
`<install>/api/BuildApiStartupFile.py`. This is automatic — no manual step is
needed.
