# API reference

The public Python API is small and stable across the package's top-level
namespace. Everything documented here is re-exported from `gmat_run`; import
from the top level rather than from submodules:

```python
from gmat_run import Mission, Results, GmatError, locate_gmat
```

The reference splits into one page per module:

- [Mission](mission.md) — load a `.script` and run it. The main entry point
  ([`Mission.load`][gmat_run.Mission.load], dotted-path field access,
  [`Mission.run`][gmat_run.Mission.run]).
- [Results](results.md) — the value returned by
  [`Mission.run`][gmat_run.Mission.run]: `reports`, `ephemerides`, `contacts`
  as lazy `Mapping[str, pandas.DataFrame]`, plus
  [`Results.persist`][gmat_run.Results.persist].
- [Install discovery](install.md) —
  [`locate_gmat`][gmat_run.locate_gmat] and the
  [`GmatInstall`][gmat_run.GmatInstall] dataclass.
- [Runtime bootstrap](runtime.md) — [`bootstrap`][gmat_run.bootstrap], the
  `gmatpy` loader behind [`Mission.load`][gmat_run.Mission.load].
- [Exceptions](errors.md) — [`GmatError`][gmat_run.GmatError] and the five
  leaf classes.
- [Parsers](parsers.md) — `gmat_run.parsers`, the standalone functions for
  ReportFile, EphemerisFile (CCSDS-OEM / STK-TimePosVel / SPK), CCSDS-AEM
  attitude, ContactLocator, and epoch-column promotion. You normally do not
  call these directly — [`Mission.run`][gmat_run.Mission.run] dispatches to
  the right one for you.
