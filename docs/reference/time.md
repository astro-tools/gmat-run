# Time scales

Leap-second-correct conversion between the five GMAT time scales (A1, TAI,
UTC, TT, TDB). All conversion goes through
[`astropy.time.Time`](https://docs.astropy.org/en/stable/time/), which owns
the IERS leap-second table; gmat-run does not bundle leap-second data of its
own.

A1 is GMAT-specific — astropy does not recognise it. Per the GMAT Mathematical
Specification, A1 leads TAI by a fixed 0.0343817 s; this module routes A1
through TAI by applying that offset before/after the astropy conversion.

This module is gated behind the `[astropy]` extra. Importing the module
without astropy installed is fine; calling the conversion functions raises a
clear `ImportError` pointing at the extra.

```bash
pip install gmat-run[astropy]
```

## Quick reference

```python
import pandas as pd
from gmat_run import Mission
from gmat_run.time import convert, convert_column

mission = Mission.load("flyby.script")
result = mission.run()

df = result.reports["ReportFile1"]   # df.attrs["epoch_scales"] is set by promote_epochs

# Series-level: convert one column from its native scale to UTC.
df["Sat.TAIGregorian"] = convert(df["Sat.TAIGregorian"], "TAI", "UTC")

# DataFrame-level: convert and update df.attrs["epoch_scales"] in one call.
convert_column(df, "Sat.TAIGregorian", "UTC")
```

::: gmat_run.time.convert

::: gmat_run.time.convert_column
