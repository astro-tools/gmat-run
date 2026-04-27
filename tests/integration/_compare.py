"""Comparison helpers shared by integration tests.

Kept tiny on purpose — this module exists so that round-trip tests touching
GMAT output (CSV goldens, OEM goldens) don't independently re-implement the
same datetime-precision dance every time pandas grows a stricter dtype check.
"""

from __future__ import annotations

import pandas as pd


def truncate_datetime_to_ms(df: pd.DataFrame) -> pd.DataFrame:
    """Floor every datetime64 column to millisecond resolution.

    Goldens round-trip through millisecond-precision text on serialise; the
    actual frame must match that resolution before
    :func:`pandas.testing.assert_frame_equal` can compare them with strict
    dtype checks.

    Timedelta columns are intentionally left alone — the contact-format
    goldens write duration as float seconds via ``dt.total_seconds()`` and
    ``%.15g``, which is round-trip-symmetric back through
    ``pd.to_timedelta(unit='s')``.
    """
    out: pd.DataFrame = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.floor("ms").astype("datetime64[ns]")
    return out
