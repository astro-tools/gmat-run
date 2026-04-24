"""In-memory aggregate of every output file GMAT wrote during a run.

:class:`Results` is the return value of :meth:`Mission.run`. It exposes three
keyed views over those outputs — :attr:`Results.reports`,
:attr:`Results.ephemerides`, and :attr:`Results.contacts` — each typed as a
``Mapping[str, pandas.DataFrame]`` and keyed by the GMAT resource name as
declared in the ``.script``.

Parsing is lazy. A ``ReportFile`` listed in :attr:`Results.reports` is read
from disk and converted to a DataFrame only on first access, then cached for
the life of the :class:`Results` instance — opening a notebook on a long run
without touching every report should not pay the parse cost for the ones it
does not look at.

The ``.eph`` ephemeris and ``ContactLocator`` parsers are scheduled for v0.2.
For v0.1 the corresponding mappings still expose their keys (so callers can
discover what GMAT wrote without branching on the version), but accessing a
value raises :class:`NotImplementedError` pointing at the parallel
:attr:`Results.ephemeris_paths` / :attr:`Results.contact_paths` mappings, which
hand back the raw :class:`pathlib.Path`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from gmat_run.parsers.reportfile import parse as _parse_reportfile

__all__ = ["Results"]


class _LazyReports(Mapping[str, pd.DataFrame]):
    """Mapping view over ``ReportFile`` outputs that parses on first access.

    The DataFrame for a given key is materialised by
    :func:`gmat_run.parsers.reportfile.parse` on the first ``__getitem__`` and
    cached on the instance; subsequent accesses return the same object. The
    parser is *not* invoked at construction, so building :class:`Results` is
    cheap even when the underlying files are large or absent.
    """

    def __init__(self, paths: Mapping[str, Path]) -> None:
        self._paths: dict[str, Path] = dict(paths)
        self._cache: dict[str, pd.DataFrame] = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key in self._cache:
            return self._cache[key]
        if key not in self._paths:
            raise KeyError(key)
        frame = _parse_reportfile(self._paths[key])
        self._cache[key] = frame
        return frame

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, key: object) -> bool:
        # Override the Mapping default — it calls __getitem__ and would parse.
        return key in self._paths


class _DeferredMapping(Mapping[str, pd.DataFrame]):
    """Mapping view whose keys are populated but value access is unimplemented.

    Used for output formats whose parsers are scheduled for a later release:
    the keys reflect what GMAT wrote so callers can iterate, ``len``, and
    membership-test as normal, but ``__getitem__`` raises
    :class:`NotImplementedError` directing the caller at the parallel
    ``*_paths`` mapping for the raw file path.
    """

    def __init__(
        self,
        paths: Mapping[str, Path],
        *,
        format_label: str,
        target_release: str,
        paths_attr: str,
    ) -> None:
        self._paths: dict[str, Path] = dict(paths)
        self._format_label = format_label
        self._target_release = target_release
        self._paths_attr = paths_attr

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key not in self._paths:
            raise KeyError(key)
        raise NotImplementedError(
            f"{self._format_label} parsing lands in {self._target_release}; "
            f"use Results.{self._paths_attr}[{key!r}] for the file path"
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, key: object) -> bool:
        # Override the Mapping default — it calls __getitem__ and would raise
        # NotImplementedError for known keys.
        return key in self._paths


class Results:
    """Aggregate of every output file GMAT wrote during a single run.

    Construct one per call to :meth:`Mission.run`. Each path mapping is keyed
    by the resource name declared in the ``.script`` (``"ReportFile1"``,
    ``"EphemerisFile1"``, ``"ContactLocator1"``, …). Path mappings are
    defensively copied and re-exposed as read-only views, so callers cannot
    mutate the run record after the fact.

    Args:
        output_dir: The working directory GMAT used for this run. Surfaced
            so callers can locate any output file gmat-run did not aggregate
            itself.
        log: GMAT's stdout and stderr captured during the run, joined into a
            single string.
        report_paths: ``{name: path}`` for every ``ReportFile`` resource.
            Defaults to empty.
        ephemeris_paths: ``{name: path}`` for every ``EphemerisFile``
            resource. Defaults to empty.
        contact_paths: ``{name: path}`` for every ``ContactLocator``
            resource. Defaults to empty.
    """

    output_dir: Path
    log: str
    reports: Mapping[str, pd.DataFrame]
    ephemerides: Mapping[str, pd.DataFrame]
    ephemeris_paths: Mapping[str, Path]
    contacts: Mapping[str, pd.DataFrame]
    contact_paths: Mapping[str, Path]

    def __init__(
        self,
        *,
        output_dir: Path,
        log: str,
        report_paths: Mapping[str, Path] | None = None,
        ephemeris_paths: Mapping[str, Path] | None = None,
        contact_paths: Mapping[str, Path] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.log = log

        eph_paths: dict[str, Path] = dict(ephemeris_paths or {})
        con_paths: dict[str, Path] = dict(contact_paths or {})

        self.reports = _LazyReports(report_paths or {})
        self.ephemeris_paths = MappingProxyType(eph_paths)
        self.contact_paths = MappingProxyType(con_paths)
        self.ephemerides = _DeferredMapping(
            eph_paths,
            format_label=".eph ephemeris",
            target_release="v0.2",
            paths_attr="ephemeris_paths",
        )
        self.contacts = _DeferredMapping(
            con_paths,
            format_label="ContactLocator",
            target_release="v0.2",
            paths_attr="contact_paths",
        )
