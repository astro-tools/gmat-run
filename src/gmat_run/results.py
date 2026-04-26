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

When :meth:`Mission.run` was called without a ``working_dir``, the artefacts
live under a :class:`tempfile.TemporaryDirectory` that is cleaned up when this
:class:`Results` is garbage-collected. Call :meth:`Results.persist` to copy
the artefacts to a permanent location before the temp dir disappears.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from gmat_run.parsers.contact import parse as _parse_contact
from gmat_run.parsers.ephemeris import parse as _parse_oem_ephemeris
from gmat_run.parsers.reportfile import parse as _parse_reportfile
from gmat_run.parsers.stk_ephemeris import is_stk_ephemeris as _is_stk_ephemeris
from gmat_run.parsers.stk_ephemeris import parse as _parse_stk_ephemeris

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

    def _rebase(self, paths: Mapping[str, Path]) -> None:
        # Replace the underlying path mapping in place. Already-cached
        # DataFrames are kept — they're independent of the on-disk files once
        # parsed.
        self._paths = dict(paths)


class _LazyEphemerides(Mapping[str, pd.DataFrame]):
    """Mapping view over ``EphemerisFile`` outputs.

    Mirrors :class:`_LazyReports` but dispatches between GMAT's two text
    ephemeris formats — CCSDS-OEM and STK-TimePosVel — by sniffing the file's
    first content line. The format choice is per-file rather than per-mapping
    because nothing stops a single ``Mission.run`` from declaring two
    ``EphemerisFile`` resources with different ``FileFormat`` settings.

    Kept as a parallel class rather than factored against a shared base
    because the codebase pattern is explicit one-class-per-output-format
    dispatch.
    """

    def __init__(self, paths: Mapping[str, Path]) -> None:
        self._paths: dict[str, Path] = dict(paths)
        self._cache: dict[str, pd.DataFrame] = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key in self._cache:
            return self._cache[key]
        if key not in self._paths:
            raise KeyError(key)
        path = self._paths[key]
        parser = _parse_stk_ephemeris if _is_stk_ephemeris(path) else _parse_oem_ephemeris
        frame = parser(path)
        self._cache[key] = frame
        return frame

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, key: object) -> bool:
        return key in self._paths

    def _rebase(self, paths: Mapping[str, Path]) -> None:
        self._paths = dict(paths)


class _LazyContacts(Mapping[str, pd.DataFrame]):
    """Mapping view over ``ContactLocator`` outputs.

    Mirrors :class:`_LazyReports`: the parser runs once per key on first
    ``__getitem__``, the resulting DataFrame is cached, and subsequent accesses
    return the same object. Membership and iteration do not parse.

    The DataFrame's columns vary with the resource's
    ``ContactLocator.ReportFormat`` (Legacy vs. one of the five tabular
    variants); ``df.attrs["report_format"]`` carries the variant name so
    downstream code can branch without inspecting the column set.
    """

    def __init__(self, paths: Mapping[str, Path]) -> None:
        self._paths: dict[str, Path] = dict(paths)
        self._cache: dict[str, pd.DataFrame] = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key in self._cache:
            return self._cache[key]
        if key not in self._paths:
            raise KeyError(key)
        frame = _parse_contact(self._paths[key])
        self._cache[key] = frame
        return frame

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, key: object) -> bool:
        return key in self._paths

    def _rebase(self, paths: Mapping[str, Path]) -> None:
        self._paths = dict(paths)


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

    # When the originating Mission.run() created an isolated temp dir, the
    # TemporaryDirectory handle is parked here so cleanup is tied to this
    # instance's GC — keeps the lazy report/ephemeris paths valid until the
    # caller drops the Results.
    _workspace: tempfile.TemporaryDirectory[str] | None

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
        self._workspace = None

        eph_paths: dict[str, Path] = dict(ephemeris_paths or {})
        con_paths: dict[str, Path] = dict(contact_paths or {})

        self.reports = _LazyReports(report_paths or {})
        self.ephemeris_paths = MappingProxyType(eph_paths)
        self.contact_paths = MappingProxyType(con_paths)
        self.ephemerides = _LazyEphemerides(eph_paths)
        self.contacts = _LazyContacts(con_paths)

    def persist(self, path: str | os.PathLike[str]) -> Results:
        """Copy every output artefact under :attr:`output_dir` into ``path``.

        Mutates the :class:`Results` in place so future report/ephemeris/contact
        access reads from the persisted location instead of the (potentially
        soon-to-be-cleaned) workspace. The :class:`tempfile.TemporaryDirectory`
        backing a default-workspace run is released as part of the call. Run
        with an explicit ``working_dir``: that directory is left intact —
        ``persist`` is a copy, never a move.

        Path mappings are rewritten so any path that lived under the old
        ``output_dir`` now points at the matching file under ``path``.
        Absolute filenames the user pinned outside the workspace via
        ``ReportFile.Filename = "/abs/elsewhere.txt"`` are kept as-is — the
        user chose that destination and we honour it. Already-parsed
        DataFrames stay cached; they are independent of the on-disk files.

        Calling ``persist`` again later moves the artefacts to the new
        destination. A no-op fast path applies when the destination already
        equals the current ``output_dir``.

        Args:
            path: Directory to copy artefacts into. Created if missing.

        Returns:
            ``self``, so the call composes with ``Mission.run().persist(...)``.
        """
        dest = Path(path).expanduser()
        if self.output_dir.exists() and dest.resolve() == self.output_dir.resolve():
            return self
        dest.mkdir(parents=True, exist_ok=True)
        if self.output_dir.exists() and self.output_dir.is_dir():
            shutil.copytree(self.output_dir, dest, dirs_exist_ok=True)

        old_dir = self.output_dir

        def _migrate(p: Path) -> Path:
            try:
                rel = p.relative_to(old_dir)
            except ValueError:
                return p
            return dest / rel

        new_reports = {n: _migrate(p) for n, p in self.reports._paths.items()}  # type: ignore[attr-defined]
        new_eph = {n: _migrate(p) for n, p in self.ephemeris_paths.items()}
        new_con = {n: _migrate(p) for n, p in self.contact_paths.items()}

        self.reports._rebase(new_reports)  # type: ignore[attr-defined]
        self.ephemerides._rebase(new_eph)  # type: ignore[attr-defined]
        self.contacts._rebase(new_con)  # type: ignore[attr-defined]
        self.ephemeris_paths = MappingProxyType(new_eph)
        self.contact_paths = MappingProxyType(new_con)

        self.output_dir = dest
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None
        return self
