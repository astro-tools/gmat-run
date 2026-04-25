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

The ``.eph`` ephemeris and ``ContactLocator`` parsers are scheduled for v0.2.
For v0.1 the corresponding mappings still expose their keys (so callers can
discover what GMAT wrote without branching on the version), but accessing a
value raises :class:`NotImplementedError` pointing at the parallel
:attr:`Results.ephemeris_paths` / :attr:`Results.contact_paths` mappings, which
hand back the raw :class:`pathlib.Path`.
"""

from __future__ import annotations

import os
import shutil
import tempfile
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

    def _rebase(self, paths: Mapping[str, Path]) -> None:
        # Replace the underlying path mapping in place. Already-cached
        # DataFrames are kept — they're independent of the on-disk files once
        # parsed.
        self._paths = dict(paths)


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
