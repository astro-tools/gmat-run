"""In-memory aggregate of every output file GMAT wrote during a run.

:class:`Results` is the return value of :meth:`Mission.run`. It exposes four
keyed views over those outputs — :attr:`Results.reports`,
:attr:`Results.ephemerides`, :attr:`Results.contacts`, and
:attr:`Results.solver_runs` — each typed as a ``Mapping[str, pandas.DataFrame]``
and keyed by the GMAT resource name as declared in the ``.script``.
:attr:`Results.converged` is a derived ``{solver: bool}`` shortcut over
:attr:`Results.solver_runs`.

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

from gmat_run._path_utils import resolve_user_path
from gmat_run.parsers.contact import parse as _parse_contact
from gmat_run.parsers.ephemeris import parse as _parse_oem_ephemeris
from gmat_run.parsers.reportfile import parse as _parse_reportfile
from gmat_run.parsers.solver_log import parse as _parse_solver_log
from gmat_run.parsers.spk import is_spk_ephemeris as _is_spk_ephemeris
from gmat_run.parsers.spk import parse as _parse_spk_ephemeris
from gmat_run.parsers.stk_ephemeris import is_stk_ephemeris as _is_stk_ephemeris
from gmat_run.parsers.stk_ephemeris import parse as _parse_stk_ephemeris
from gmat_run.summary import format_results_html as _format_results_html
from gmat_run.writers.oem import write_oem as _write_oem

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

    Mirrors :class:`_LazyReports` but dispatches between GMAT's three
    ephemeris formats — SPK, STK-TimePosVel, and CCSDS-OEM — by sniffing
    the file's content. SPK and STK are positively detected (DAF/SPK
    magic and the ``stk.v.X.Y`` banner respectively); CCSDS-OEM is the
    fallback. The format choice is per-file rather than per-mapping
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
        if _is_spk_ephemeris(path):
            frame = _parse_spk_ephemeris(path)
        elif _is_stk_ephemeris(path):
            frame = _parse_stk_ephemeris(path)
        else:
            frame = _parse_oem_ephemeris(path)
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


class _LazySolverRuns(Mapping[str, pd.DataFrame]):
    """Mapping view over ``Solver`` iteration logs (the per-solver ``.data`` file).

    Mirrors :class:`_LazyReports`: the parser runs once per key on first
    ``__getitem__``, the DataFrame is cached, and membership/iteration do not
    parse. Each key carries both its file path and the solver's
    ``MaximumIterations`` — the ``.data`` file does not record the latter, but
    :func:`gmat_run.parsers.solver_log.parse` needs it to tell a max-iteration
    stop apart from a generic failure.

    The DataFrame's columns depend on the solver type (``df.attrs["solver_type"]``):
    a ``DifferentialCorrector`` carries the goal quartet, a ``Yukon`` carries
    ``cost`` and per-constraint residuals. See
    :func:`gmat_run.parsers.solver_log.parse` for the full schema.
    """

    def __init__(self, paths: Mapping[str, Path], max_iterations: Mapping[str, int]) -> None:
        self._paths: dict[str, Path] = dict(paths)
        self._max_iterations: dict[str, int] = dict(max_iterations)
        self._cache: dict[str, pd.DataFrame] = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key in self._cache:
            return self._cache[key]
        if key not in self._paths:
            raise KeyError(key)
        frame = _parse_solver_log(self._paths[key], max_iterations=self._max_iterations.get(key))
        self._cache[key] = frame
        return frame

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, key: object) -> bool:
        return key in self._paths

    def _rebase(self, paths: Mapping[str, Path]) -> None:
        # Replace the path mapping in place; MaximumIterations is intrinsic to
        # the run, not the file location, so it is left untouched.
        self._paths = dict(paths)


class Results:
    """Aggregate of every output file GMAT wrote during a single run.

    Construct one per call to :meth:`Mission.run`. Each path mapping is keyed
    by the resource name declared in the ``.script`` (``"ReportFile1"``,
    ``"EphemerisFile1"``, ``"ContactLocator1"``, ``"DC"``, …). Path mappings are
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
        solver_paths: ``{name: path}`` for every ``Solver`` resource whose
            ``.data`` iteration log was found on disk after the run. Defaults
            to empty.
        solver_max_iterations: ``{name: MaximumIterations}`` for those same
            solvers. Passed through to the solver-log parser so a
            max-iteration stop can be told apart from a generic failure.
            Defaults to empty.
    """

    output_dir: Path
    log: str
    reports: Mapping[str, pd.DataFrame]
    report_paths: Mapping[str, Path]
    ephemerides: Mapping[str, pd.DataFrame]
    ephemeris_paths: Mapping[str, Path]
    contacts: Mapping[str, pd.DataFrame]
    contact_paths: Mapping[str, Path]
    solver_runs: Mapping[str, pd.DataFrame]
    solver_paths: Mapping[str, Path]

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
        solver_paths: Mapping[str, Path] | None = None,
        solver_max_iterations: Mapping[str, int] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.log = log
        self._workspace = None

        rep_paths: dict[str, Path] = dict(report_paths or {})
        eph_paths: dict[str, Path] = dict(ephemeris_paths or {})
        con_paths: dict[str, Path] = dict(contact_paths or {})
        slv_paths: dict[str, Path] = dict(solver_paths or {})

        self.reports = _LazyReports(rep_paths)
        self.report_paths = MappingProxyType(rep_paths)
        self.ephemeris_paths = MappingProxyType(eph_paths)
        self.contact_paths = MappingProxyType(con_paths)
        self.solver_paths = MappingProxyType(slv_paths)
        self.ephemerides = _LazyEphemerides(eph_paths)
        self.contacts = _LazyContacts(con_paths)
        self.solver_runs = _LazySolverRuns(slv_paths, solver_max_iterations or {})

    def __repr__(self) -> str:
        return (
            f"Results(reports={len(self.reports)}, "
            f"ephemerides={len(self.ephemerides)}, "
            f"contacts={len(self.contacts)}, "
            f"solver_runs={len(self.solver_runs)})"
        )

    def _repr_html_(self) -> str:
        return _format_results_html(
            report_names=tuple(self.reports),
            ephemeris_names=tuple(self.ephemerides),
            contact_names=tuple(self.contacts),
            solver_run_names=tuple(self.solver_runs),
        )

    @property
    def converged(self) -> dict[str, bool]:
        """``{solver name: bool}`` — did each solver run reach its goal?

        A convenience view over :attr:`solver_runs` for the common branching
        case (``if not result.converged["DC"]: ...``). Same keys as
        :attr:`solver_runs`; ``{}`` when the mission declared no solvers.

        Reading this materialises every solver run (it inspects each
        DataFrame's ``attrs["converged"]``), so the lazy-parse cost is paid on
        first access — the same trade-off as iterating :attr:`solver_runs`
        values directly.
        """
        return {name: bool(self.solver_runs[name].attrs["converged"]) for name in self.solver_runs}

    def persist(self, path: str | os.PathLike[str]) -> Results:
        """Copy every output artefact under :attr:`output_dir` into ``path``.

        Mutates the :class:`Results` in place so future
        report/ephemeris/contact/solver-run access reads from the persisted
        location instead of the (potentially soon-to-be-cleaned) workspace.
        The :class:`tempfile.TemporaryDirectory` backing a default-workspace
        run is released as part of the call. Run with an explicit
        ``working_dir``: that directory is left intact — ``persist`` is a copy,
        never a move.

        Path mappings are rewritten so any path that lived under the old
        ``output_dir`` now points at the matching file under ``path``.
        Absolute filenames the user pinned outside the workspace via
        ``ReportFile.Filename = "/abs/elsewhere.txt"`` are kept as-is — the
        user chose that destination and we honour it. Already-parsed
        DataFrames stay cached; they are independent of the on-disk files.

        Calling ``persist`` again later copies the artefacts to the new
        destination. A no-op fast path applies when the destination already
        equals the current ``output_dir``.

        Args:
            path: Directory to copy artefacts into. Created if missing.
                ``~`` is expanded and relative paths are resolved against
                the caller's CWD at submit time, so :attr:`output_dir`
                stays absolute after the call.

        Returns:
            ``self``, so the call composes with ``Mission.run().persist(...)``.
        """
        dest = resolve_user_path(path)
        if self.output_dir.exists() and dest == self.output_dir.resolve():
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

        new_reports = {n: _migrate(p) for n, p in self.report_paths.items()}
        new_eph = {n: _migrate(p) for n, p in self.ephemeris_paths.items()}
        new_con = {n: _migrate(p) for n, p in self.contact_paths.items()}
        new_slv = {n: _migrate(p) for n, p in self.solver_paths.items()}

        self.reports._rebase(new_reports)  # type: ignore[attr-defined]
        self.ephemerides._rebase(new_eph)  # type: ignore[attr-defined]
        self.contacts._rebase(new_con)  # type: ignore[attr-defined]
        self.solver_runs._rebase(new_slv)  # type: ignore[attr-defined]
        self.report_paths = MappingProxyType(new_reports)
        self.ephemeris_paths = MappingProxyType(new_eph)
        self.contact_paths = MappingProxyType(new_con)
        self.solver_paths = MappingProxyType(new_slv)

        self.output_dir = dest
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None
        return self

    def write_oem(
        self,
        name: str,
        path: str | os.PathLike[str],
        *,
        originator: str = "gmat-run",
        object_name: str | None = None,
    ) -> Path:
        """Write the ephemeris keyed by ``name`` to ``path`` as a CCSDS-OEM file.

        The DataFrame is materialised through :attr:`ephemerides` (so the
        same lazy-parse and cache contract applies) and emitted via
        :func:`gmat_run.writers.oem.write_oem`. Requires the
        ``[ccsds-ndm]`` extra; raises :class:`ImportError` with an install
        hint if it is missing.

        Args:
            name: Resource name under :attr:`ephemerides`.
            path: Destination ``.oem`` file. Parent directories are created.
            originator: ``ORIGINATOR`` header value. Defaults to
                ``"gmat-run"``.
            object_name: Override for the ``OBJECT_NAME`` meta field. When
                ``None``, falls back to ``df.attrs["object_name"]``.

        Returns:
            The destination ``Path``.

        Raises:
            KeyError: ``name`` is not a known ephemeris resource.
            ImportError: ``ccsds-ndm`` is not installed.
            ValueError: the ephemeris DataFrame is missing required
                metadata for OEM emission. See
                :func:`gmat_run.writers.oem.write_oem` for the full list.
        """
        df = self.ephemerides[name]
        return _write_oem(df, path, originator=originator, object_name=object_name)

    def write_oem_all(
        self,
        dirpath: str | os.PathLike[str],
        *,
        originator: str = "gmat-run",
    ) -> Path:
        """Write every ephemeris in :attr:`ephemerides` to ``dirpath`` as OEM.

        Each file is named ``<name>.oem`` after its resource key. ``dirpath``
        is created if missing. A run with no ephemerides is a no-op (the
        directory is still created).

        Args:
            dirpath: Destination directory.
            originator: ``ORIGINATOR`` header for every emitted file.

        Returns:
            ``dirpath`` as a resolved :class:`pathlib.Path`.
        """
        dest_dir = resolve_user_path(dirpath)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in self.ephemerides:
            self.write_oem(name, dest_dir / f"{name}.oem", originator=originator)
        return dest_dir
