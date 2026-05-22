"""Load and override fields on an existing GMAT ``.script``.

The single public entry point is :class:`Mission`. :meth:`Mission.load`
discovers a local GMAT install, bootstraps ``gmatpy``, parses a
``.script`` into the live GMAT object graph, and returns a handle.

Field access uses dotted-path keys against that graph. The grammar covers
three flavours:

- ``mission["Sat.SMA"]`` — top-level ``Resource.Field``.
- ``mission["FM.Drag.CSSISpaceWeatherFile"]`` — sub-resource fields via
  ``Resource.SubResource.Field`` (N≥2 dots after the resource name).
- ``mission["my_var.Value"]`` — script-level ``Create Variable`` blocks,
  addressed via the ``Variable.Value`` suffix.

Reads return typed Python values; writes coerce through the same type
gates and round-trip with reads (``mission[path] = v; assert mission[path] == v``).
Everything after the first dot is passed verbatim to GMAT, which routes
sub-resource paths internally — no Python-side object-graph walk.

:meth:`Mission.run` executes the loaded mission sequence headlessly, captures
GMAT's log into the returned :class:`~gmat_run.results.Results`, and surfaces
engine errors as :class:`~gmat_run.errors.GmatRunError`.

``mission.gmat`` exposes the bootstrapped ``gmatpy`` module as an escape
hatch for advanced callers. It is not part of the stable public surface.
"""

from __future__ import annotations

import difflib
import os
import tempfile
import warnings
from collections.abc import Iterable, Iterator, Mapping
from contextlib import suppress
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Final

import numpy as np
import pandas as pd

from gmat_run._path_utils import resolve_user_path
from gmat_run.errors import GmatFieldError, GmatLoadError, GmatRunError
from gmat_run.install import GmatInstall, locate_gmat
from gmat_run.parsers.aem_ephemeris import parse as _parse_aem_ephemeris
from gmat_run.results import Results
from gmat_run.runtime import bootstrap
from gmat_run.summary import MissionSummary, build_mission_summary

__all__ = ["Mission"]


# How many candidate field names to surface in "did you mean" suggestions.
_SUGGESTION_LIMIT: Final = 3

# GMAT subscriber/event-locator subclasses gmat-run records in Results. Other
# subscribers (OrbitView, GroundTrackPlot, XYPlot) are GUI plotters with no
# file output and are skipped.
_OUTPUT_TYPES: Final = ("ReportFile", "EphemerisFile", "ContactLocator")

# Object-type enum names probed on the gmat module to enumerate output
# resources. ReportFile / EphemerisFile live under SUBSCRIBER; ContactLocator
# is an EventLocator. Both buckets are walked and de-duplicated.
_OUTPUT_TYPE_ENUM_ATTRS: Final = ("SUBSCRIBER", "EVENT_LOCATOR")

# Status code returned by gmat.RunScript() / Moderator.RunScript() on success.
# Negative codes signal initialization or execution failures (see
# .claude/skills/gmat-python/references/commands.md for the table).
_RUNSCRIPT_OK: Final = 1

# Spacecraft.Attitude value that selects the CCSDS-AEM reader path. Per the
# EphemerisFile / Attitude docs in GMAT R2026a, this is the only attitude
# model that consumes an external file via Spacecraft.AttitudeFileName.
_AEM_ATTITUDE_VALUE: Final = "CCSDS-AEM"

# Object-type-enum attribute Mission.attitude_inputs probes to enumerate
# Spacecraft. Defensive ``getattr`` lookup mirrors _OUTPUT_TYPE_ENUM_ATTRS so
# a fake gmat module without this enum simply yields zero attitude inputs
# instead of crashing.
_SPACECRAFT_TYPE_ENUM_ATTR: Final = "SPACECRAFT"

# Object-type-enum attribute used to enumerate Solver resources
# (DifferentialCorrector, Yukon, …) when discovering per-solver ``.data``
# iteration logs. Probed with the same defensive ``getattr`` as the others.
_SOLVER_TYPE_ENUM_ATTR: Final = "SOLVER"

# One engine path field rewritten for the duration of a single run(): the
# resource name, the field name (``"Filename"`` / ``"ReportFile"``), and the
# value the field held before the rewrite. run() rolls each of these back in a
# finally so the engine field reflects the loaded script between runs — see
# Mission._restore_output_paths and issue #115.
_OutputPathRestore = tuple[str, str, str]


class _LazyAttitudeInputs(Mapping[str, pd.DataFrame]):
    """Mapping view over CCSDS-AEM files referenced by Spacecraft resources.

    Mirrors the lazy pattern of :class:`gmat_run.results._LazyEphemerides`:
    parses each file on first access via
    :func:`gmat_run.parsers.aem_ephemeris.parse` and caches the resulting
    DataFrame. Construction is cheap — the underlying paths are not opened.
    """

    def __init__(self, paths: Mapping[str, Path]) -> None:
        self._paths: dict[str, Path] = dict(paths)
        self._cache: dict[str, pd.DataFrame] = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key in self._cache:
            return self._cache[key]
        if key not in self._paths:
            raise KeyError(key)
        frame = _parse_aem_ephemeris(self._paths[key])
        self._cache[key] = frame
        return frame

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, key: object) -> bool:
        return key in self._paths


class Mission:
    """A loaded GMAT ``.script`` with dotted-path field access.

    Construct via :meth:`load`. Reads return Python-typed values; writes
    coerce to the GMAT-expected type and reject the rest with
    :class:`~gmat_run.errors.GmatFieldError`.
    """

    install: GmatInstall
    script_path: Path

    _gmat: ModuleType
    _type_map: dict[int, str]
    _attitude_input_paths: dict[str, Path] | None
    _attitude_inputs: _LazyAttitudeInputs | None

    def __init__(
        self,
        *,
        gmat: ModuleType,
        install: GmatInstall,
        script_path: Path,
    ) -> None:
        self._gmat = gmat
        self.install = install
        self.script_path = script_path
        self._type_map = _build_type_map(gmat)
        # attitude_inputs discovery is lazy — run once on first property access
        # and cached. Re-walking the Spacecraft registry per access would
        # silently mask script edits that happened after load, but the
        # ``Mission`` contract is "view of the loaded script", so a one-shot
        # snapshot at first access matches the rest of the surface.
        self._attitude_input_paths = None
        self._attitude_inputs = None

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        gmat_root: str | os.PathLike[str] | None = None,
    ) -> Mission:
        """Load a GMAT ``.script`` and return a :class:`Mission` handle.

        Discovers GMAT via :func:`~gmat_run.install.locate_gmat` (honouring
        ``gmat_root`` / ``GMAT_ROOT``), bootstraps ``gmatpy`` via
        :func:`~gmat_run.runtime.bootstrap`, and parses the script via
        ``gmat.LoadScript``.

        ``path`` is resolved against the caller's CWD at submit time
        (``~`` is expanded, relative paths become absolute). The resolved
        value is exposed via :attr:`script_path`.

        Raises:
            GmatNotFoundError: No usable GMAT install was found.
            GmatLoadError: gmatpy could not be loaded, or
                ``LoadScript`` returned ``False`` (parse error — check the
                GMAT log file).
        """
        script_path = resolve_user_path(path)
        install = locate_gmat(gmat_root)
        gmat = bootstrap(install)
        if not gmat.LoadScript(str(script_path)):
            raise GmatLoadError(
                f"GMAT could not parse '{script_path}'; "
                "check the GMAT log file for the underlying error"
            )
        # Resolve every loaded Spacecraft against itself so that pre-run field
        # writes (e.g. ``mission["Sat.SMA"] = 7100``) see a fully-coupled state.
        # Without this, GMAT's setter validator runs against an unresolved
        # internal Cartesian state and rejects otherwise-valid Keplerian writes
        # with a bogus "ECC > 1" error. The sandbox-wide ``gmat.Initialize()``
        # would also work but breaks scripts with EventLocator resources
        # (Ex_ContactLocatorAllFormats: a second pre-run init flips
        # ``RunScript`` to a failure status), so target Spacecraft only.
        # ``RunScript`` re-initialises internally, so this only enables the
        # pre-run write path; it does not change run-time semantics.
        api_exception = _get_api_exception(gmat)
        try:
            _initialize_spacecraft(gmat)
        except api_exception as exc:
            raise GmatLoadError(
                f"GMAT raised {type(exc).__name__} initialising '{script_path}': {exc}"
            ) from exc
        return cls(gmat=gmat, install=install, script_path=script_path)

    @property
    def attitude_input_paths(self) -> Mapping[str, Path]:
        """Resolved paths of every CCSDS-AEM file consumed by the loaded script.

        Walks every Spacecraft resource on first access; selects those with
        ``Attitude == "CCSDS-AEM"``; reads ``AttitudeFileName`` and resolves
        relative paths against :attr:`script_path`'s parent directory (the
        same convention GMAT itself uses, since ``LoadScript`` is invoked
        with the script's full path). Absolute paths are kept as-is.

        The returned mapping is keyed by Spacecraft resource name and is a
        read-only view. Discovery is cached for the life of the Mission.

        Returns an empty mapping when no Spacecraft uses CCSDS-AEM, when the
        gmat module does not expose a ``SPACECRAFT`` enum, or when the
        registry walk raises (defensive: an obscure plugin failure should
        not block the rest of the Mission's surface).
        """
        if self._attitude_input_paths is None:
            self._attitude_input_paths = self._discover_attitude_inputs()
        return MappingProxyType(self._attitude_input_paths)

    @property
    def attitude_inputs(self) -> Mapping[str, pd.DataFrame]:
        """Parsed CCSDS-AEM attitude files consumed by the loaded script.

        Lazy: each entry is parsed on first ``__getitem__`` via
        :func:`gmat_run.parsers.aem_ephemeris.parse`, then cached. Iterating
        the mapping or calling ``len`` is free — only access materialises a
        DataFrame.

        Keyed by Spacecraft resource name. Use :attr:`attitude_input_paths`
        to recover the raw path without parsing.
        """
        if self._attitude_inputs is None:
            self._attitude_inputs = _LazyAttitudeInputs(self.attitude_input_paths)
        return self._attitude_inputs

    @property
    def gmat(self) -> ModuleType:
        """The bootstrapped ``gmatpy`` module.

        Escape hatch for callers that need raw SWIG access. Not part of the
        stable public surface — the documented contract is the dotted-path
        ``__getitem__`` / ``__setitem__`` interface.
        """
        return self._gmat

    def summary(self) -> MissionSummary:
        """Return a structured snapshot of this loaded mission.

        See :class:`~gmat_run.summary.MissionSummary` for the schema. Walks
        the gmat object graph each call — there is no cache, so a Mission
        whose fields were edited via ``__setitem__`` and re-summarised
        reflects the latest state. The walk only enumerates resources by name
        and the command graph one level deep; it does not materialise field
        values (use ``mission["Sat.SMA"]`` for that).
        """
        return build_mission_summary(self._gmat, self.script_path)

    def __repr__(self) -> str:
        summary = self.summary()
        return (
            f"Mission({summary.script_name!r}, "
            f"spacecraft={summary.spacecraft_count}, "
            f"commands={summary.command_count})"
        )

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()

    def __getitem__(self, dotted: str) -> Any:
        """Return the value of the GMAT field at ``dotted``.

        ``dotted`` is one of ``"Resource.Field"``,
        ``"Resource.SubResource.Field"`` (e.g. ``"FM.Drag.CSSISpaceWeatherFile"``),
        or ``"<variable>.Value"`` for script-level ``Create Variable`` blocks.
        Everything after the first dot is passed to GMAT as a field name;
        GMAT routes sub-resource access internally.

        Raises:
            GmatFieldError: ``Resource`` does not exist in the loaded script,
                or the (sub-)field name is unknown on the resolved object.
        """
        resource, field = _split_path(dotted)
        obj = self._resolve_resource(resource, dotted, value=None)
        pid = self._resolve_field(obj, field, dotted, value=None)
        type_code = obj.GetParameterType(pid)
        return self._read(obj, pid, field, type_code)

    def __setitem__(self, dotted: str, value: Any) -> None:
        """Write ``value`` to the GMAT field at ``dotted``.

        Grammar matches :meth:`__getitem__`. The value is type-coerced
        against GMAT's parameter type — ``int`` / ``float`` /
        ``numpy.floating`` / ``numpy.integer`` are interchangeable for
        real-typed fields, ``bool`` is required for boolean fields, etc.

        ``Variable.Value`` writes the numeric value of a script-level
        ``Create Variable`` block; ``Resource.SubResource.Field`` writes
        through to a sub-resource (e.g.
        ``mission["FM.Drag.CSSISpaceWeatherFile"] = "/abs/path"``).

        Raises:
            GmatFieldError: ``Resource`` does not exist; the (sub-)field
                name is unknown; the value type does not match the field's
                expected type; or the engine rejected the write.
        """
        resource, field = _split_path(dotted, value=value)
        obj = self._resolve_resource(resource, dotted, value=value)
        pid = self._resolve_field(obj, field, dotted, value=value)
        type_code = obj.GetParameterType(pid)
        coerced = self._coerce(type_code, value, dotted)
        kind = self._type_map.get(type_code, "string")
        # No preemptive `IsParameterReadOnly` gate: it raises for calculated
        # parameters (e.g. Spacecraft.SMA) and false-positives for delegated
        # fields on PropSetup. Let the engine reject the write itself.
        try:
            if kind == "rmatrix":
                # gmatpy's SetField has no matrix overload (only RealArray /
                # StringArray / scalars), so an RMATRIX field is written via
                # SetMatrix with a gmat Rmatrix — symmetric with _read's
                # GetMatrix. A nested list passed to SetField is rejected.
                obj.SetMatrix(field, self._build_rmatrix(coerced))
            else:
                obj.SetField(field, coerced)
        except Exception as exc:
            raise GmatFieldError(
                f"GMAT rejected write to '{dotted}': {exc}",
                dotted,
                value,
            ) from exc

    def run(
        self,
        *,
        working_dir: str | os.PathLike[str] | None = None,
        overwrite: bool = False,
    ) -> Results:
        """Execute the loaded mission sequence and return a :class:`Results`.

        Redirects every relative ``ReportFile``, ``EphemerisFile``, and
        ``ContactLocator`` output and the GMAT log into ``working_dir`` (or an
        isolated temp directory when ``None``), runs ``gmat.RunScript()``, and
        builds a :class:`Results` populated with the resolved output paths and
        the captured log.

        **Filename rewrite.** A relative ``Filename`` on an output subscriber
        is rewritten to an absolute path inside ``working_dir`` before the
        run, so a stock script's outputs land in the per-run workspace
        instead of the GMAT install's default output directory (the charter's
        "no pollution of the user's cwd" rule). Absolute filenames in the
        script are left alone — the user has a specific destination in mind
        and the run honours it. The rewrite is the only mechanism GMAT
        actually consults at write time; ``FileManager.OUTPUT_PATH`` is
        cached per-subscriber at Initialize time and ignored thereafter.
        The rewrite is reverted once the run is over — reading
        ``mission["RF.Filename"]`` afterwards yields the script's declared
        value again, not the workspace path. ``Mission`` stays a view of the
        loaded script (as with :attr:`attitude_inputs`); the resolved output
        locations live on the returned :class:`Results`. Each :meth:`run`
        therefore redirects independently of any earlier run on the same
        ``Mission``.

        **Solver logs.** Every ``Target`` / ``Optimize`` run writes a
        per-``Solver`` ``.data`` iteration log. Its ``<Solver>.ReportFile``
        field is rewritten into ``working_dir`` by the same mechanism, so the
        log is surfaced through :attr:`Results.solver_runs` and shares the
        workspace's lifetime. A ``Solver`` resource that no ``Target`` /
        ``Optimize`` block exercises writes nothing and is simply absent from
        the mapping.

        **Working directory** (when ``working_dir`` is set explicitly):

        * The directory is created if missing; if creation or any write into
          it fails, :class:`~gmat_run.errors.GmatRunError` is raised before
          ``RunScript`` is invoked, with ``GmatRunError.path`` set to the
          offending directory.
        * If ``working_dir`` resolves to the same directory as
          :attr:`script_path`'s parent, a :class:`UserWarning` is emitted —
          GMAT outputs in that case may overwrite the user's source files.
        * Pre-existing files inside ``working_dir`` whose names match the
          run's resolved output paths are detected before ``RunScript``.
          Default policy: raise :class:`~gmat_run.errors.GmatRunError` and
          skip the run, so the prior run's artefacts are not silently mixed
          with the new run's. Pass ``overwrite=True`` to unlink the colliding
          files and proceed. The gate is scoped to files inside
          ``working_dir`` only — absolute filenames pinned outside (see
          below) are the user's destination and are never touched.
        * After a successful run, any output that landed outside
          ``working_dir`` (because the script declared an absolute
          ``Filename``) is summarised as a one-line ``[gmat-run] note: …``
          notice prepended to :attr:`Results.log`.

        Args:
            working_dir: Directory GMAT writes its outputs into. ``~`` is
                expanded and relative paths are resolved against the
                caller's CWD at submit time, so the stored
                :attr:`Results.output_dir` is always absolute regardless of
                what the caller passed in. ``None`` creates a fresh
                :class:`tempfile.TemporaryDirectory` whose lifetime is tied
                to the returned :class:`Results` — the directory survives
                until the caller drops the result, so lazy report parsing
                keeps working without a context manager. Call
                :meth:`Results.persist` before that to copy the artefacts to a
                permanent location.
            overwrite: When ``True``, unlink any pre-existing files in
                ``working_dir`` that collide with the run's resolved output
                paths before invoking ``RunScript``. When ``False`` (the
                default), a collision raises
                :class:`~gmat_run.errors.GmatRunError`. Ignored when
                ``working_dir`` is ``None`` (a fresh temp directory cannot
                contain colliding artefacts).

        Raises:
            GmatRunError: ``RunScript`` returned a non-success status, raised
                a GMAT engine exception, or a pre-run gate failed
                (``working_dir`` not creatable, not writable, or contained
                colliding output files with ``overwrite=False``). The
                captured log is attached via ``GmatRunError.log`` (empty for
                pre-run gate failures); the offending directory or file is
                attached via ``GmatRunError.path``.
        """
        workspace_path, tempdir = _prepare_workspace(working_dir)
        if working_dir is not None:
            self._warn_if_workspace_is_script_dir(workspace_path)

        # Walk every output subscriber once: bucket the paths and rewrite each
        # relative Filename to an absolute path inside the workspace so GMAT
        # writes where we expect. This sidesteps FileManager.OUTPUT_PATH /
        # GmatGlobal.SetOutputPath, which look like the right knobs but don't
        # actually redirect ReportFile/EphemerisFile output once the script
        # has been parsed: the resolved absolute path is cached on each
        # subscriber, and overriding the Filename field is the only setting
        # the engine consults at write time.
        report_paths, ephemeris_paths, contact_paths, sub_restores = self._rewrite_output_paths(
            workspace_path
        )
        # Solver .data logs are redirected the same way. The returned paths are
        # *expected* locations — a Solver no Target/Optimize block exercises
        # writes nothing, so existence is rechecked after the run below.
        try:
            solver_expected, solver_max_iterations, solver_restores = self._discover_solver_outputs(
                workspace_path
            )
        except Exception:
            # _rewrite_output_paths already pinned subscriber Filenames into
            # the workspace; if solver discovery then rejects a path, roll
            # those back before propagating so a refused run leaves the engine
            # reflecting the loaded script (issue #115).
            self._restore_output_paths(sub_restores)
            raise
        all_paths = (
            *report_paths.values(),
            *ephemeris_paths.values(),
            *contact_paths.values(),
            *solver_expected.values(),
        )
        # Both helpers pinned relative engine path fields into workspace_path.
        # Roll them back once the run is over — success or failure — so the
        # next run() on this Mission redirects from the script's declared
        # Filename instead of this run's workspace (issue #115).
        restores = [*sub_restores, *solver_restores]
        try:
            # Pre-run gate: pre-existing artefacts inside working_dir. Bounded
            # to *resolved* output paths that live under workspace_path —
            # absolute filenames pinned by the script outside the workspace are
            # the user's destination and we honour them. Default raises;
            # overwrite=True unlinks the colliding files first.
            _check_inside_workspace_collisions(workspace_path, all_paths, overwrite=overwrite)
            # Create parent directories for nested output paths now the gate
            # has passed — GMAT does not create them itself, and a relative
            # Filename with subdirectories (issue #119) resolves to a nested
            # path that would otherwise fail to write.
            _create_output_dirs(workspace_path, all_paths)
            log_path = workspace_path / "GmatLog.txt"

            self._gmat.UseLogFile(str(log_path))

            api_exception = _get_api_exception(self._gmat)
            try:
                status = int(self._gmat.RunScript())
            except api_exception as exc:
                log = _safe_read(log_path)
                self._release_log_handle()
                raise GmatRunError(
                    f"GMAT raised {type(exc).__name__} during RunScript: {exc}",
                    log=log,
                ) from exc

            log = _safe_read(log_path)
            # Repoint UseLogFile away from the workspace before yielding
            # control. GMAT's MessageInterface holds the log file open for the
            # lifetime of the gmatpy module, so on Windows any later attempt to
            # delete the temp workspace (Results.persist, GC of the
            # TemporaryDirectory) hits WinError 32 on GmatLog.txt. Redirecting
            # to os.devnull closes the workspace handle; the log content has
            # already been captured into ``log`` above. Subsequent GMAT
            # operations log to the null sink until the next mission.run()
            # repoints the handle again — accepted, since gmat-run's public
            # surface ends at Results.
            self._release_log_handle()
            if status != _RUNSCRIPT_OK:
                raise GmatRunError(
                    f"GMAT RunScript returned status {status}; expected {_RUNSCRIPT_OK}",
                    log=log,
                )

            # Out-of-workspace notice: any output the script pinned at an
            # absolute path outside workspace_path gets a one-line summary at
            # the top of the captured log so callers see the trail without
            # having to re-walk the path mappings themselves.
            notice = _format_outside_workspace_notice(workspace_path, all_paths)
            if notice:
                log = notice + log

            # Keep only the solver logs GMAT actually wrote — a declared-but-
            # unused Solver leaves the no-solver mapping empty rather than
            # raising.
            solver_paths = {n: p for n, p in solver_expected.items() if p.exists()}
            results = Results(
                output_dir=workspace_path,
                log=log,
                report_paths=report_paths,
                ephemeris_paths=ephemeris_paths,
                contact_paths=contact_paths,
                solver_paths=solver_paths,
                solver_max_iterations={
                    n: solver_max_iterations[n] for n in solver_paths if n in solver_max_iterations
                },
            )
            # See project memory `gmat-run Mission.run temp-dir lifetime ties
            # to Results`: the temp dir must outlive Mission.run so lazy report
            # parsing on `result.reports[name]` still finds the file on disk.
            results._workspace = tempdir
            return results
        finally:
            # Engine path fields are pinned only for the duration of the run.
            # Restoring them here — on the success return, on a GmatRunError,
            # and on the pre-run collision gate — keeps the subscriber and
            # solver fields reflecting the loaded script, so the next run()
            # resolves its outputs afresh. Resolved paths live on Results.
            self._restore_output_paths(restores)

    def _warn_if_workspace_is_script_dir(self, workspace_path: Path) -> None:
        """Emit a UserWarning if ``workspace_path`` resolves to the script's directory.

        Running into the script's own directory risks GMAT overwriting the
        ``.script`` itself (or sibling source files) when the script's
        ``ReportFile.Filename`` happens to collide with a source name.
        Resolution is required on both sides — relative ``working_dir`` and a
        symlinked install both want to compare canonical paths. Failure to
        resolve (e.g. permission denied on a parent symlink) silently skips
        the warning; the writability gate would have already raised.
        """
        try:
            workspace_resolved = workspace_path.resolve()
            script_dir_resolved = self.script_path.parent.resolve()
        except OSError:
            return
        if workspace_resolved != script_dir_resolved:
            return
        warnings.warn(
            f"working_dir '{workspace_path}' is the script's own directory; "
            "outputs may overwrite the script or other source files",
            UserWarning,
            stacklevel=3,
        )

    # --- discovery helpers ----------------------------------------------------

    def _discover_attitude_inputs(self) -> dict[str, Path]:
        """Walk Spacecraft resources and bucket every CCSDS-AEM input.

        For each Spacecraft whose ``Attitude`` field equals ``"CCSDS-AEM"``,
        read ``AttitudeFileName`` and resolve relative paths against
        :attr:`script_path`'s parent directory. Spacecraft using any other
        attitude model are ignored. Spacecraft that fail to expose either
        field are skipped silently — the rest of the script may still be
        usable, and a missing AttitudeFileName is something the user can
        diagnose by inspecting the script directly.
        """
        type_id = getattr(self._gmat, _SPACECRAFT_TYPE_ENUM_ATTR, None)
        if type_id is None:
            return {}
        moderator = self._gmat.Moderator.Instance()
        try:
            names = list(moderator.GetListOfObjects(type_id))
        except Exception:
            return {}
        script_dir = self.script_path.parent
        inputs: dict[str, Path] = {}
        for name in names:
            obj = self._gmat.GetObject(name)
            if obj is None:
                continue
            try:
                attitude = str(obj.GetField("Attitude"))
            except Exception:
                continue
            if attitude != _AEM_ATTITUDE_VALUE:
                continue
            try:
                filename = str(obj.GetField("AttitudeFileName"))
            except Exception:
                continue
            if not filename:
                continue
            resolved = Path(filename)
            if not resolved.is_absolute():
                resolved = (script_dir / resolved).resolve()
            inputs[name] = resolved
        return inputs

    # --- run helpers ----------------------------------------------------------

    def _release_log_handle(self) -> None:
        """Close GMAT's hold on the previous log path by repointing it.

        ``UseLogFile(os.devnull)`` is the cheapest way to make GMAT's
        MessageInterface drop the file handle on the previous path. The
        ``suppress`` exists only as defence in depth — failure here would
        propagate as an unrelated error during a successful run, masking the
        Results the caller wanted.
        """
        with suppress(Exception):
            self._gmat.UseLogFile(os.devnull)

    def _rewrite_output_paths(
        self, workspace_path: Path
    ) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path], list[_OutputPathRestore]]:
        """Bucket subscriber output paths and pin each one to ``workspace_path``.

        Walks every ``ReportFile`` / ``EphemerisFile`` / ``ContactLocator`` in
        the configuration. For each: reads its declared ``Filename`` and
        resolves it against ``workspace_path`` via :func:`_resolve_output_path`
        — a relative filename keeps its subdirectory structure under the
        workspace (issue #119), an absolute path is left as-is — then writes
        the resolved path back to the engine via ``SetField("Filename", ...)``
        and records it in the appropriate return bucket. Resilient to missing
        type-enum attributes and broken objects — skips quietly rather than
        aborting the whole run.

        Done in two phases: phase one discovers and resolves every output
        path, phase two writes them back. The split means a relative
        ``Filename`` that would escape the workspace (a ``..`` component)
        raises in phase one — before any ``SetField`` — so a refused run
        leaves every engine field untouched.

        Every relative ``Filename`` that gets rewritten also yields a restore
        entry — ``(resource_name, "Filename", declared_value)`` — so
        :meth:`run` can return the engine field to its script-declared state
        once the run is over (see :meth:`_restore_output_paths`). Without that
        rollback the rewritten absolute path would survive into the next
        ``run()``, where the ``is_absolute()`` check would mistake it for a
        user-pinned destination and skip the redirect (issue #115). An
        absolute ``Filename`` is left untouched and produces no restore entry.

        Returns:
            ``(report_paths, ephemeris_paths, contact_paths, restores)``. The
            three path maps are keyed by resource name; ``restores`` is the
            list of rewritten fields :meth:`run` must roll back afterwards.
        """
        moderator = self._gmat.Moderator.Instance()
        reports: dict[str, Path] = {}
        ephemerides: dict[str, Path] = {}
        contacts: dict[str, Path] = {}
        restores: list[_OutputPathRestore] = []
        bucket = {
            "ReportFile": reports,
            "EphemerisFile": ephemerides,
            "ContactLocator": contacts,
        }
        seen: set[str] = set()
        # Phase one — discover every output subscriber and resolve its declared
        # Filename. _resolve_output_path raises here, before any field is
        # mutated, if a relative path would escape the workspace.
        pending: list[tuple[str, Any, str, str, Path]] = []
        for enum_attr in _OUTPUT_TYPE_ENUM_ATTRS:
            type_id = getattr(self._gmat, enum_attr, None)
            if type_id is None:
                continue
            try:
                names = list(moderator.GetListOfObjects(type_id))
            except Exception:
                # Fall through — the other bucket may still resolve.
                continue
            for name in names:
                if name in seen:
                    continue
                seen.add(name)
                obj = self._gmat.GetObject(name)
                if obj is None:
                    continue
                try:
                    type_name = obj.GetTypeName()
                except Exception:
                    continue
                if type_name not in _OUTPUT_TYPES:
                    continue
                try:
                    declared = str(obj.GetField("Filename"))
                except Exception:
                    continue
                resolved = _resolve_output_path(name, declared, workspace_path)
                pending.append((name, obj, type_name, declared, resolved))
        # Phase two — pin each relative Filename into the workspace. Deferred
        # to its own pass so a workspace-escape error above aborts the run
        # before any engine field has been touched.
        for name, obj, type_name, declared, resolved in pending:
            if not Path(declared).is_absolute():
                with suppress(Exception):
                    obj.SetField("Filename", str(resolved))
                    # Only recorded once SetField has actually landed —
                    # a suppressed failure leaves nothing to restore.
                    restores.append((name, "Filename", declared))
            bucket[type_name][name] = resolved
        return reports, ephemerides, contacts, restores

    def _discover_solver_outputs(
        self, workspace_path: Path
    ) -> tuple[dict[str, Path], dict[str, int], list[_OutputPathRestore]]:
        """Enumerate ``Solver`` resources and pin each ``.data`` log to the workspace.

        Walks every ``Solver`` resource via ``Moderator.GetListOfObjects``. For
        each: derives the expected ``.data`` path — the ``ReportFile`` field
        when set, otherwise the default ``<TypeName><SolverName>.data`` — and,
        for a relative or unset value, rewrites ``ReportFile`` to an absolute
        path inside ``workspace_path`` via ``SetField`` so GMAT writes where we
        expect. Resolution goes through :func:`_resolve_output_path`, so a
        relative value keeps its subdirectory structure under the workspace
        (issue #119) and an absolute ``ReportFile`` — the user's chosen
        destination — is left alone, mirroring :meth:`_rewrite_output_paths`.
        The solver's ``MaximumIterations`` is read alongside — the ``.data``
        file does not record it, but the parser needs it to classify a
        max-iteration stop.

        Resilient to a missing ``SOLVER`` enum and to broken objects — skips
        quietly rather than aborting the run. Done in two phases like
        :meth:`_rewrite_output_paths`: a relative value that would escape the
        workspace (a ``..`` component) raises during phase one, before any
        ``SetField``.

        Each rewritten ``ReportFile`` yields a restore entry, exactly as in
        :meth:`_rewrite_output_paths`, so :meth:`run` can roll the engine
        field back to its declared value once the run is over (issue #115).

        Returns:
            ``(paths, max_iterations, restores)`` keyed by solver resource
            name (``restores`` is a flat list). The paths are *expected*
            locations; :meth:`run` rechecks existence after the run, since an
            unexercised ``Solver`` writes nothing.
        """
        type_id = getattr(self._gmat, _SOLVER_TYPE_ENUM_ATTR, None)
        if type_id is None:
            return {}, {}, []
        moderator = self._gmat.Moderator.Instance()
        try:
            names = list(moderator.GetListOfObjects(type_id))
        except Exception:
            return {}, {}, []
        paths: dict[str, Path] = {}
        max_iterations: dict[str, int] = {}
        restores: list[_OutputPathRestore] = []
        # Phase one — resolve every solver .data path. _resolve_output_path
        # raises here, before any field is mutated, if a relative ReportFile
        # would escape the workspace.
        pending: list[tuple[str, Any, str, str, Path]] = []
        for name in names:
            obj = self._gmat.GetObject(name)
            if obj is None:
                continue
            try:
                type_name = obj.GetTypeName()
            except Exception:
                continue
            declared = ""
            with suppress(Exception):
                declared = str(obj.GetField("ReportFile"))
            effective = declared if declared else f"{type_name}{name}.data"
            resolved = _resolve_output_path(name, effective, workspace_path)
            pending.append((name, obj, declared, effective, resolved))
            with suppress(Exception):
                max_iterations[name] = int(float(str(obj.GetField("MaximumIterations"))))
        # Phase two — pin each relative ReportFile into the workspace, deferred
        # so a phase-one escape error aborts before any field is mutated.
        for name, obj, declared, effective, resolved in pending:
            paths[name] = resolved
            if not Path(effective).is_absolute():
                with suppress(Exception):
                    obj.SetField("ReportFile", str(resolved))
                    # Recorded only after SetField lands — see the matching
                    # note in _rewrite_output_paths.
                    restores.append((name, "ReportFile", declared))
        return paths, max_iterations, restores

    def _restore_output_paths(self, restores: Iterable[_OutputPathRestore]) -> None:
        """Roll back the engine path fields rewritten for one :meth:`run` call.

        Each entry pairs a resource name with the field it owns and the value
        that field held before :meth:`_rewrite_output_paths` /
        :meth:`_discover_solver_outputs` pinned it into the run's workspace.
        Restoring them leaves every subscriber and solver reflecting the
        *loaded script* once the run is over, so a second :meth:`run` on the
        same :class:`Mission` redirects from the originally declared
        ``Filename`` rather than the previous run's workspace (issue #115).

        The resource is re-fetched by name rather than carried as a live SWIG
        proxy — gmatpy hands back a fresh proxy per call, so the name is the
        stable handle. Each ``SetField`` is best-effort: a failure to roll
        back must not mask the run's real outcome, so it is suppressed.
        """
        for name, field, original in restores:
            with suppress(Exception):
                obj = self._gmat.GetObject(name)
                if obj is not None:
                    obj.SetField(field, original)

    # --- internal helpers -----------------------------------------------------

    def _resolve_resource(self, name: str, dotted: str, *, value: Any) -> Any:
        try:
            obj = self._gmat.GetObject(name)
        except AttributeError as exc:
            # gmatpy's GetObject raises AttributeError from inside the SWIG
            # wrapper when the name does not resolve (it calls
            # `val.GetTypeName()` on a NULL pointer).
            raise GmatFieldError(
                f"unknown resource '{name}' (no object by that name in the loaded script)",
                dotted,
                value,
            ) from exc
        if obj is None:
            raise GmatFieldError(
                f"unknown resource '{name}' (no object by that name in the loaded script)",
                dotted,
                value,
            )
        return obj

    def _resolve_field(self, obj: Any, field: str, dotted: str, *, value: Any) -> int:
        try:
            return int(obj.GetParameterID(field))
        except Exception as exc:
            type_name = _safe_type_name(obj)
            suggestions = _suggest_fields(obj, field)
            hint = f"; did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise GmatFieldError(
                f"unknown field '{field}' on {type_name}{hint}",
                dotted,
                value,
            ) from exc

    def _read(self, obj: Any, pid: int, field: str, type_code: int) -> Any:
        kind = self._type_map.get(type_code, "string")
        if kind == "real":
            return float(obj.GetNumber(field))
        if kind == "integer":
            try:
                return int(obj.GetIntegerParameter(pid))
            except Exception:
                # Some plugins implement only the string-backed accessor.
                return int(obj.GetField(field))
        if kind == "boolean":
            try:
                return bool(obj.GetBooleanParameter(pid))
            except Exception:
                raw = obj.GetField(field)
                return _parse_bool(raw)
        if kind == "string_array":
            return list(obj.GetStringArrayParameter(pid))
        if kind == "rvector":
            return [float(x) for x in obj.GetVector(field)]
        if kind == "rmatrix":
            matrix = obj.GetMatrix(field)
            return [
                [float(matrix.GetElement(i, j)) for j in range(matrix.GetNumColumns())]
                for i in range(matrix.GetNumRows())
            ]
        # STRING_TYPE, FILENAME_TYPE, OBJECT_TYPE, ENUMERATION_TYPE, and
        # anything we did not classify — fall back to the string form.
        return str(obj.GetField(field))

    def _coerce(self, type_code: int, value: Any, dotted: str) -> Any:
        # Strip numpy first so the type checks below see native Python types.
        # Notebook users routinely pass np.float64 / np.int64 / np.bool_ /
        # ndarray without thinking — the alternative is a confusing rejection
        # for what looks like a valid number/array.
        value = _strip_numpy(value)
        kind = self._type_map.get(type_code, "string")
        if kind == "real":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise self._type_mismatch(dotted, value, "a real number")
            return float(value)
        if kind == "integer":
            if isinstance(value, bool):
                raise self._type_mismatch(dotted, value, "an integer")
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise self._type_mismatch(dotted, value, "an integer")
        if kind == "boolean":
            if not isinstance(value, bool):
                raise self._type_mismatch(dotted, value, "a bool")
            return value
        if kind == "string_array":
            if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) for x in value):
                raise self._type_mismatch(dotted, value, "a list of strings")
            return list(value)
        if kind == "rvector":
            if not isinstance(value, (list, tuple)) or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) for x in value
            ):
                raise self._type_mismatch(dotted, value, "a list of numbers")
            return [float(x) for x in value]
        if kind == "rmatrix":
            if (
                not isinstance(value, (list, tuple))
                or not value
                or any(
                    not isinstance(row, (list, tuple))
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in row)
                    for row in value
                )
            ):
                raise self._type_mismatch(dotted, value, "a list of lists of numbers")
            return [[float(x) for x in row] for row in value]
        # string-like (STRING / FILENAME / OBJECT / ENUMERATION / unknown)
        if not isinstance(value, str):
            raise self._type_mismatch(dotted, value, "a string")
        return value

    @staticmethod
    def _type_mismatch(dotted: str, value: Any, expected: str) -> GmatFieldError:
        return GmatFieldError(
            f"type mismatch for '{dotted}': expected {expected}, got {type(value).__name__}",
            dotted,
            value,
        )

    def _build_rmatrix(self, rows: list[list[float]]) -> Any:
        """Build a gmat ``Rmatrix`` from a coerced nested list of floats.

        gmatpy's ``SetField`` has no matrix overload, so an RMATRIX field is
        written through ``SetMatrix`` with a real ``Rmatrix``. ``_coerce`` has
        already validated ``rows`` as a non-empty list of numeric rows.
        """
        n_rows = len(rows)
        n_cols = len(rows[0])
        matrix = self._gmat.Rmatrix(n_rows, n_cols)
        for i, row in enumerate(rows):
            for j, element in enumerate(row):
                matrix.SetElement(i, j, element)
        return matrix


# --- module-level helpers -----------------------------------------------------


def _strip_numpy(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays into native Python types.

    Handles three shapes: ``numpy.bool_`` and other numpy scalar dtypes (via
    ``.item()``), ``numpy.ndarray`` (via ``.tolist()``, which recursively
    yields native types), and Python lists/tuples that may contain numpy
    elements (walk and convert in place). Anything else is returned
    untouched so the rest of ``_coerce``'s type checks fire as written.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_strip_numpy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_numpy(v) for v in value)
    return value


def _split_path(dotted: str, *, value: Any = None) -> tuple[str, str]:
    # Grammar: ``Resource.Field`` (single-dot), ``Resource.SubResource.Field``
    # (multi-dot, e.g. ``FM.Drag.CSSISpaceWeatherFile``), and the special-case
    # ``Variable.Value`` for script-level ``Create Variable`` blocks.
    # Everything after the first dot is passed verbatim to GMAT, which routes
    # dotted sub-resource paths internally via ``GetParameterID`` / ``GetField``
    # / ``SetField`` on the top-level Resource — no Python-side walk needed.
    if not isinstance(dotted, str) or "." not in dotted:
        raise GmatFieldError(
            f"invalid dotted path '{dotted}'; expected at least one dot "
            "('Resource.Field' or 'Resource.SubResource.Field')",
            str(dotted),
            value,
        )
    resource, field = dotted.split(".", 1)
    if not resource or not field:
        raise GmatFieldError(
            f"invalid dotted path '{dotted}'; both segments must be non-empty",
            dotted,
            value,
        )
    return resource, field


def _build_type_map(gmat: ModuleType) -> dict[int, str]:
    """Map GMAT type-code integers to internal kind tags.

    Resolved once per :class:`Mission` so unit tests can stand up a fake
    ``gmat`` module with whatever integer values they like — only the
    relative mapping matters.
    """
    pairs: list[tuple[str, str]] = [
        ("REAL_TYPE", "real"),
        ("INTEGER_TYPE", "integer"),
        ("UNSIGNED_INT_TYPE", "integer"),
        ("BOOLEAN_TYPE", "boolean"),
        ("STRING_TYPE", "string"),
        ("FILENAME_TYPE", "string"),
        ("OBJECT_TYPE", "string"),
        ("ENUMERATION_TYPE", "string"),
        ("STRINGARRAY_TYPE", "string_array"),
        ("OBJECTARRAY_TYPE", "string_array"),
        ("RVECTOR_TYPE", "rvector"),
        ("RMATRIX_TYPE", "rmatrix"),
    ]
    mapping: dict[int, str] = {}
    for attr, kind in pairs:
        code = getattr(gmat, attr, None)
        if isinstance(code, int):
            mapping[code] = kind
    return mapping


def _suggest_fields(obj: Any, field: str) -> list[str]:
    try:
        count = int(obj.GetParameterCount())
        names = [str(obj.GetParameterText(i)) for i in range(count)]
    except Exception:
        return []
    return difflib.get_close_matches(field, names, n=_SUGGESTION_LIMIT)


def _safe_type_name(obj: Any) -> str:
    try:
        return f"{obj.GetTypeName()} '{obj.GetName()}'"
    except Exception:
        return "object"


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "on", "1"}
    return bool(raw)


def _prepare_workspace(
    working_dir: str | os.PathLike[str] | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Resolve the run's output directory and (optional) tempdir owner.

    When ``working_dir`` is None we mint a fresh :class:`TemporaryDirectory`
    and return its handle so the caller can park it on the resulting
    :class:`Results` to extend its lifetime — writability is implicit and
    the directory cannot collide with the script's own location, so neither
    gate runs.

    A user-supplied path is created on demand and probed for writability with
    a unique :func:`tempfile.mkstemp` file inside it; either failure raises
    :class:`~gmat_run.errors.GmatRunError` before ``RunScript`` is invoked,
    with ``GmatRunError.path`` set to the offending directory.
    """
    if working_dir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="gmat-run-")
        return Path(tempdir.name), tempdir
    path = resolve_user_path(working_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GmatRunError(
            f"working_dir '{path}' could not be created: {exc}",
            log="",
            path=path,
        ) from exc
    _probe_writable(path)
    return path, None


def _probe_writable(path: Path) -> None:
    """Confirm ``path`` accepts writes by creating and unlinking a probe file.

    ``os.access`` is unreliable on Windows ACLs and on filesystems where the
    directory mode-bits don't capture the effective permission, so the only
    portable answer is to actually try writing. ``tempfile.mkstemp`` is used
    instead of a fixed probe-file name so the check is collision-free even if
    the user happens to keep an unrelated dotfile of the same name.
    """
    try:
        fd, probe = tempfile.mkstemp(prefix=".gmat-run-write-probe-", dir=str(path))
    except OSError as exc:
        raise GmatRunError(
            f"working_dir '{path}' is not writable: {exc}",
            log="",
            path=path,
        ) from exc
    os.close(fd)
    with suppress(OSError):
        os.unlink(probe)


def _resolve_output_path(name: str, declared: str, workspace_path: Path) -> Path:
    """Resolve a subscriber / solver output path against ``workspace_path``.

    An absolute ``declared`` path is the user's chosen destination and is
    returned unchanged. A relative path is pinned under ``workspace_path``
    with its subdirectory structure preserved, so two outputs declared with
    distinct relative paths that share a basename — ``runs/a/out.txt`` and
    ``runs/b/out.txt`` — resolve to distinct files instead of both collapsing
    onto ``workspace/out.txt`` (issue #119).

    A relative path containing a ``..`` component is rejected with
    :class:`~gmat_run.errors.GmatRunError`: it could resolve outside
    ``workspace_path``, and gmat-run will not write beyond the workspace it
    manages. Callers resolve every output path before mutating any engine
    field, so a raised error leaves the run un-started and the workspace
    untouched.
    """
    path = Path(declared)
    if path.is_absolute():
        return path
    if ".." in path.parts:
        raise GmatRunError(
            f"resource '{name}' declares a relative output path '{declared}' "
            f"with a '..' component that could escape working_dir "
            f"'{workspace_path}'; use a path inside the workspace or an "
            f"absolute path",
            log="",
            path=workspace_path,
        )
    return workspace_path / path


def _create_output_dirs(workspace_path: Path, paths: Iterable[Path]) -> None:
    """Pre-create parent directories for output paths inside the workspace.

    Preserving a relative ``Filename``'s subdirectory structure under
    ``workspace_path`` (issue #119) means a ``ReportFile`` declared as
    ``runs/a/out.txt`` resolves to a nested path — but GMAT does not create
    output directories itself, so the write would fail unless ``runs/a/``
    exists first. This walks the resolved output set and creates every parent
    that lives under the workspace. Absolute paths the script pinned outside
    the workspace are the user's destination and are left alone.

    Runs after the collision gate so a refused run still leaves the workspace
    untouched. A path directly in the workspace has the (already-created)
    workspace as its parent, so ``mkdir`` is a no-op there.
    """
    try:
        workspace_resolved = workspace_path.resolve()
    except OSError:
        return
    for p in paths:
        if not _is_inside(p, workspace_resolved):
            continue
        with suppress(OSError):
            p.parent.mkdir(parents=True, exist_ok=True)


def _check_inside_workspace_collisions(
    workspace_path: Path,
    paths: Iterable[Path],
    *,
    overwrite: bool,
) -> None:
    """Gate two output-path collision classes inside the workspace.

    Walks every resolved output path; only those that live under
    ``workspace_path`` (i.e. relative ``Filename`` rewrites, not user-pinned
    absolute paths) participate. Two classes are caught:

    * **Intra-run** — two outputs of *this* run resolved to the same path. It
      raises regardless of ``overwrite``: ``overwrite`` governs pre-existing
      files, not two writers racing onto one path within a single run, and
      letting the run proceed would silently clobber one output.
    * **Pre-existing** — a file already on disk at a resolved output path.
      With ``overwrite=False`` (the default) this raises
      :class:`~gmat_run.errors.GmatRunError` listing the collisions; with
      ``overwrite=True`` the files are unlinked instead.

    The check runs before ``RunScript`` so a refused run leaves the workspace
    untouched (the error message tells the caller exactly what to clear, or
    pass ``overwrite=True``). ``GmatLog.txt`` itself is not script-declared
    and never participates in this gate.
    """
    try:
        workspace_resolved = workspace_path.resolve()
    except OSError:
        # An unresolvable workspace means we cannot tell what is inside; skip
        # the gate rather than raise a confusing collision error. Writability
        # would have already failed if this path were genuinely broken.
        return
    inside = [p for p in paths if _is_inside(p, workspace_resolved)]
    # Intra-run collision: two resolved output paths are identical. With
    # relative subdirectory structure preserved (issue #119) this only
    # happens when two resources declare the same relative Filename — a
    # collision the script itself carries, which gmat-run surfaces rather
    # than letting the second writer silently clobber the first.
    seen: set[Path] = set()
    duplicates: list[Path] = []
    for p in inside:
        if p in seen and p not in duplicates:
            duplicates.append(p)
        seen.add(p)
    if duplicates:
        listing = ", ".join(str(p) for p in duplicates)
        raise GmatRunError(
            (
                f"working_dir '{workspace_path}': multiple outputs resolve to "
                f"the same path: {listing}; give the colliding subscribers "
                f"distinct Filename values"
            ),
            log="",
            path=workspace_path,
        )
    collisions = [p for p in inside if p.exists()]
    if not collisions:
        return
    if overwrite:
        for p in collisions:
            with suppress(OSError):
                p.unlink()
        return
    listing = ", ".join(str(p) for p in collisions)
    raise GmatRunError(
        (
            f"working_dir '{workspace_path}' already contains output files: "
            f"{listing}; pass overwrite=True to clear them and re-run"
        ),
        log="",
        path=workspace_path,
    )


def _format_outside_workspace_notice(workspace_path: Path, paths: Iterable[Path]) -> str:
    """Build the one-line "out-of-workspace" notice for :attr:`Results.log`.

    Returns ``""`` when every output landed inside ``workspace_path``;
    otherwise a single line of the form ``[gmat-run] note: N output(s)
    landed outside working_dir: <comma-separated paths>\\n`` so the notice
    is visible at the top of the captured log without disturbing the GMAT
    text below it.
    """
    try:
        workspace_resolved = workspace_path.resolve()
    except OSError:
        return ""
    outside = [p for p in paths if not _is_inside(p, workspace_resolved)]
    if not outside:
        return ""
    listing = ", ".join(str(p) for p in outside)
    return f"[gmat-run] note: {len(outside)} output(s) landed outside working_dir: {listing}\n"


def _is_inside(path: Path, workspace_resolved: Path) -> bool:
    """Return True if ``path`` resolves to a location under ``workspace_resolved``.

    ``path`` may not exist yet (collision check fires before the run); that
    is fine — :meth:`Path.resolve` returns the would-be absolute path against
    the existing parents. ``OSError`` from a broken symlink along the way
    falls through to ``False`` so the path is treated as outside (the
    collision gate skips it; the out-of-workspace notice flags it).
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        return False
    return True


def _safe_read(path: Path) -> str:
    """Read ``path`` as UTF-8, returning ``""`` on any I/O failure.

    The log file may be missing entirely if ``UseLogFile`` was rejected, or
    truncated if the engine crashed mid-write. Either way we want to surface
    *something* on the resulting :class:`~gmat_run.errors.GmatRunError` rather
    than tripping over the read.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _initialize_spacecraft(gmat: ModuleType) -> None:
    """Call ``Initialize`` on every Spacecraft loaded into the sandbox.

    Walks the Spacecraft registry via ``Moderator.GetListOfObjects`` keyed by
    the ``SPACECRAFT`` enum and invokes ``Initialize`` on each handle that
    exposes one. Missing enums, missing moderator, or objects without
    ``Initialize`` are tolerated silently — fakes vary, and the only callers
    that need this method to succeed are the Keplerian-field-write tests on
    real GMAT, which always have all three.
    """
    type_id = getattr(gmat, _SPACECRAFT_TYPE_ENUM_ATTR, None)
    if type_id is None:
        return
    moderator_proxy = getattr(gmat, "Moderator", None)
    if moderator_proxy is None:
        return
    moderator = moderator_proxy.Instance()
    try:
        names = list(moderator.GetListOfObjects(type_id))
    except Exception:
        return
    for name in names:
        obj = gmat.GetObject(name)
        init = getattr(obj, "Initialize", None) if obj is not None else None
        if init is None:
            continue
        init()


def _get_api_exception(gmat: ModuleType) -> type[BaseException]:
    """Return the gmat engine's exception type, falling back to ``Exception``.

    Real gmatpy exposes ``APIException``; test fakes don't always bother. The
    fallback keeps the ``except`` clause well-formed without burdening every
    fixture with a stub class.
    """
    exc = getattr(gmat, "APIException", None)
    if isinstance(exc, type) and issubclass(exc, BaseException):
        return exc
    return Exception
