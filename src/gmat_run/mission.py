"""Load and override fields on an existing GMAT ``.script``.

The single public entry point is :class:`Mission`. :meth:`Mission.load`
discovers a local GMAT install, bootstraps ``gmatpy``, parses a
``.script`` into the live GMAT object graph, and returns a handle.

Field access uses dotted-path keys against that graph: ``mission["Sat.SMA"]``
returns a typed Python value, ``mission["Sat.SMA"] = 7000`` writes one back
through ``SetField``. The path format is exactly ``Resource.Field`` — one dot,
two non-empty segments. Owned-object pass-through (``Sat.Tanks.MainTank.…``)
is deferred.

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
from collections.abc import Iterator, Mapping
from contextlib import suppress
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Final

import pandas as pd

from gmat_run.errors import GmatFieldError, GmatLoadError, GmatRunError
from gmat_run.install import GmatInstall, locate_gmat
from gmat_run.parsers.aem_ephemeris import parse as _parse_aem_ephemeris
from gmat_run.results import Results
from gmat_run.runtime import bootstrap

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

        Raises:
            GmatNotFoundError: No usable GMAT install was found.
            GmatLoadError: gmatpy could not be loaded, or
                ``LoadScript`` returned ``False`` (parse error — check the
                GMAT log file).
        """
        script_path = Path(path).expanduser()
        install = locate_gmat(gmat_root)
        gmat = bootstrap(install)
        if not gmat.LoadScript(str(script_path)):
            raise GmatLoadError(
                f"GMAT could not parse '{script_path}'; "
                "check the GMAT log file for the underlying error"
            )
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

    def __getitem__(self, dotted: str) -> Any:
        resource, field = _split_path(dotted)
        obj = self._resolve_resource(resource, dotted, value=None)
        pid = self._resolve_field(obj, field, dotted, value=None)
        type_code = obj.GetParameterType(pid)
        return self._read(obj, pid, field, type_code)

    def __setitem__(self, dotted: str, value: Any) -> None:
        resource, field = _split_path(dotted, value=value)
        obj = self._resolve_resource(resource, dotted, value=value)
        pid = self._resolve_field(obj, field, dotted, value=value)
        type_code = obj.GetParameterType(pid)
        coerced = self._coerce(type_code, value, dotted)
        # No preemptive `IsParameterReadOnly` gate: it raises for calculated
        # parameters (e.g. Spacecraft.SMA) and false-positives for delegated
        # fields on PropSetup. Let the engine reject the write itself.
        try:
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
        After :meth:`run` returns, reading ``mission["RF.Filename"]`` yields
        the resolved absolute path, which points at the file on disk.

        Args:
            working_dir: Directory GMAT writes its outputs into. ``None``
                creates a fresh :class:`tempfile.TemporaryDirectory` whose
                lifetime is tied to the returned :class:`Results` — the
                directory survives until the caller drops the result, so lazy
                report parsing keeps working without a context manager. Call
                :meth:`Results.persist` before that to copy the artefacts to a
                permanent location.

        Raises:
            GmatRunError: ``RunScript`` returned a non-success status or
                raised a GMAT engine exception. The captured log is attached
                via ``GmatRunError.log``.
        """
        workspace_path, tempdir = _prepare_workspace(working_dir)
        log_path = workspace_path / "GmatLog.txt"

        # Walk every output subscriber once: bucket the paths and rewrite each
        # relative Filename to an absolute path inside the workspace so GMAT
        # writes where we expect. This sidesteps FileManager.OUTPUT_PATH /
        # GmatGlobal.SetOutputPath, which look like the right knobs but don't
        # actually redirect ReportFile/EphemerisFile output once the script
        # has been parsed: the resolved absolute path is cached on each
        # subscriber, and overriding the Filename field is the only setting
        # the engine consults at write time.
        report_paths, ephemeris_paths, contact_paths = self._rewrite_output_paths(workspace_path)
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
        # Repoint UseLogFile away from the workspace before yielding control.
        # GMAT's MessageInterface holds the log file open for the lifetime of
        # the gmatpy module, so on Windows any later attempt to delete the
        # temp workspace (Results.persist, GC of the TemporaryDirectory) hits
        # WinError 32 on GmatLog.txt. Redirecting to os.devnull closes the
        # workspace handle; the log content has already been captured into
        # ``log`` above. Subsequent GMAT operations log to the null sink
        # until the next mission.run() repoints the handle again — accepted,
        # since gmat-run's public surface ends at Results.
        self._release_log_handle()
        if status != _RUNSCRIPT_OK:
            raise GmatRunError(
                f"GMAT RunScript returned status {status}; expected {_RUNSCRIPT_OK}",
                log=log,
            )

        results = Results(
            output_dir=workspace_path,
            log=log,
            report_paths=report_paths,
            ephemeris_paths=ephemeris_paths,
            contact_paths=contact_paths,
        )
        # See project memory `gmat-run Mission.run temp-dir lifetime ties to
        # Results`: the temp dir must outlive Mission.run so lazy report
        # parsing on `result.reports[name]` still finds the file on disk.
        results._workspace = tempdir
        return results

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
    ) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
        """Bucket subscriber output paths and pin each one to ``workspace_path``.

        Walks every ``ReportFile`` / ``EphemerisFile`` / ``ContactLocator`` in
        the configuration. For each: reads its declared ``Filename``, resolves
        a relative filename against ``workspace_path`` (preserving an absolute
        path as-is), writes the resolved absolute path back to the engine via
        ``SetField("Filename", ...)``, and records the path in the appropriate
        return bucket. Resilient to missing type-enum attributes and broken
        objects — skips quietly rather than aborting the whole run.

        Returns:
            ``(report_paths, ephemeris_paths, contact_paths)`` keyed by
            resource name.
        """
        moderator = self._gmat.Moderator.Instance()
        reports: dict[str, Path] = {}
        ephemerides: dict[str, Path] = {}
        contacts: dict[str, Path] = {}
        bucket = {
            "ReportFile": reports,
            "EphemerisFile": ephemerides,
            "ContactLocator": contacts,
        }
        seen: set[str] = set()
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
                path = Path(declared)
                if path.is_absolute():
                    resolved = path
                else:
                    resolved = workspace_path / path.name
                    with suppress(Exception):
                        obj.SetField("Filename", str(resolved))
                bucket[type_name][name] = resolved
        return reports, ephemerides, contacts

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


# --- module-level helpers -----------------------------------------------------


def _split_path(dotted: str, *, value: Any = None) -> tuple[str, str]:
    if not isinstance(dotted, str) or dotted.count(".") != 1:
        raise GmatFieldError(
            f"invalid dotted path '{dotted}'; expected exactly one dot ('Resource.Field')",
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
    :class:`Results` to extend its lifetime. A user-supplied path is created
    on demand; no tempdir is allocated and ``None`` is returned in its slot.
    """
    if working_dir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="gmat-run-")
        return Path(tempdir.name), tempdir
    path = Path(working_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path, None


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
