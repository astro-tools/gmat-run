"""Load and override fields on an existing GMAT ``.script``.

The single public entry point is :class:`Mission`. :meth:`Mission.load`
discovers a local GMAT install, bootstraps ``gmatpy``, parses a
``.script`` into the live GMAT object graph, and returns a handle.

Field access uses dotted-path keys against that graph: ``mission["Sat.SMA"]``
returns a typed Python value, ``mission["Sat.SMA"] = 7000`` writes one back
through ``SetField``. The path format is exactly ``Resource.Field`` — one dot,
two non-empty segments. Owned-object pass-through (``Sat.Tanks.MainTank.…``)
is deferred.

``mission.gmat`` exposes the bootstrapped ``gmatpy`` module as an escape
hatch for advanced callers. It is not part of the stable public surface.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from gmat_run.errors import GmatFieldError, GmatLoadError
from gmat_run.install import GmatInstall, locate_gmat
from gmat_run.runtime import bootstrap

__all__ = ["Mission"]


# How many candidate field names to surface in "did you mean" suggestions.
_SUGGESTION_LIMIT: Final = 3


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
        if obj.IsParameterReadOnly(pid):
            raise GmatFieldError(
                f"field '{dotted}' is read-only outside the mission sequence",
                dotted,
                value,
            )
        type_code = obj.GetParameterType(pid)
        coerced = self._coerce(type_code, value, dotted)
        obj.SetField(field, coerced)

    # --- internal helpers -----------------------------------------------------

    def _resolve_resource(self, name: str, dotted: str, *, value: Any) -> Any:
        obj = self._gmat.GetObject(name)
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
            return [[float(matrix.GetElement(i, j)) for j in range(matrix.GetNumColumns())]
                    for i in range(matrix.GetNumRows())]
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
            if not isinstance(value, (list, tuple)) or not value or any(
                not isinstance(row, (list, tuple))
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in row)
                for row in value
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
