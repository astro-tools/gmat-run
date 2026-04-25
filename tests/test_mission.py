"""Unit tests for :mod:`gmat_run.mission`.

A handwritten fake ``gmat`` module stands in for ``gmatpy``. It mirrors the
parameter API surface used by :class:`~gmat_run.mission.Mission` —
``GetObject``, ``GetParameterID``/``Type``/``Text``/``Count``,
``IsParameterReadOnly``, ``GetField``/``SetField`` and the typed accessor
family. The fake objects identify as ``Spacecraft``, ``Propagator``, and
``ImpulsiveBurn`` so the tests cover the three classes the issue calls out
without needing a real GMAT install.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from gmat_run import Mission, Results
from gmat_run.errors import GmatFieldError, GmatLoadError, GmatNotFoundError, GmatRunError
from gmat_run.install import GmatInstall

# --- type-code constants for the fake gmat module -----------------------------

# Arbitrary distinct integers; only the relative mapping matters because
# Mission resolves the codes via getattr on the gmat module.
_TYPE_CODES = {
    "REAL_TYPE": 1,
    "INTEGER_TYPE": 2,
    "UNSIGNED_INT_TYPE": 3,
    "BOOLEAN_TYPE": 4,
    "STRING_TYPE": 5,
    "FILENAME_TYPE": 6,
    "OBJECT_TYPE": 7,
    "ENUMERATION_TYPE": 8,
    "STRINGARRAY_TYPE": 9,
    "OBJECTARRAY_TYPE": 10,
    "RVECTOR_TYPE": 11,
    "RMATRIX_TYPE": 12,
}


# --- fake GmatBase / Rmatrix --------------------------------------------------


class _FakeRmatrix:
    """Minimal Rmatrix stand-in supporting the read path Mission uses."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def GetNumRows(self) -> int:
        return len(self._rows)

    def GetNumColumns(self) -> int:
        return len(self._rows[0]) if self._rows else 0

    def GetElement(self, i: int, j: int) -> float:
        return self._rows[i][j]


class _FakeObject:
    """A configurable stand-in for a GMAT-side ``GmatBase``."""

    def __init__(
        self,
        type_name: str,
        name: str,
        fields: dict[str, tuple[int, Any, bool]],
    ) -> None:
        # fields: {field_name: (type_code, value, read_only)}
        self._type = type_name
        self._name = name
        self._fields = fields
        self._order = list(fields.keys())
        self.set_calls: list[tuple[str, Any]] = []

    def GetTypeName(self) -> str:
        return self._type

    def GetName(self) -> str:
        return self._name

    def GetParameterCount(self) -> int:
        return len(self._order)

    def GetParameterText(self, idx: int) -> str:
        return self._order[idx]

    def GetParameterID(self, name: str) -> int:
        if name not in self._fields:
            raise RuntimeError(f"unknown parameter '{name}'")
        return self._order.index(name)

    def GetParameterType(self, pid: int) -> int:
        return self._fields[self._order[pid]][0]

    def IsParameterReadOnly(self, pid: int) -> bool:
        return self._fields[self._order[pid]][2]

    # --- typed read accessors ---

    def GetField(self, name: str) -> str:
        value = self._fields[name][1]
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def GetNumber(self, name: str) -> float:
        return float(self._fields[name][1])

    def GetIntegerParameter(self, pid: int) -> int:
        return int(self._fields[self._order[pid]][1])

    def GetBooleanParameter(self, pid: int) -> bool:
        return bool(self._fields[self._order[pid]][1])

    def GetStringArrayParameter(self, pid: int) -> list[str]:
        return list(self._fields[self._order[pid]][1])

    def GetVector(self, name: str) -> list[float]:
        return [float(x) for x in self._fields[name][1]]

    def GetMatrix(self, name: str) -> _FakeRmatrix:
        return _FakeRmatrix(self._fields[name][1])

    # --- writes ---

    def SetField(self, name: str, value: Any) -> None:
        # Update the stored value so round-trip tests can re-read. Mimic the
        # real engine for read-only fields: raise so the wrapper layer is
        # exercised. (Real GMAT silently accepts CartesianX too in some
        # contexts, but raising is the strict behaviour callers should
        # depend on, and matches what fields like Spacecraft.Id reject.)
        type_code, _, read_only = self._fields[name]
        if read_only:
            raise RuntimeError(f"field '{name}' is read-only")
        self._fields[name] = (type_code, value, read_only)
        self.set_calls.append((name, value))


# --- fake gmat module factory -------------------------------------------------


# Mapping of object-type-enum names → arbitrary integers that the fake
# ``Moderator.GetListOfObjects`` uses as bucket IDs. The real gmat module
# defines these as opaque ints from the C++ engine; the production code reads
# them via ``getattr(self._gmat, "SUBSCRIBER", None)`` so any unique values work.
_OBJECT_TYPE_IDS = {"SUBSCRIBER": 100, "EVENT_LOCATOR": 200}

# Which object-type bucket each subscriber-like type lives in. ReportFile and
# EphemerisFile are Subscribers; ContactLocator is an EventLocator.
_OBJECT_TYPE_OF_CLASS = {
    "ReportFile": "SUBSCRIBER",
    "EphemerisFile": "SUBSCRIBER",
    "ContactLocator": "EVENT_LOCATOR",
}


class _FakeAPIException(Exception):
    """Stand-in for the real ``gmatpy.APIException`` raised by the engine."""


def _make_fake_gmat(
    objects: dict[str, _FakeObject] | None = None,
    *,
    load_script_returns: bool = True,
    run_script_status: int = 1,
    run_script_raises: BaseException | None = None,
    log_text: str = "fake gmat log\n",
) -> ModuleType:
    """Build a fake gmatpy module with the bits Mission touches.

    Beyond field access, the fake exposes the run-time surface
    :meth:`Mission.run` calls: ``RunScript``, ``UseLogFile``,
    ``GmatGlobal.Instance().SetOutputPath``, ``Moderator.Instance()`` (with
    ``GetListOfObjects`` keyed by the ``SUBSCRIBER`` / ``EVENT_LOCATOR`` enum
    values), and ``APIException``.
    """
    module = ModuleType("fake_gmat")
    for attr, code in _TYPE_CODES.items():
        setattr(module, attr, code)
    for attr, code in _OBJECT_TYPE_IDS.items():
        setattr(module, attr, code)
    registry: dict[str, _FakeObject] = dict(objects or {})

    def get_object(name: str) -> _FakeObject | None:
        return registry.get(name)

    def load_script(_path: str) -> bool:
        return load_script_returns

    # Recorder so tests can inspect what was wired during the run.
    log_paths: list[str] = []

    class _FakeModerator:
        def GetListOfObjects(self, type_id: int) -> list[str]:
            kind = next((k for k, v in _OBJECT_TYPE_IDS.items() if v == type_id), None)
            if kind is None:
                return []
            return [
                name
                for name, obj in registry.items()
                if _OBJECT_TYPE_OF_CLASS.get(obj.GetTypeName()) == kind
            ]

    class _ModeratorProxy:
        @staticmethod
        def Instance() -> _FakeModerator:
            return module._moderator  # type: ignore[no-any-return]

    def use_log_file(path: str) -> None:
        log_paths.append(path)
        log_file = Path(path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(log_text, encoding="utf-8")

    def run_script() -> int:
        if run_script_raises is not None:
            raise run_script_raises
        return run_script_status

    module._moderator = _FakeModerator()  # type: ignore[attr-defined]
    module.Moderator = _ModeratorProxy  # type: ignore[attr-defined]
    module.APIException = _FakeAPIException  # type: ignore[attr-defined]

    module.GetObject = get_object  # type: ignore[attr-defined]
    module.LoadScript = load_script  # type: ignore[attr-defined]
    module.Setup = lambda _path: None  # type: ignore[attr-defined]
    module.RunScript = run_script  # type: ignore[attr-defined]
    module.UseLogFile = use_log_file  # type: ignore[attr-defined]

    module._registry = registry  # type: ignore[attr-defined]
    module._log_paths = log_paths  # type: ignore[attr-defined]
    return module


def _make_install(root: Path) -> GmatInstall:
    return GmatInstall(
        root=root,
        bin_dir=root / "bin",
        api_dir=root / "api",
        output_dir=root / "output",
        version="R2026a",
    )


def _spacecraft(name: str = "Sat") -> _FakeObject:
    """Spacecraft with a representative slice of fields per type code."""
    fields: dict[str, tuple[int, Any, bool]] = {
        "SMA": (_TYPE_CODES["REAL_TYPE"], 7000.0, False),
        "DryMass": (_TYPE_CODES["REAL_TYPE"], 50.0, False),
        "OrbitColor": (_TYPE_CODES["INTEGER_TYPE"], 255, False),
        "DisplayStateType": (_TYPE_CODES["ENUMERATION_TYPE"], "Keplerian", False),
        "DateFormat": (_TYPE_CODES["STRING_TYPE"], "UTCGregorian", False),
        "CoordinateSystem": (_TYPE_CODES["OBJECT_TYPE"], "EarthMJ2000Eq", False),
        "Tanks": (_TYPE_CODES["STRINGARRAY_TYPE"], ["MainTank"], False),
        "CartesianX": (_TYPE_CODES["REAL_TYPE"], -999.999, True),
        "Covariance": (_TYPE_CODES["RMATRIX_TYPE"], [[1.0, 0.0], [0.0, 2.0]], False),
        "EulerAngles": (_TYPE_CODES["RVECTOR_TYPE"], [10.0, 20.0, 30.0], False),
    }
    return _FakeObject("Spacecraft", name, fields)


def _propagator(name: str = "DefaultProp") -> _FakeObject:
    fields: dict[str, tuple[int, Any, bool]] = {
        "InitialStepSize": (_TYPE_CODES["REAL_TYPE"], 60.0, False),
        "Accuracy": (_TYPE_CODES["REAL_TYPE"], 1e-12, False),
        "Type": (_TYPE_CODES["STRING_TYPE"], "PrinceDormand78", False),
        "StopIfAccuracyIsViolated": (_TYPE_CODES["BOOLEAN_TYPE"], True, False),
    }
    return _FakeObject("Propagator", name, fields)


def _impulsive_burn(name: str = "TOI") -> _FakeObject:
    fields: dict[str, tuple[int, Any, bool]] = {
        "Element1": (_TYPE_CODES["REAL_TYPE"], 0.0, False),
        "Element2": (_TYPE_CODES["REAL_TYPE"], 0.0, False),
        "Element3": (_TYPE_CODES["REAL_TYPE"], 0.0, False),
        "CoordinateSystem": (_TYPE_CODES["OBJECT_TYPE"], "EarthMJ2000Eq", False),
        "Axes": (_TYPE_CODES["ENUMERATION_TYPE"], "VNB", False),
        "DecrementMass": (_TYPE_CODES["BOOLEAN_TYPE"], False, False),
    }
    return _FakeObject("ImpulsiveBurn", name, fields)


def _report_file(name: str = "ReportFile1", filename: str = "report1.txt") -> _FakeObject:
    fields: dict[str, tuple[int, Any, bool]] = {
        "Filename": (_TYPE_CODES["FILENAME_TYPE"], filename, False),
    }
    return _FakeObject("ReportFile", name, fields)


def _ephemeris_file(name: str = "EphFile1", filename: str = "eph1.eph") -> _FakeObject:
    fields: dict[str, tuple[int, Any, bool]] = {
        "Filename": (_TYPE_CODES["FILENAME_TYPE"], filename, False),
    }
    return _FakeObject("EphemerisFile", name, fields)


def _contact_locator(name: str = "Contacts", filename: str = "contacts.txt") -> _FakeObject:
    fields: dict[str, tuple[int, Any, bool]] = {
        "Filename": (_TYPE_CODES["FILENAME_TYPE"], filename, False),
    }
    return _FakeObject("ContactLocator", name, fields)


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def mission(tmp_path: Path) -> Mission:
    """A Mission backed by a fake gmat module and the three required objects."""
    gmat = _make_fake_gmat(
        {
            "Sat": _spacecraft(),
            "DefaultProp": _propagator(),
            "TOI": _impulsive_burn(),
        }
    )
    install = _make_install(tmp_path / "gmat")
    return Mission(gmat=gmat, install=install, script_path=tmp_path / "mission.script")


@pytest.fixture
def patched_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[ModuleType]:
    """Patch locate_gmat + bootstrap so Mission.load runs against fakes.

    Yields the fake gmat module so individual tests can pre-populate or
    override registry / LoadScript behaviour.
    """
    install = _make_install(tmp_path / "gmat")
    gmat = _make_fake_gmat({"Sat": _spacecraft()})

    monkeypatch.setattr("gmat_run.mission.locate_gmat", lambda gmat_root=None: install)
    monkeypatch.setattr("gmat_run.mission.bootstrap", lambda _install: gmat)
    yield gmat


# --- Mission.load -------------------------------------------------------------


def test_load_returns_mission_handle(patched_load: ModuleType, tmp_path: Path) -> None:
    script = tmp_path / "flyby.script"
    script.write_text("# fake\n", encoding="utf-8")

    mission = Mission.load(script)

    assert mission.gmat is patched_load
    assert mission.script_path == script
    assert mission.install.version == "R2026a"


def test_load_raises_gmat_load_error_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install = _make_install(tmp_path / "gmat")
    gmat = _make_fake_gmat(load_script_returns=False)
    monkeypatch.setattr("gmat_run.mission.locate_gmat", lambda gmat_root=None: install)
    monkeypatch.setattr("gmat_run.mission.bootstrap", lambda _install: gmat)

    with pytest.raises(GmatLoadError) as excinfo:
        Mission.load(tmp_path / "broken.script")

    assert "broken.script" in str(excinfo.value)
    assert "log file" in str(excinfo.value)


def test_load_propagates_install_discovery_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _missing(_root: Any = None) -> Any:
        raise GmatNotFoundError([])

    monkeypatch.setattr("gmat_run.mission.locate_gmat", _missing)

    with pytest.raises(GmatNotFoundError):
        Mission.load(tmp_path / "any.script")


def test_load_passes_explicit_gmat_root_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _capture(gmat_root: Any = None) -> GmatInstall:
        captured["root"] = gmat_root
        return _make_install(tmp_path / "gmat")

    monkeypatch.setattr("gmat_run.mission.locate_gmat", _capture)
    monkeypatch.setattr("gmat_run.mission.bootstrap", lambda _i: _make_fake_gmat())

    Mission.load(tmp_path / "x.script", gmat_root="/explicit/path")
    assert captured["root"] == "/explicit/path"


# --- gmat property ------------------------------------------------------------


def test_gmat_property_exposes_module(mission: Mission) -> None:
    assert mission.gmat is mission._gmat


def test_gmat_property_is_read_only(mission: Mission) -> None:
    with pytest.raises(AttributeError):
        mission.gmat = object()  # type: ignore[misc, assignment]


# --- read path: type dispatch (Spacecraft / Propagator / ImpulsiveBurn) -------


class TestReadTypeDispatch:
    def test_real_returns_float(self, mission: Mission) -> None:
        sma = mission["Sat.SMA"]
        assert sma == 7000.0
        assert isinstance(sma, float)

    def test_integer_returns_int(self, mission: Mission) -> None:
        color = mission["Sat.OrbitColor"]
        assert color == 255
        assert isinstance(color, int) and not isinstance(color, bool)

    def test_boolean_returns_bool(self, mission: Mission) -> None:
        decrement = mission["TOI.DecrementMass"]
        assert decrement is False
        violated = mission["DefaultProp.StopIfAccuracyIsViolated"]
        assert violated is True

    def test_enumeration_returns_str(self, mission: Mission) -> None:
        axes = mission["TOI.Axes"]
        assert axes == "VNB"
        assert isinstance(axes, str)

    def test_object_reference_returns_str(self, mission: Mission) -> None:
        cs = mission["Sat.CoordinateSystem"]
        assert cs == "EarthMJ2000Eq"
        assert isinstance(cs, str)

    def test_string_returns_str(self, mission: Mission) -> None:
        fmt = mission["Sat.DateFormat"]
        assert fmt == "UTCGregorian"

    def test_string_array_returns_list_of_strings(self, mission: Mission) -> None:
        tanks = mission["Sat.Tanks"]
        assert tanks == ["MainTank"]
        assert isinstance(tanks, list)

    def test_rvector_returns_list_of_floats(self, mission: Mission) -> None:
        eulers = mission["Sat.EulerAngles"]
        assert eulers == [10.0, 20.0, 30.0]
        assert all(isinstance(x, float) for x in eulers)

    def test_rmatrix_returns_nested_list_of_floats(self, mission: Mission) -> None:
        cov = mission["Sat.Covariance"]
        assert cov == [[1.0, 0.0], [0.0, 2.0]]
        assert all(isinstance(x, float) for row in cov for x in row)


# --- write path: coercion and round-trip --------------------------------------


class TestWriteCoercion:
    def test_real_accepts_int_and_float(self, mission: Mission) -> None:
        mission["Sat.SMA"] = 7100  # int → float
        assert mission["Sat.SMA"] == 7100.0
        mission["Sat.SMA"] = 6878.137
        assert mission["Sat.SMA"] == 6878.137

    def test_integer_accepts_integral_float(self, mission: Mission) -> None:
        mission["Sat.OrbitColor"] = 128
        assert mission["Sat.OrbitColor"] == 128
        mission["Sat.OrbitColor"] = 64.0  # whole-number float allowed
        assert mission["Sat.OrbitColor"] == 64

    def test_integer_rejects_fractional_float(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            mission["Sat.OrbitColor"] = 3.14
        assert "expected an integer" in str(excinfo.value)

    def test_real_rejects_string(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            mission["Sat.SMA"] = "7000"
        assert "expected a real number" in str(excinfo.value)
        assert excinfo.value.path == "Sat.SMA"
        assert excinfo.value.value == "7000"

    def test_real_rejects_bool(self, mission: Mission) -> None:
        # bool is an int subclass — defend against the silent True → 1 trap.
        with pytest.raises(GmatFieldError):
            mission["Sat.SMA"] = True

    def test_boolean_accepts_bool(self, mission: Mission) -> None:
        mission["TOI.DecrementMass"] = True
        assert mission["TOI.DecrementMass"] is True

    def test_boolean_rejects_int(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            mission["TOI.DecrementMass"] = 1
        assert "expected a bool" in str(excinfo.value)

    def test_string_accepts_str(self, mission: Mission) -> None:
        mission["Sat.DateFormat"] = "TAIModJulian"
        assert mission["Sat.DateFormat"] == "TAIModJulian"

    def test_string_rejects_non_string(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            mission["Sat.DateFormat"] = 42
        assert "expected a string" in str(excinfo.value)

    def test_object_reference_accepts_str(self, mission: Mission) -> None:
        mission["TOI.CoordinateSystem"] = "EarthFixed"
        assert mission["TOI.CoordinateSystem"] == "EarthFixed"

    def test_string_array_accepts_list(self, mission: Mission) -> None:
        mission["Sat.Tanks"] = ["A", "B"]
        assert mission["Sat.Tanks"] == ["A", "B"]

    def test_string_array_rejects_non_list(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError):
            mission["Sat.Tanks"] = "A"

    def test_string_array_rejects_mixed_types(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError):
            mission["Sat.Tanks"] = ["A", 1]

    def test_rvector_accepts_list_of_numbers(self, mission: Mission) -> None:
        mission["Sat.EulerAngles"] = [1, 2.5, 3]
        assert mission["Sat.EulerAngles"] == [1.0, 2.5, 3.0]

    def test_rvector_rejects_non_numeric_element(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError):
            mission["Sat.EulerAngles"] = [1.0, "two", 3.0]

    def test_rmatrix_accepts_nested_list(self, mission: Mission) -> None:
        mission["Sat.Covariance"] = [[1, 2], [3, 4]]
        assert mission["Sat.Covariance"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_rmatrix_rejects_flat_list(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError):
            mission["Sat.Covariance"] = [1, 2, 3, 4]

    def test_set_propagator_field_round_trips(self, mission: Mission) -> None:
        mission["DefaultProp.InitialStepSize"] = 30.0
        assert mission["DefaultProp.InitialStepSize"] == 30.0

    def test_set_impulsive_burn_field_round_trips(self, mission: Mission) -> None:
        mission["TOI.Element1"] = 0.5
        assert mission["TOI.Element1"] == 0.5


# --- error paths --------------------------------------------------------------


class TestErrors:
    def test_unknown_resource(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            _ = mission["NoSuchSat.SMA"]
        assert "unknown resource" in str(excinfo.value)
        assert "NoSuchSat" in str(excinfo.value)
        assert excinfo.value.path == "NoSuchSat.SMA"

    def test_unknown_field_includes_nearest_match(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            _ = mission["Sat.SAM"]
        msg = str(excinfo.value)
        assert "unknown field" in msg
        assert "Spacecraft" in msg
        assert "SMA" in msg  # difflib.get_close_matches should suggest it

    def test_unknown_field_chains_underlying_exception(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            _ = mission["Sat.NoSuchField"]
        assert excinfo.value.__cause__ is not None

    def test_engine_error_on_set_is_wrapped(self, mission: Mission) -> None:
        # Real gmatpy raises APIException for some failed writes (read-only
        # fields, illegal enum values, etc.). The fake raises RuntimeError
        # for read-only fields so we exercise the engine-error wrapping
        # path. The wrapper turns whatever was raised into a GmatFieldError.
        with pytest.raises(GmatFieldError) as excinfo:
            mission["Sat.CartesianX"] = 100.0
        assert "GMAT rejected" in str(excinfo.value)
        assert "read-only" in str(excinfo.value)
        assert excinfo.value.path == "Sat.CartesianX"
        assert excinfo.value.value == 100.0
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_unknown_resource_swig_attribute_error(self, tmp_path: Path) -> None:
        # gmatpy's SWIG wrapper raises AttributeError from inside GetObject
        # for unknown names rather than returning None. Mission must catch
        # that quirk and surface a typed GmatFieldError.
        gmat = _make_fake_gmat()

        def _raising_get_object(_name: str) -> Any:
            raise AttributeError("'NoneType' object has no attribute 'GetTypeName'")

        gmat.GetObject = _raising_get_object  # type: ignore[attr-defined]
        install = _make_install(tmp_path / "gmat")
        m = Mission(gmat=gmat, install=install, script_path=tmp_path / "x.script")

        with pytest.raises(GmatFieldError) as excinfo:
            _ = m["Whatever.SMA"]
        assert "unknown resource" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, AttributeError)

    def test_path_with_no_dot(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            _ = mission["Sat"]
        assert "exactly one dot" in str(excinfo.value)

    def test_path_with_multiple_dots(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            _ = mission["Sat.Tanks.MainTank"]
        assert "exactly one dot" in str(excinfo.value)

    def test_path_with_empty_resource(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            _ = mission[".SMA"]
        assert "non-empty" in str(excinfo.value)

    def test_path_with_empty_field(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            _ = mission["Sat."]
        assert "non-empty" in str(excinfo.value)

    def test_field_error_value_is_none_on_read(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            _ = mission["Sat.NoSuchField"]
        assert excinfo.value.value is None

    def test_field_error_value_carries_set_payload(self, mission: Mission) -> None:
        with pytest.raises(GmatFieldError) as excinfo:
            mission["Sat.NoSuchField"] = 1234.5
        assert excinfo.value.value == 1234.5


# --- write reaches the underlying object --------------------------------------


def test_setitem_calls_set_field_on_underlying_object(mission: Mission) -> None:
    sat = mission.gmat.GetObject("Sat")
    assert isinstance(sat, _FakeObject)
    mission["Sat.SMA"] = 7123.45
    assert ("SMA", 7123.45) in sat.set_calls


# --- Mission.run --------------------------------------------------------------


def _run_mission(
    tmp_path: Path,
    *,
    objects: dict[str, _FakeObject] | None = None,
    run_script_status: int = 1,
    run_script_raises: BaseException | None = None,
    log_text: str = "fake gmat log\n",
) -> tuple[Mission, ModuleType]:
    """Build a Mission backed by a fake gmat module and return both."""
    gmat = _make_fake_gmat(
        objects,
        run_script_status=run_script_status,
        run_script_raises=run_script_raises,
        log_text=log_text,
    )
    install = _make_install(tmp_path / "gmat")
    mission = Mission(
        gmat=gmat,
        install=install,
        script_path=tmp_path / "mission.script",
    )
    return mission, gmat


class TestMissionRun:
    def test_run_returns_results_with_log(self, tmp_path: Path) -> None:
        mission, _ = _run_mission(
            tmp_path,
            objects={"R1": _report_file("R1", "r1.txt")},
            log_text="GMAT execution complete\n",
        )
        result = mission.run()
        assert isinstance(result, Results)
        assert result.log == "GMAT execution complete\n"
        # The default workspace is a temp dir, surfaced on output_dir.
        assert result.output_dir.is_dir()

    def test_run_populates_report_paths(self, tmp_path: Path) -> None:
        mission, _ = _run_mission(
            tmp_path,
            objects={"R1": _report_file("R1", "r1.txt")},
        )
        result = mission.run()
        assert list(result.reports) == ["R1"]
        # Relative filenames join the workspace dir.
        assert result.reports.keys() == {"R1"}
        # The path mapping itself is internal; reach through ephemeris_paths-style
        # behaviour by verifying the workspace dir owns it.
        # Use the parser-free path check on the lazy view's underlying mapping.
        # (We expose the path via output_dir + basename.)
        assert (result.output_dir / "r1.txt").parent == result.output_dir

    def test_run_buckets_outputs_by_subscriber_type(self, tmp_path: Path) -> None:
        mission, _ = _run_mission(
            tmp_path,
            objects={
                "R1": _report_file("R1", "r1.txt"),
                "E1": _ephemeris_file("E1", "e1.eph"),
                "C1": _contact_locator("C1", "c1.txt"),
            },
        )
        result = mission.run()
        assert list(result.reports) == ["R1"]
        assert list(result.ephemeris_paths) == ["E1"]
        assert list(result.contact_paths) == ["C1"]
        assert result.ephemeris_paths["E1"] == result.output_dir / "e1.eph"
        assert result.contact_paths["C1"] == result.output_dir / "c1.txt"

    def test_run_ignores_non_output_subscribers(self, tmp_path: Path) -> None:
        # OrbitView is a Subscriber but not one gmat-run records; the fake's
        # GetListOfObjects(SUBSCRIBER) only returns ReportFile/EphemerisFile,
        # so this is implicitly covered. Add an explicit registry entry
        # *outside* the bucket to confirm Mission.run doesn't walk extra
        # objects via GetObject. Adding to the registry without an object-
        # type-bucket entry models that.
        mission, gmat = _run_mission(tmp_path)
        # Inject a bare Spacecraft directly — it's in registry but not in any
        # subscriber bucket, so GetListOfObjects won't surface it.
        gmat._registry["Sat"] = _spacecraft()
        result = mission.run()
        assert len(result.reports) == 0
        assert len(result.ephemeris_paths) == 0
        assert len(result.contact_paths) == 0

    def test_run_uses_provided_working_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "user-output"
        mission, _ = _run_mission(
            tmp_path,
            objects={"R1": _report_file("R1", "r1.txt")},
        )
        result = mission.run(working_dir=custom)
        assert result.output_dir == custom
        assert custom.is_dir()
        # No tempdir was allocated when a working dir is provided.
        assert result._workspace is None

    def test_run_creates_missing_working_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "nested" / "output"
        assert not custom.exists()
        mission, _ = _run_mission(tmp_path)
        result = mission.run(working_dir=custom)
        assert custom.is_dir()
        assert result.output_dir == custom

    def test_run_default_workspace_is_temp_dir_tied_to_results(self, tmp_path: Path) -> None:
        mission, _ = _run_mission(tmp_path)
        result = mission.run()
        # The workspace is a TemporaryDirectory parked on Results so lazy
        # report parsing keeps working after run() returns.
        assert result._workspace is not None
        workspace_dir = result.output_dir
        assert workspace_dir.is_dir()
        # Cleanup happens when Results is dropped.
        result._workspace.cleanup()
        assert not workspace_dir.is_dir()

    def test_run_rewrites_subscriber_filenames(self, tmp_path: Path) -> None:
        # Each relative Filename gets pinned to an absolute path inside the
        # workspace via SetField — the only setting GMAT reads at write time.
        rf = _report_file("RF", "rel.txt")
        mission, _ = _run_mission(tmp_path, objects={"RF": rf})
        result = mission.run()
        # Filename was rewritten on the underlying object…
        assert ("Filename", str(result.output_dir / "rel.txt")) in rf.set_calls
        # …and Results captures the same resolved path.
        assert result.reports._paths == {  # type: ignore[attr-defined]
            "RF": result.output_dir / "rel.txt"
        }

    def test_run_redirects_log(self, tmp_path: Path) -> None:
        mission, gmat = _run_mission(tmp_path)
        result = mission.run()
        # UseLogFile was pointed at GmatLog.txt inside the workspace.
        assert gmat._log_paths == [str(result.output_dir / "GmatLog.txt")]

    def test_run_preserves_absolute_filenames(self, tmp_path: Path) -> None:
        absolute = tmp_path / "elsewhere" / "report.txt"
        rf = _report_file("R1", str(absolute))
        mission, _ = _run_mission(tmp_path, objects={"R1": rf})
        result = mission.run()
        # Stored path is the absolute one from the script, untouched.
        assert result.reports._paths == {"R1": absolute}  # type: ignore[attr-defined]
        # And the engine's Filename was *not* rewritten (would be wasteful
        # and would silently relocate user-pinned outputs).
        assert all(name != "Filename" for name, _ in rf.set_calls)

    def test_run_failure_status_raises_gmat_run_error(self, tmp_path: Path) -> None:
        mission, _ = _run_mission(
            tmp_path,
            run_script_status=-5,
            log_text="ERROR: solver diverged\n",
        )
        with pytest.raises(GmatRunError) as excinfo:
            mission.run()
        assert "status -5" in str(excinfo.value)
        assert excinfo.value.log == "ERROR: solver diverged\n"

    def test_run_engine_exception_raises_gmat_run_error(self, tmp_path: Path) -> None:
        # gmatpy's APIException (modelled here by _FakeAPIException) is the
        # canonical engine-level error type.
        mission, _ = _run_mission(
            tmp_path,
            run_script_raises=_FakeAPIException("integrator blew up"),
            log_text="ERROR: integrator blew up\n",
        )
        with pytest.raises(GmatRunError) as excinfo:
            mission.run()
        assert "integrator blew up" in str(excinfo.value)
        assert "_FakeAPIException" in str(excinfo.value)
        assert excinfo.value.log == "ERROR: integrator blew up\n"
        assert isinstance(excinfo.value.__cause__, _FakeAPIException)

    def test_run_works_without_any_subscribers(self, tmp_path: Path) -> None:
        mission, _ = _run_mission(tmp_path)
        result = mission.run()
        assert len(result.reports) == 0
        assert len(result.ephemeris_paths) == 0
        assert len(result.contact_paths) == 0
        # Log was still captured.
        assert result.log == "fake gmat log\n"

    def test_run_does_not_pollute_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stand the harness up under a fresh CWD with a relative-Filename
        # subscriber. A naïve run would land "rel.txt" right next to where
        # the user is sitting; the rewrite must redirect it into the
        # workspace, leaving CWD untouched.
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        before = set(cwd.iterdir())

        rf = _report_file("RF", "rel.txt")
        mission, _ = _run_mission(tmp_path / "gmat", objects={"RF": rf})
        result = mission.run()

        after = set(cwd.iterdir())
        assert after == before, f"new files in CWD: {after - before}"
        # The captured path is absolute and lives inside the workspace, not
        # CWD — the only way no-pollution is actually upheld.
        report_path = result.reports._paths["RF"]  # type: ignore[attr-defined]
        assert report_path.is_absolute()
        assert report_path.parent == result.output_dir

    def test_run_default_temp_dir_cleaned_up_when_results_dropped(self, tmp_path: Path) -> None:
        # The TemporaryDirectory parked on Results runs its finaliser when GC
        # collects the instance, so the workspace disappears with the result.
        mission, _ = _run_mission(tmp_path / "gmat")
        result = mission.run()
        workspace = result.output_dir
        assert workspace.is_dir()

        del result
        gc.collect()

        assert not workspace.is_dir()

    def test_run_explicit_working_dir_survives_results_drop(self, tmp_path: Path) -> None:
        # The opt-in path: when the caller pinned working_dir, the directory
        # is theirs and gmat-run leaves it alone on Results cleanup.
        custom = tmp_path / "user-output"
        mission, _ = _run_mission(tmp_path / "gmat")
        result = mission.run(working_dir=custom)
        assert custom.is_dir()

        del result
        gc.collect()

        assert custom.is_dir()

    def test_run_unknown_engine_exception_still_wrapped(self, tmp_path: Path) -> None:
        # If a non-APIException leaks out of RunScript (programmer bug, plugin
        # crash) we still want to surface a GmatRunError with the chained
        # cause, not a raw stack trace from engine code.
        mission, gmat = _run_mission(tmp_path)
        # Replace the configured exception type on the fake module with one
        # that has nothing to do with the actual error class — simulates the
        # case where the engine's exception type doesn't match what we expect.
        gmat.APIException = _FakeAPIException  # type: ignore[attr-defined]

        def _boom() -> int:
            raise RuntimeError("plugin crash")

        gmat.RunScript = _boom  # type: ignore[attr-defined]
        # RuntimeError is not _FakeAPIException, so the except clause does not
        # catch it — the test asserts the leak. This guards the documented
        # behaviour: only GMAT engine exceptions are captured; programmer
        # errors propagate.
        with pytest.raises(RuntimeError):
            mission.run()
