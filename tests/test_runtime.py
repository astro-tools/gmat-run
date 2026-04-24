"""Unit tests for gmat_run.runtime.

All tests fabricate a fake install layout under ``tmp_path`` and a minimal
fake ``gmatpy`` package; no real GMAT install is required.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from gmat_run import runtime
from gmat_run.errors import GmatLoadError
from gmat_run.install import GmatInstall

# --- fixtures & helpers -------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_runtime() -> Iterator[None]:
    """Restore sys.path, sys.modules (gmatpy), and the module-level cache.

    bootstrap() mutates all three. Without this fixture, state leaks between
    tests and the fake gmatpy from one test is visible to the next.
    """

    def _drop_gmatpy() -> None:
        for mod in [m for m in sys.modules if m == "gmatpy" or m.startswith("gmatpy.")]:
            del sys.modules[mod]

    path_snapshot = sys.path.copy()
    runtime._bootstrapped = None
    _drop_gmatpy()
    yield
    _drop_gmatpy()
    sys.path[:] = path_snapshot
    runtime._bootstrapped = None


def _make_install(root: Path) -> GmatInstall:
    """Fabricate a minimal GMAT install layout and return a GmatInstall."""
    (root / "bin" / "gmatpy").mkdir(parents=True)
    (root / "api").mkdir()
    (root / "api" / "load_gmat.py").write_text("# fake load_gmat\n", encoding="utf-8")
    return GmatInstall(
        root=root,
        bin_dir=root / "bin",
        api_dir=root / "api",
        output_dir=root / "output",
        version="R2026a",
    )


def _write_fake_gmatpy(install: GmatInstall, *, import_error: str | None = None) -> None:
    """Plant a fake gmatpy package in ``install.bin_dir/gmatpy``."""
    init = install.bin_dir / "gmatpy" / "__init__.py"
    if import_error is not None:
        init.write_text(f"raise ImportError({import_error!r})\n", encoding="utf-8")
    else:
        init.write_text(
            "SETUP_CALLS = []\n"
            "def Setup(path):\n"
            "    SETUP_CALLS.append(path)\n",
            encoding="utf-8",
        )


def _write_fake_builder(
    install: GmatInstall,
    *,
    exit_code: int = 0,
    stderr_message: str = "builder failed",
) -> Path:
    """Write a stand-in for BuildApiStartupFile.py.

    Happy-path variant writes ``<bin>/api_startup_file.txt`` and increments a
    run counter (so tests can prove the builder ran exactly once). Failure
    variant prints ``stderr_message`` and exits non-zero.
    """
    script = install.api_dir / "BuildApiStartupFile.py"
    counter = install.root / "_builder_runs"
    if exit_code == 0:
        script.write_text(
            "import os\n"
            "here = os.path.dirname(os.path.abspath(__file__))\n"
            "root = os.path.dirname(here)\n"
            "with open(os.path.join(root, 'bin', 'api_startup_file.txt'), 'w') as f:\n"
            "    f.write('# fake api startup\\n')\n"
            f"counter = {str(counter)!r}\n"
            "prev = 0\n"
            "if os.path.isfile(counter):\n"
            "    with open(counter) as f: prev = int(f.read() or '0')\n"
            "with open(counter, 'w') as f: f.write(str(prev + 1))\n",
            encoding="utf-8",
        )
    else:
        script.write_text(
            f"import sys\nprint({stderr_message!r}, file=sys.stderr)\nsys.exit({exit_code})\n",
            encoding="utf-8",
        )
    return script


def _builder_run_count(install: GmatInstall) -> int:
    counter = install.root / "_builder_runs"
    if not counter.is_file():
        return 0
    return int(counter.read_text() or "0")


# --- happy path ---------------------------------------------------------------


def test_bootstrap_returns_gmatpy_module_with_setup_called(tmp_path: Path) -> None:
    install = _make_install(tmp_path / "gmat")
    _write_fake_gmatpy(install)
    (install.bin_dir / "api_startup_file.txt").write_text("# pre-existing\n")

    gmat = runtime.bootstrap(install)

    assert [str(install.bin_dir / "api_startup_file.txt")] == gmat.SETUP_CALLS
    assert str(install.bin_dir) in sys.path


def test_bootstrap_skips_builder_when_startup_file_exists(tmp_path: Path) -> None:
    install = _make_install(tmp_path / "gmat")
    _write_fake_gmatpy(install)
    _write_fake_builder(install)
    (install.bin_dir / "api_startup_file.txt").write_text("# pre-existing\n")

    runtime.bootstrap(install)

    assert _builder_run_count(install) == 0


# --- startup file generation --------------------------------------------------


def test_bootstrap_generates_startup_file_when_missing(tmp_path: Path) -> None:
    install = _make_install(tmp_path / "gmat")
    _write_fake_gmatpy(install)
    _write_fake_builder(install)

    runtime.bootstrap(install)

    startup_file = install.bin_dir / "api_startup_file.txt"
    assert startup_file.is_file()
    assert _builder_run_count(install) == 1


def test_bootstrap_raises_when_builder_fails(tmp_path: Path) -> None:
    install = _make_install(tmp_path / "gmat")
    _write_fake_gmatpy(install)
    _write_fake_builder(install, exit_code=2, stderr_message="no startup file")

    with pytest.raises(GmatLoadError) as excinfo:
        runtime.bootstrap(install)

    import subprocess

    assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)
    assert "no startup file" in str(excinfo.value)
    assert "BuildApiStartupFile.py" in str(excinfo.value)


# --- idempotency & single-install invariant -----------------------------------


def test_bootstrap_is_idempotent_for_same_install(tmp_path: Path) -> None:
    install = _make_install(tmp_path / "gmat")
    _write_fake_gmatpy(install)
    _write_fake_builder(install)

    first = runtime.bootstrap(install)
    second = runtime.bootstrap(install)

    assert first is second
    # Setup called exactly once despite two bootstrap calls.
    assert [str(install.bin_dir / "api_startup_file.txt")] == first.SETUP_CALLS
    # Builder ran exactly once (startup file cached after the first call).
    assert _builder_run_count(install) == 1


def test_bootstrap_rejects_second_install(tmp_path: Path) -> None:
    install_a = _make_install(tmp_path / "gmat-a")
    install_b = _make_install(tmp_path / "gmat-b")
    _write_fake_gmatpy(install_a)
    _write_fake_gmatpy(install_b)
    (install_a.bin_dir / "api_startup_file.txt").write_text("# a\n")
    (install_b.bin_dir / "api_startup_file.txt").write_text("# b\n")

    runtime.bootstrap(install_a)
    with pytest.raises(GmatLoadError) as excinfo:
        runtime.bootstrap(install_b)

    assert str(install_a.root) in str(excinfo.value)
    assert "second" in str(excinfo.value).lower()


# --- gmatpy import failure ----------------------------------------------------


def test_bootstrap_wraps_import_error_as_gmat_load_error(tmp_path: Path) -> None:
    install = _make_install(tmp_path / "gmat")
    _write_fake_gmatpy(install, import_error="py3.99 not supported")
    (install.bin_dir / "api_startup_file.txt").write_text("# ok\n")

    with pytest.raises(GmatLoadError) as excinfo:
        runtime.bootstrap(install)

    assert isinstance(excinfo.value.__cause__, ImportError)
    assert str(install.bin_dir) in str(excinfo.value)
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert py in str(excinfo.value)
