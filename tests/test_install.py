"""Unit tests for gmat_run.install.

All tests fabricate fake install layouts under ``tmp_path``; no real GMAT
install is required.
"""

import shutil
from pathlib import Path
from typing import Any

import pytest

from gmat_run import install
from gmat_run.errors import GmatNotFoundError
from gmat_run.install import GmatInstall, locate_gmat


def _make_install(
    root: Path,
    *,
    version_in_startup: str | None = "R2026a",
    version_in_readme: str | None = None,
    has_bin: bool = True,
    has_gmatpy: bool = True,
    has_api: bool = True,
    has_load_gmat: bool = True,
) -> Path:
    """Fabricate a (possibly misshapen) GMAT install layout under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    if has_bin:
        bin_dir = root / "bin"
        bin_dir.mkdir(exist_ok=True)
        if has_gmatpy:
            (bin_dir / "gmatpy").mkdir(exist_ok=True)
        if version_in_startup is not None:
            (bin_dir / "gmat_startup_file.txt").write_text(
                f"# GMAT startup file (version {version_in_startup})\n",
                encoding="utf-8",
            )
        else:
            (bin_dir / "gmat_startup_file.txt").write_text(
                "# GMAT startup file with no version marker\n",
                encoding="utf-8",
            )
    if has_api:
        api_dir = root / "api"
        api_dir.mkdir(exist_ok=True)
        if has_load_gmat:
            (api_dir / "load_gmat.py").write_text("# fake load_gmat\n", encoding="utf-8")
    if version_in_readme is not None:
        (root / "README.txt").write_text(
            f"Welcome to GMAT Version {version_in_readme}\n",
            encoding="utf-8",
        )
    return root


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean GMAT_ROOT, no platform globs, and no PATH hits."""
    monkeypatch.delenv("GMAT_ROOT", raising=False)
    monkeypatch.setattr(install, "_platform_install_globs", list)
    monkeypatch.setattr(shutil, "which", lambda _name: None)


# --- argument override --------------------------------------------------------


def test_gmat_root_argument_happy_path(tmp_path: Path) -> None:
    root = _make_install(tmp_path / "gmat-R2026a")
    result = locate_gmat(gmat_root=root)
    assert isinstance(result, GmatInstall)
    assert result.root == root
    assert result.bin_dir == root / "bin"
    assert result.api_dir == root / "api"
    assert result.output_dir == root / "output"
    assert result.version == "R2026a"


def test_gmat_root_argument_accepts_string(tmp_path: Path) -> None:
    root = _make_install(tmp_path / "gmat-R2026a")
    result = locate_gmat(gmat_root=str(root))
    assert result.root == root


def test_gmat_root_argument_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_install(tmp_path / "gmat-R2026a")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    result = locate_gmat(gmat_root="~/gmat-R2026a")
    assert result.root == root


def test_gmat_root_argument_invalid_does_not_fall_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit override must fail loudly, not fall back to env/globs/PATH."""
    valid = _make_install(tmp_path / "valid")
    monkeypatch.setenv("GMAT_ROOT", str(valid))  # would succeed if we fell through
    with pytest.raises(GmatNotFoundError) as excinfo:
        locate_gmat(gmat_root=tmp_path / "does-not-exist")
    assert len(excinfo.value.attempts) == 1
    step, path, reason = excinfo.value.attempts[0]
    assert step == "gmat_root argument"
    assert path == tmp_path / "does-not-exist"
    assert reason == "does not exist"


# --- env var ------------------------------------------------------------------


def test_env_var_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_install(tmp_path / "gmat-R2026a")
    monkeypatch.setenv("GMAT_ROOT", str(root))
    result = locate_gmat()
    assert result.root == root


def test_env_var_invalid_does_not_fall_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = _make_install(tmp_path / "gmat-fallback")
    monkeypatch.setenv("GMAT_ROOT", str(tmp_path / "missing"))
    monkeypatch.setattr(install, "_platform_install_globs", lambda: [fallback])
    with pytest.raises(GmatNotFoundError) as excinfo:
        locate_gmat()
    # Fallback exists but must not be considered.
    assert fallback.exists()
    assert all(step == "GMAT_ROOT env var" for step, _, _ in excinfo.value.attempts)


def test_env_var_empty_string_treated_as_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GMAT_ROOT", "")
    with pytest.raises(GmatNotFoundError) as excinfo:
        locate_gmat()
    steps = [step for step, _, _ in excinfo.value.attempts]
    # Empty env var is reported as "not set" alongside the other empty steps.
    assert "GMAT_ROOT env var" in steps
    assert ("GMAT_ROOT env var", None, "not set") in excinfo.value.attempts


# --- platform globs -----------------------------------------------------------


def test_platform_glob_picks_lexically_greatest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    older = _make_install(tmp_path / "gmat-R2024a")
    middle = _make_install(tmp_path / "gmat-R2025b")
    newest = _make_install(tmp_path / "gmat-R2026a")
    # Pass them in arbitrary order to prove the function sorts, not the input.
    monkeypatch.setattr(install, "_platform_install_globs", lambda: [older, newest, middle])
    assert locate_gmat().root == newest


def test_platform_glob_skips_misshapen_and_picks_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Newer-named directory is misshapen; older one is valid and should win.
    broken = _make_install(tmp_path / "gmat-R2026a", has_gmatpy=False)
    valid = _make_install(tmp_path / "gmat-R2024a")
    monkeypatch.setattr(install, "_platform_install_globs", lambda: [broken, valid])
    assert locate_gmat().root == valid


def test_platform_glob_basename_sort_across_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sort by basename, not full path, so a path under /opt doesn't outrank ~/."""
    older = _make_install(tmp_path / "opt" / "gmat-R2024a")
    newer = _make_install(tmp_path / "home" / "gmat-R2026a")
    # Even though "/opt/..." sorts after "/home/..." lexically, basename order wins.
    monkeypatch.setattr(install, "_platform_install_globs", lambda: [older, newer])
    assert locate_gmat().root == newer


@pytest.mark.parametrize(
    ("platform", "expected_pattern_substr"),
    [
        ("win32", "Program Files"),
        ("linux", "gmat-"),
        ("darwin", "Applications"),
        ("freebsd", "gmat-"),  # falls through to Linux-style defaults
    ],
)
def test_glob_patterns_per_platform(platform: str, expected_pattern_substr: str) -> None:
    patterns = install._glob_patterns_for_platform(platform)
    assert patterns
    assert any(expected_pattern_substr in p for p in patterns)


# --- PATH ---------------------------------------------------------------------


def test_path_discovery_via_gmat_console(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_install(tmp_path / "gmat-R2026a")
    binary = root / "bin" / "GmatConsole"
    binary.touch()
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: str(binary) if name == "GmatConsole" else None,
    )
    assert locate_gmat().root == root


def test_path_discovery_skips_invalid_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Binary on PATH but its parent.parent is not a real install.
    bogus_bin = tmp_path / "elsewhere" / "bin"
    bogus_bin.mkdir(parents=True)
    binary = bogus_bin / "GmatConsole"
    binary.touch()
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: str(binary) if name == "GmatConsole" else None,
    )
    with pytest.raises(GmatNotFoundError) as excinfo:
        locate_gmat()
    assert any(step == "PATH" for step, _, _ in excinfo.value.attempts)


# --- version detection --------------------------------------------------------


def test_version_from_startup_file(tmp_path: Path) -> None:
    root = _make_install(tmp_path / "gmat", version_in_startup="R2025b")
    assert locate_gmat(gmat_root=root).version == "R2025b"


def test_version_from_readme_when_startup_lacks_marker(tmp_path: Path) -> None:
    root = _make_install(
        tmp_path / "gmat",
        version_in_startup=None,
        version_in_readme="R2024a",
    )
    assert locate_gmat(gmat_root=root).version == "R2024a"


def test_version_startup_wins_over_readme(tmp_path: Path) -> None:
    root = _make_install(
        tmp_path / "gmat",
        version_in_startup="R2026a",
        version_in_readme="R2020a",
    )
    assert locate_gmat(gmat_root=root).version == "R2026a"


def test_version_none_when_no_marker_anywhere(tmp_path: Path) -> None:
    root = _make_install(tmp_path / "gmat", version_in_startup=None, version_in_readme=None)
    assert locate_gmat(gmat_root=root).version is None


# --- misshapen installs -------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected_reason"),
    [
        ({"has_bin": False}, "missing bin/ directory"),
        ({"has_gmatpy": False}, "missing bin/gmatpy/ directory"),
        ({"has_load_gmat": False}, "missing api/load_gmat.py"),
        ({"has_api": False}, "missing api/load_gmat.py"),
    ],
)
def test_validate_install_rejects_misshapen(
    tmp_path: Path, kwargs: dict[str, Any], expected_reason: str
) -> None:
    root = _make_install(tmp_path / "broken", **kwargs)
    with pytest.raises(GmatNotFoundError) as excinfo:
        locate_gmat(gmat_root=root)
    assert excinfo.value.attempts[0][2] == expected_reason


def test_validate_install_rejects_nonexistent(tmp_path: Path) -> None:
    with pytest.raises(GmatNotFoundError) as excinfo:
        locate_gmat(gmat_root=tmp_path / "nope")
    assert excinfo.value.attempts[0][2] == "does not exist"


def test_validate_install_rejects_file(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("hi")
    with pytest.raises(GmatNotFoundError) as excinfo:
        locate_gmat(gmat_root=not_a_dir)
    assert excinfo.value.attempts[0][2] == "not a directory"


# --- error reporting ----------------------------------------------------------


def test_error_lists_every_search_step_when_all_empty() -> None:
    with pytest.raises(GmatNotFoundError) as excinfo:
        locate_gmat()
    steps = [step for step, _, _ in excinfo.value.attempts]
    assert steps == [
        "GMAT_ROOT env var",
        "platform install paths",
        "PATH",
    ]
    rendered = str(excinfo.value)
    assert "No usable GMAT install found" in rendered
    for step in steps:
        assert step in rendered


def test_error_message_includes_probed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    monkeypatch.setattr(install, "_platform_install_globs", lambda: [bad])
    with pytest.raises(GmatNotFoundError) as excinfo:
        locate_gmat()
    assert str(bad) in str(excinfo.value)
