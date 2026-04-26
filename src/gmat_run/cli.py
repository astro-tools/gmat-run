"""gmat-run command-line interface.

Minimal entry point: one subcommand, ``run``, plus the standard ``--help``
and ``--version`` flags. Wraps :class:`gmat_run.mission.Mission` and maps the
typed exceptions in :mod:`gmat_run.errors` onto stable exit codes so shell
scripts can branch on failure mode without parsing stderr.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

from gmat_run import __version__
from gmat_run.errors import (
    GmatLoadError,
    GmatNotFoundError,
    GmatOutputParseError,
    GmatRunError,
)
from gmat_run.mission import Mission

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["main"]

# Stable. The collision with argparse's own exit-2 (bad CLI args) is
# accepted: both signal "couldn't get going" — shell scripts can disambiguate
# from stderr if they care.
EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_NOT_FOUND = 2
EXIT_LOAD = 3
EXIT_RUN = 4
EXIT_PARSE = 5


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch the chosen subcommand.

    Returns the exit code; argparse handles ``--help`` and ``--version`` by
    raising :class:`SystemExit` itself, so callers that wrap this should not
    re-catch those.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run(script=args.script, out=args.out, gmat_root=args.gmat_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmat-run",
        description="Run GMAT mission scripts from Python.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"gmat-run {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run",
        help="Load a .script, run it, and print a summary.",
        description=(
            "Load SCRIPT, execute its mission sequence headlessly, and print "
            "a summary of the outputs to stdout. Use --out to copy the "
            "outputs out of the temporary workspace before it is cleaned up."
        ),
    )
    run_parser.add_argument("script", help="Path to a GMAT .script file.")
    run_parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="Persist the run's outputs into DIR (created if missing).",
    )
    run_parser.add_argument(
        "--gmat-root",
        metavar="PATH",
        default=None,
        help="Path to the GMAT install. Overrides discovery and $GMAT_ROOT.",
    )
    return parser


def _run(*, script: str, out: str | None, gmat_root: str | None) -> int:
    try:
        mission = Mission.load(script, gmat_root=gmat_root)
        result = mission.run()
        if out is not None:
            result.persist(out)
        _print_summary(
            output_dir=str(result.output_dir),
            reports=result.reports,
            ephemerides=result.ephemerides,
            contacts=result.contacts,
        )
    except GmatNotFoundError as exc:
        print(f"gmat-run: {exc}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except GmatLoadError as exc:
        print(f"gmat-run: {exc}", file=sys.stderr)
        return EXIT_LOAD
    except GmatRunError as exc:
        print(f"gmat-run: {exc}", file=sys.stderr)
        return EXIT_RUN
    except GmatOutputParseError as exc:
        print(f"gmat-run: {exc}", file=sys.stderr)
        return EXIT_PARSE
    except Exception as exc:
        print(f"gmat-run: unexpected error: {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED
    return EXIT_OK


def _print_summary(
    *,
    output_dir: str,
    reports: Mapping[str, pd.DataFrame],
    ephemerides: Mapping[str, pd.DataFrame],
    contacts: Mapping[str, pd.DataFrame],
) -> None:
    print(f"Output directory: {output_dir}")
    _print_section("Reports", reports)
    _print_section("Ephemerides", ephemerides)
    _print_section("Contacts", contacts)


def _print_section(label: str, frames: Mapping[str, pd.DataFrame]) -> None:
    if not frames:
        print(f"{label}: (none)")
        return
    print(f"{label}:")
    for name in frames:
        # Materialises the DataFrame on first access — accepted cost; the CLI
        # is the place to actually count rows. Callers that want the path
        # without parsing can still reach result.report_paths from Python.
        print(f"  {name}: {len(frames[name])} rows")
