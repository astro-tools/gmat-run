"""gmat-run: run GMAT mission scripts from Python and get results as pandas DataFrames."""

from gmat_run.errors import (
    GmatError,
    GmatFieldError,
    GmatLoadError,
    GmatNotFoundError,
    GmatOutputParseError,
    GmatRunError,
)
from gmat_run.install import GmatInstall, locate_gmat
from gmat_run.mission import Mission
from gmat_run.results import Results
from gmat_run.runtime import bootstrap

__version__ = "0.1.1"

__all__ = [
    "GmatError",
    "GmatFieldError",
    "GmatInstall",
    "GmatLoadError",
    "GmatNotFoundError",
    "GmatOutputParseError",
    "GmatRunError",
    "Mission",
    "Results",
    "__version__",
    "bootstrap",
    "locate_gmat",
]
