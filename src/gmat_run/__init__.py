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
from gmat_run.runtime import bootstrap

__version__ = "0.0.0"

__all__ = [
    "GmatError",
    "GmatFieldError",
    "GmatInstall",
    "GmatLoadError",
    "GmatNotFoundError",
    "GmatOutputParseError",
    "GmatRunError",
    "__version__",
    "bootstrap",
    "locate_gmat",
]
