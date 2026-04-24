"""gmat-run: run GMAT mission scripts from Python and get results as pandas DataFrames."""

from gmat_run.errors import GmatNotFoundError, GmatRunError
from gmat_run.install import GmatInstall, locate_gmat

__version__ = "0.0.0"

__all__ = [
    "GmatInstall",
    "GmatNotFoundError",
    "GmatRunError",
    "__version__",
    "locate_gmat",
]
