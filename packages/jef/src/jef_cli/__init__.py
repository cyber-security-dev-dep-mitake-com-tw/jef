"""JEF command line."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("jef")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"
__all__ = ["__version__"]
